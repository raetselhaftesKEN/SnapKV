# -*- coding: utf-8 -*-
"""
Mistral e2e dual-forward KV reconstruction-first VAE training
--------------------------------------------------------------
Goal:
- maximize KV reconstruction fidelity
- use for KV cache compression/decompression evaluation
- NOT optimized for later predictor learning

Key design:
1) pre-RoPE KV only
2) K/V split
3) per-kv-head compression
4) almost deterministic bottleneck
5) reconstruction-first losses:
   total = ntp * lm + rec * mse + cos * cosine + rel_l2 * relative_l2 + tiny_kl * KL
6) raw teacher KV offloaded to CPU immediately to save VRAM
7) head-wise VAE processed in chunks to reduce peak memory
"""

import os
import math
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

from datasets import load_from_disk, DatasetDict, load_dataset
from transformers import AutoTokenizer, AutoConfig, set_seed
from transformers.models.mistral.modeling_mistral import (
    MistralAttention,
    MistralDecoderLayer,
    MistralModel,
    MistralForCausalLM,
    repeat_kv,
    apply_rotary_pos_emb,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast


# =========================================================
# Global run mode
# =========================================================
CURRENT_RUN_MODE = None
MODEL_LOG_STEPS = 10
CUR_STEP = 1


# =========================================================
# Config helpers
# =========================================================
@dataclass
class ExtraVAEArgs:
    per_head_latent_size: int = 32
    vae_hidden_size: int = 256
    split_kv: bool = True

    kl_weight: float = 1e-6
    kl_warmup_steps: int = 1000
    free_bits: float = 0.0

    rec_weight: float = 1.0
    ntp_weight: float = 1.0
    cos_weight: float = 0.25
    rel_l2_weight: float = 0.25

    collect_kv_before_rope: bool = True
    sample_during_train: bool = False
    deterministic_eval: bool = True

    use_vae: bool = True
    use_sdpa: bool = True

    share_vae_across_layers: bool = False
    vae_group_size: int = 1

    logvar_min: float = -8.0
    logvar_max: float = -2.0

    latent_stats_log_steps: int = 10
    latent_hist_save_steps: int = 100
    latent_hist_max_points: int = 4096
    latent_hist_dirname: str = "latent_stats"

    head_chunk_size: int = 1024


def inject_extra_config(config, extra: ExtraVAEArgs):
    for k, v in asdict(extra).items():
        setattr(config, k, v)
    return config


# =========================================================
# Save trainable-only checkpoints
# =========================================================
def save_trainable_checkpoint(model, output_dir, step, extra_config=None):
    ckpt_dir = os.path.join(output_dir, f"step_{step}")
    os.makedirs(ckpt_dir, exist_ok=True)

    trainable_state = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_state[name] = param.detach().cpu()

    torch.save(trainable_state, os.path.join(ckpt_dir, "trainable_vae_only.bin"))

    if hasattr(model, "config") and model.config is not None:
        model.config.save_pretrained(ckpt_dir)

    if extra_config is not None:
        with open(os.path.join(ckpt_dir, "extra_vae_config.json"), "w", encoding="utf-8") as f:
            json.dump(extra_config, f, ensure_ascii=False, indent=2)


# =========================================================
# Utility
# =========================================================
def tensor_basic_stats(x: torch.Tensor) -> Dict[str, float]:
    x = x.detach().float()
    return {
        "mean": x.mean().item(),
        "std": x.std(unbiased=False).item(),
        "min": x.min().item(),
        "max": x.max().item(),
        "abs_mean": x.abs().mean().item(),
    }


def sample_flat_values(x: torch.Tensor, max_points: int) -> torch.Tensor:
    flat = x.detach().reshape(-1)
    if flat.numel() <= max_points:
        return flat.float().cpu()
    idx = torch.randperm(flat.numel(), device=flat.device)[:max_points]
    return flat.index_select(0, idx).float().cpu()


def merge_weighted_stats(stats_list: List[Dict[str, float]]) -> Dict[str, float]:
    if not stats_list:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "abs_mean": 0.0, "numel": 0}

    total_n = sum(int(s["numel"]) for s in stats_list)
    if total_n == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "abs_mean": 0.0, "numel": 0}

    mean = sum(s["mean"] * s["numel"] for s in stats_list) / total_n
    second_moment = sum(((s["std"] ** 2) + (s["mean"] ** 2)) * s["numel"] for s in stats_list) / total_n
    var = max(0.0, second_moment - mean ** 2)
    abs_mean = sum(s["abs_mean"] * s["numel"] for s in stats_list) / total_n
    return {
        "mean": mean,
        "std": var ** 0.5,
        "min": min(s["min"] for s in stats_list),
        "max": max(s["max"] for s in stats_list),
        "abs_mean": abs_mean,
        "numel": total_n,
    }


def save_latent_snapshot(
    output_dir: str,
    dirname: str,
    step: int,
    mu_samples: List[torch.Tensor],
    logvar_samples: List[torch.Tensor],
    meta: Dict[str, Any],
):
    import numpy as np

    save_dir = Path(output_dir) / dirname
    save_dir.mkdir(parents=True, exist_ok=True)

    payload = {"meta": json.dumps(meta, ensure_ascii=False)}

    if mu_samples:
        payload["mu"] = torch.cat(mu_samples, dim=0).numpy()
    if logvar_samples:
        payload["logvar"] = torch.cat(logvar_samples, dim=0).numpy()

    np.savez_compressed(save_dir / f"latent_step_{step}.npz", **payload)


def cosine_recon_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred.float()
    target = target.float()
    cos = F.cosine_similarity(pred, target, dim=-1)
    return (1.0 - cos).mean()


def relative_l2_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = pred.float()
    target = target.float()
    num = (pred - target).pow(2).sum(dim=-1)
    den = target.pow(2).sum(dim=-1).clamp_min(eps)
    return (num / den).mean()


def rms_normalize_last_dim(x: torch.Tensor, eps: float = 1e-6):
    rms = x.pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()
    return x / rms, rms


# =========================================================
# KL helpers
# =========================================================
def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return kl.sum(dim=-1)


def apply_free_bits(kl_per_token: torch.Tensor, free_bits: float) -> torch.Tensor:
    if free_bits <= 0:
        return kl_per_token
    return torch.clamp(kl_per_token, min=free_bits)


# =========================================================
# Reconstruction-first VAE
# =========================================================
class HeadwiseVAECore(nn.Module):
    """
    Compress one KV head vector at a time.
    Input shape: [N, head_dim]
    """
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int, logvar_min: float = -8.0, logvar_max: float = -2.0):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max

        self.enc_in = nn.Linear(input_dim, hidden_dim)
        self.enc_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.enc_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.enc_ln = nn.LayerNorm(hidden_dim)

        self.mu_proj = nn.Linear(hidden_dim, latent_dim)
        self.logvar_proj = nn.Linear(hidden_dim, latent_dim)

        self.dec_in = nn.Linear(latent_dim, hidden_dim)
        self.dec_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.dec_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dec_ln = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: torch.Tensor):
        x_norm, x_rms = rms_normalize_last_dim(x)

        h = F.silu(self.enc_in(x_norm))
        h_res = h
        h = F.silu(self.enc_fc1(h))
        h = self.enc_fc2(h)
        h = self.enc_ln(h + h_res)
        h = F.silu(h)

        mu = self.mu_proj(h)
        logvar_raw = self.logvar_proj(h)
        logvar = torch.clamp(logvar_raw, min=self.logvar_min, max=self.logvar_max)
        return mu, logvar, x_rms

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor, training: bool = True):
        if not training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, x_rms: torch.Tensor):
        h = F.silu(self.dec_in(z))
        h_res = h
        h = F.silu(self.dec_fc1(h))
        h = self.dec_fc2(h)
        h = self.dec_ln(h + h_res)
        h = F.silu(h)
        out_norm = self.out_proj(h)
        out = out_norm * x_rms
        return out

    def forward(self, x: torch.Tensor, deterministic: bool = False):
        mu, logvar, x_rms = self.encode(x)
        z = self.reparameterize(mu, logvar, training=(self.training and not deterministic))
        recon = self.decode(z, x_rms)
        return recon, mu, logvar, z


class LayerwiseKVVAE(nn.Module):
    """
    Reconstruction-first KV VAE:
    - split K and V
    - compress each kv-head independently
    - process in chunks to reduce peak memory
    """
    def __init__(
        self,
        num_kv_heads: int,
        head_dim: int,
        per_head_latent_size: int,
        hidden_dim: int,
        logvar_min: float = -8.0,
        logvar_max: float = -2.0,
        chunk_size: int = 1024,
    ):
        super().__init__()
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.per_head_latent_size = per_head_latent_size
        self.chunk_size = chunk_size

        self.k_core = HeadwiseVAECore(
            input_dim=head_dim,
            latent_dim=per_head_latent_size,
            hidden_dim=hidden_dim,
            logvar_min=logvar_min,
            logvar_max=logvar_max,
        )
        self.v_core = HeadwiseVAECore(
            input_dim=head_dim,
            latent_dim=per_head_latent_size,
            hidden_dim=hidden_dim,
            logvar_min=logvar_min,
            logvar_max=logvar_max,
        )

    def _run_core(self, x_flat: torch.Tensor, core: nn.Module, deterministic: bool):
        bsz, seq_len, total_dim = x_flat.shape
        x = x_flat.view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        x = x.reshape(bsz * seq_len * self.num_kv_heads, self.head_dim)

        recons = []
        mus = []
        logvars = []

        for start in range(0, x.size(0), self.chunk_size):
            end = min(start + self.chunk_size, x.size(0))
            x_chunk = x[start:end]

            recon, mu, logvar, _ = core(x_chunk, deterministic=deterministic)

            recons.append(recon)
            mus.append(mu)
            logvars.append(logvar)

        recon = torch.cat(recons, dim=0)
        mu = torch.cat(mus, dim=0)
        logvar = torch.cat(logvars, dim=0)

        recon = recon.view(bsz, seq_len, self.num_kv_heads, self.head_dim).reshape(bsz, seq_len, total_dim)
        mu = mu.view(bsz, seq_len, self.num_kv_heads * self.per_head_latent_size)
        logvar = logvar.view(bsz, seq_len, self.num_kv_heads * self.per_head_latent_size)
        return recon, mu, logvar

    def forward(self, key_states_flat: torch.Tensor, value_states_flat: torch.Tensor, deterministic: bool = False):
        k_recon, k_mu, k_logvar = self._run_core(key_states_flat, self.k_core, deterministic=deterministic)
        v_recon, v_mu, v_logvar = self._run_core(value_states_flat, self.v_core, deterministic=deterministic)

        mu = torch.cat([k_mu, v_mu], dim=-1)
        logvar = torch.cat([k_logvar, v_logvar], dim=-1)
        return k_recon, v_recon, mu, logvar


class SharedVAEPool(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.group_size = int(getattr(config, "vae_group_size", 1))
        self.num_layers = int(config.num_hidden_layers)
        self.num_groups = math.ceil(self.num_layers / self.group_size)

        head_dim = config.hidden_size // config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        chunk_size = int(getattr(config, "head_chunk_size", 1024))

        self.group_vaes = nn.ModuleList([
            LayerwiseKVVAE(
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                per_head_latent_size=int(getattr(config, "per_head_latent_size", 32)),
                hidden_dim=int(getattr(config, "vae_hidden_size", 256)),
                logvar_min=float(getattr(config, "logvar_min", -8.0)),
                logvar_max=float(getattr(config, "logvar_max", -2.0)),
                chunk_size=chunk_size,
            )
            for _ in range(self.num_groups)
        ])

    def get_vae(self, layer_idx: int) -> nn.Module:
        return self.group_vaes[layer_idx // self.group_size]


# =========================================================
# Attention helpers
# =========================================================
def eager_mistral_attention(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout_p: float,
    training: bool,
    scaling: float,
):
    attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2)) * scaling

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    softmax_dtype = query_states.dtype if query_states.dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.float32
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=softmax_dtype).to(query_states.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout_p, training=training)
    attn_output = torch.matmul(attn_weights, value_states)
    return attn_output, attn_weights


def sdpa_mistral_attention(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout_p: float,
    training: bool,
    scaling: float,
):
    attn_output = F.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=attention_mask,
        dropout_p=dropout_p if training else 0.0,
        is_causal=False,
        scale=scaling,
    )
    return attn_output, None


# =========================================================
# Mistral Attention with VAE
# =========================================================
class MistralAttentionVAE(MistralAttention):
    def __init__(self, config, layer_idx: int, shared_vae: Optional[nn.Module] = None):
        super().__init__(config, layer_idx)

        self.config = config
        self.layer_idx = layer_idx
        self.collect_kv_before_rope = getattr(config, "collect_kv_before_rope", True)
        assert self.collect_kv_before_rope, "This script only supports pre-RoPE KV VAE."

        self.sample_during_train = getattr(config, "sample_during_train", False)
        self.deterministic_eval = getattr(config, "deterministic_eval", True)
        self.scaling = self.head_dim ** -0.5
        self.use_sdpa = getattr(config, "use_sdpa", True)

        if shared_vae is not None:
            self.kv_vae = shared_vae
        else:
            self.kv_vae = LayerwiseKVVAE(
                num_kv_heads=self.num_key_value_heads,
                head_dim=self.head_dim,
                per_head_latent_size=int(getattr(config, "per_head_latent_size", 32)),
                hidden_dim=int(getattr(config, "vae_hidden_size", 256)),
                logvar_min=float(getattr(config, "logvar_min", -8.0)),
                logvar_max=float(getattr(config, "logvar_max", -2.0)),
                chunk_size=int(getattr(config, "head_chunk_size", 1024)),
            )

        # CPU teacher buffers
        self.buffer_raw_k_cpu = None
        self.buffer_raw_v_cpu = None

        # scalar losses only
        self.buffer_rec_loss = None
        self.buffer_cos_loss = None
        self.buffer_rel_l2_loss = None
        self.buffer_kl = None

        # stats
        self.buffer_mu_stats = None
        self.buffer_logvar_stats = None
        self.buffer_mu_sample = None
        self.buffer_logvar_sample = None

    def _vae_reconstruct(self, key_states_flat: torch.Tensor, value_states_flat: torch.Tensor):
        deterministic = (not self.training and self.deterministic_eval) or (self.training and not self.sample_during_train)

        recon_k, recon_v, mu, logvar = self.kv_vae(
            key_states_flat,
            value_states_flat,
            deterministic=deterministic,
        )

        kl_raw = kl_divergence(mu, logvar)
        kl_fb = apply_free_bits(kl_raw, float(getattr(self.config, "free_bits", 0.0)))
        self.buffer_kl = kl_fb.mean()

        if self.buffer_raw_k_cpu is None or self.buffer_raw_v_cpu is None:
            raise RuntimeError(f"Layer {self.layer_idx}: raw KV buffers are missing before reconstruction loss computation.")

        raw_k = self.buffer_raw_k_cpu.to(device=recon_k.device, dtype=recon_k.dtype, non_blocking=True)
        raw_v = self.buffer_raw_v_cpu.to(device=recon_v.device, dtype=recon_v.dtype, non_blocking=True)

        self.buffer_rec_loss = 0.5 * (
            F.mse_loss(recon_k, raw_k) +
            F.mse_loss(recon_v, raw_v)
        )

        self.buffer_cos_loss = 0.5 * (
            cosine_recon_loss(recon_k, raw_k) +
            cosine_recon_loss(recon_v, raw_v)
        )

        self.buffer_rel_l2_loss = 0.5 * (
            relative_l2_loss(recon_k, raw_k) +
            relative_l2_loss(recon_v, raw_v)
        )

        mu_stats = tensor_basic_stats(mu)
        mu_stats["numel"] = mu.numel()
        logvar_stats = tensor_basic_stats(logvar)
        logvar_stats["numel"] = logvar.numel()
        self.buffer_mu_stats = mu_stats
        self.buffer_logvar_stats = logvar_stats

        max_points = int(getattr(self.config, "latent_hist_max_points", 4096))
        self.buffer_mu_sample = sample_flat_values(mu, max_points)
        self.buffer_logvar_sample = sample_flat_values(logvar, max_points)

        # release teacher KV immediately
        self.buffer_raw_k_cpu = None
        self.buffer_raw_v_cpu = None
        del raw_k, raw_v

        return recon_k, recon_v

    def raw_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states_flat = self.k_proj(hidden_states)
        value_states_flat = self.v_proj(hidden_states)

        # offload raw teacher KV to CPU immediately
        self.buffer_raw_k_cpu = key_states_flat.detach().to(torch.float16).cpu()
        self.buffer_raw_v_cpu = value_states_flat.detach().to(torch.float16).cpu()

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states_flat.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states_flat.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        elif position_ids is not None:
            kv_seq_len = int(position_ids.max().item()) + 1
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        else:
            raise ValueError("Need either position_embeddings or position_ids for Mistral attention.")

        if use_cache or past_key_value is not None:
            raise NotImplementedError("This training script does not support KV cache / generation.")

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_impl = eager_mistral_attention if output_attentions or not self.use_sdpa else sdpa_mistral_attention
        attn_output, attn_weights = attn_impl(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            attention_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            training=self.training,
            scaling=self.scaling,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        global CURRENT_RUN_MODE

        if CURRENT_RUN_MODE == "raw":
            return self.raw_forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states_flat = self.k_proj(hidden_states)
        value_states_flat = self.v_proj(hidden_states)

        recon_k_flat, recon_v_flat = self._vae_reconstruct(key_states_flat, value_states_flat)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = recon_k_flat.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = recon_v_flat.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        elif position_ids is not None:
            kv_seq_len = int(position_ids.max().item()) + 1
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        else:
            raise ValueError("Need either position_embeddings or position_ids for Mistral attention.")

        if use_cache or past_key_value is not None:
            raise NotImplementedError("This training script does not support KV cache / generation.")

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_impl = eager_mistral_attention if output_attentions or not self.use_sdpa else sdpa_mistral_attention
        attn_output, attn_weights = attn_impl(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            attention_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            training=self.training,
            scaling=self.scaling,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, None


# =========================================================
# Layer / Model replacement
# =========================================================
class MistralDecoderLayerVAE(MistralDecoderLayer):
    def __init__(self, config, layer_idx: int, shared_vae: Optional[nn.Module] = None):
        super().__init__(config, layer_idx)
        self.self_attn = MistralAttentionVAE(config=config, layer_idx=layer_idx, shared_vae=shared_vae)


class MistralModelVAE(MistralModel):
    def __init__(self, config):
        super().__init__(config)

        if getattr(config, "share_vae_across_layers", False):
            self.shared_vae_pool = SharedVAEPool(config)
            self.layers = nn.ModuleList([
                MistralDecoderLayerVAE(
                    config,
                    layer_idx=i,
                    shared_vae=self.shared_vae_pool.get_vae(i),
                )
                for i in range(config.num_hidden_layers)
            ])
        else:
            self.shared_vae_pool = None
            self.layers = nn.ModuleList([
                MistralDecoderLayerVAE(config, layer_idx=i, shared_vae=None)
                for i in range(config.num_hidden_layers)
            ])

        self.post_init()


class MistralForCausalLMVAE(MistralForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = MistralModelVAE(config)
        self.mse = nn.MSELoss()
        self.post_init()

    def compute_lm_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Any] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        global CURRENT_RUN_MODE, CUR_STEP

        if labels is None and input_ids is not None:
            labels = input_ids.clone()

        CURRENT_RUN_MODE = "raw"
        with torch.no_grad():
            _ = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=inputs_embeds,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
                **kwargs,
            )

        CURRENT_RUN_MODE = "comp"
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        lm_loss = self.compute_lm_loss(logits, labels)

        rec_loss = None
        cos_loss = None
        rel_l2 = None
        kl_loss = None
        counted_layers = 0

        mu_stats_list = []
        logvar_stats_list = []
        mu_samples = []
        logvar_samples = []

        for mod in self.modules():
            if isinstance(mod, MistralAttentionVAE):
                if mod.buffer_rec_loss is None:
                    continue

                rec_loss = mod.buffer_rec_loss if rec_loss is None else (rec_loss + mod.buffer_rec_loss)
                cos_loss = mod.buffer_cos_loss if cos_loss is None else (cos_loss + mod.buffer_cos_loss)
                rel_l2 = mod.buffer_rel_l2_loss if rel_l2 is None else (rel_l2 + mod.buffer_rel_l2_loss)
                kl_loss = mod.buffer_kl if kl_loss is None else (kl_loss + mod.buffer_kl)

                counted_layers += 1

                if mod.buffer_mu_stats is not None:
                    mu_stats_list.append(mod.buffer_mu_stats)
                if mod.buffer_logvar_stats is not None:
                    logvar_stats_list.append(mod.buffer_logvar_stats)
                if mod.buffer_mu_sample is not None:
                    mu_samples.append(mod.buffer_mu_sample)
                if mod.buffer_logvar_sample is not None:
                    logvar_samples.append(mod.buffer_logvar_sample)

                mod.buffer_rec_loss = None
                mod.buffer_cos_loss = None
                mod.buffer_rel_l2_loss = None
                mod.buffer_kl = None
                mod.buffer_mu_stats = None
                mod.buffer_logvar_stats = None
                mod.buffer_mu_sample = None
                mod.buffer_logvar_sample = None
                mod.buffer_raw_k_cpu = None
                mod.buffer_raw_v_cpu = None

        if counted_layers > 0:
            rec_loss = rec_loss / counted_layers
            cos_loss = cos_loss / counted_layers
            rel_l2 = rel_l2 / counted_layers
            kl_loss = kl_loss / counted_layers
        else:
            rec_loss = hidden_states.new_tensor(0.0)
            cos_loss = hidden_states.new_tensor(0.0)
            rel_l2 = hidden_states.new_tensor(0.0)
            kl_loss = hidden_states.new_tensor(0.0)

        warmup_steps = max(1, int(getattr(self.config, "kl_warmup_steps", 1000)))
        kl_scale = min(1.0, float(CUR_STEP) / warmup_steps)
        effective_kl_weight = float(self.config.kl_weight) * kl_scale

        total_loss = (
            self.config.ntp_weight * lm_loss
            + self.config.rec_weight * rec_loss
            + self.config.cos_weight * cos_loss
            + self.config.rel_l2_weight * rel_l2
            + effective_kl_weight * kl_loss
        )

        mu_stats = merge_weighted_stats(mu_stats_list)
        logvar_stats = merge_weighted_stats(logvar_stats_list)

        CUR_STEP += 1
        latent_log_every = max(1, int(getattr(self.config, "latent_stats_log_steps", MODEL_LOG_STEPS)))

        if CUR_STEP % MODEL_LOG_STEPS == 0:
            msg = (
                f"[model step {CUR_STEP}] total={total_loss.item():.6f} "
                f"lm={lm_loss.item():.6f} rec={rec_loss.item():.6f} "
                f"cos={cos_loss.item():.6f} rel={rel_l2.item():.6f} "
                f"kl={kl_loss.item():.6f} kl_w={effective_kl_weight:.8f}"
            )
            if CUR_STEP % latent_log_every == 0:
                msg += (
                    f" mu_mean={mu_stats['mean']:.6f} mu_std={mu_stats['std']:.6f}"
                    f" logvar_mean={logvar_stats['mean']:.6f} logvar_std={logvar_stats['std']:.6f}"
                )
            print(msg)

        save_every = int(getattr(self.config, "latent_hist_save_steps", 0))
        output_dir = getattr(self.config, "output_dir", None)
        if save_every > 0 and output_dir is not None and CUR_STEP % save_every == 0:
            save_latent_snapshot(
                output_dir=output_dir,
                dirname=getattr(self.config, "latent_hist_dirname", "latent_stats"),
                step=CUR_STEP,
                mu_samples=mu_samples,
                logvar_samples=logvar_samples,
                meta={
                    "step": CUR_STEP,
                    "mu_stats": mu_stats,
                    "logvar_stats": logvar_stats,
                    "counted_layers": counted_layers,
                    "effective_kl_weight": effective_kl_weight,
                },
            )

        return CausalLMOutputWithPast(
            loss=total_loss,
            logits=logits,
            past_key_values=None,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# =========================================================
# Data utilities
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser()

    # paths
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--output_dir", type=str, required=True)

    # data
    parser.add_argument("--streaming", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--max_length", type=int, default=768)
    parser.add_argument("--max_train_samples", type=int, default=-1)
    parser.add_argument("--tokenized_dataset", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--num_workers", type=int, default=0)

    # vae
    parser.add_argument("--per_head_latent_size", type=int, default=32)
    parser.add_argument("--vae_hidden_size", type=int, default=256)
    parser.add_argument("--split_kv", type=lambda x: str(x).lower() == "true", default=True)

    parser.add_argument("--kl_weight", type=float, default=1e-6)
    parser.add_argument("--kl_warmup_steps", type=int, default=1000)
    parser.add_argument("--free_bits", type=float, default=0.0)

    parser.add_argument("--rec_weight", type=float, default=1.0)
    parser.add_argument("--ntp_weight", type=float, default=1.0)
    parser.add_argument("--cos_weight", type=float, default=0.25)
    parser.add_argument("--rel_l2_weight", type=float, default=0.25)

    parser.add_argument("--sample_during_train", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--deterministic_eval", type=lambda x: str(x).lower() == "true", default=True)

    parser.add_argument("--share_vae_across_layers", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--vae_group_size", type=int, default=1)
    parser.add_argument("--logvar_min", type=float, default=-8.0)
    parser.add_argument("--logvar_max", type=float, default=-2.0)

    parser.add_argument("--latent_stats_log_steps", type=int, default=10)
    parser.add_argument("--latent_hist_save_steps", type=int, default=100)
    parser.add_argument("--latent_hist_max_points", type=int, default=4096)
    parser.add_argument("--latent_hist_dirname", type=str, default="latent_stats")

    parser.add_argument("--head_chunk_size", type=int, default=1024)

    # train
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--bf16", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--fp16", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--gradient_checkpointing", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--use_sdpa", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def is_pretokenized_dataset(dataset_obj, text_column: str) -> bool:
    column_names = None
    if hasattr(dataset_obj, "column_names"):
        column_names = dataset_obj.column_names
    if isinstance(column_names, dict):
        values = list(column_names.values())
        column_names = values[0] if len(values) > 0 else []
    if column_names is None:
        return False
    return ("input_ids" in column_names) and (text_column not in column_names)


def load_training_source(args):
    if os.path.exists(args.dataset_path):
        ds = load_from_disk(args.dataset_path)
        if isinstance(ds, DatasetDict):
            if args.dataset_split not in ds:
                raise ValueError(f"Split '{args.dataset_split}' not found in local DatasetDict.")
            ds = ds[args.dataset_split]
        return ds

    return load_dataset(
        args.dataset_path,
        args.dataset_config_name,
        split=args.dataset_split,
        streaming=args.streaming,
    )


class PackedTextIterableDataset(IterableDataset):
    def __init__(self, hf_dataset, tokenizer, text_column: str, block_size: int, add_eos_token: bool = True, max_samples: int = -1):
        super().__init__()
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.block_size = block_size
        self.add_eos_token = add_eos_token
        self.max_samples = max_samples

    def __iter__(self):
        token_buffer = []
        yielded = 0

        eos_id = self.tokenizer.eos_token_id
        if eos_id is None:
            raise ValueError("Tokenizer must have eos_token_id for packed text training.")

        for example in self.hf_dataset:
            if self.max_samples > 0 and yielded >= self.max_samples:
                break

            if self.text_column not in example:
                raise KeyError(f"text_column='{self.text_column}' not found. Available keys: {list(example.keys())}")

            text = example[self.text_column]
            if text is None:
                continue
            if not isinstance(text, str):
                text = str(text)
            if len(text.strip()) == 0:
                continue

            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            if self.add_eos_token:
                token_ids = token_ids + [eos_id]
            if len(token_ids) == 0:
                continue

            token_buffer.extend(token_ids)

            while len(token_buffer) >= self.block_size:
                block = token_buffer[: self.block_size]
                token_buffer = token_buffer[self.block_size :]
                yielded += 1
                yield {
                    "input_ids": block,
                    "attention_mask": [1] * self.block_size,
                    "labels": block.copy(),
                }
                if self.max_samples > 0 and yielded >= self.max_samples:
                    break


class PretokenizedTruncationIterableDataset(IterableDataset):
    def __init__(self, hf_dataset, block_size: int, max_samples: int = -1):
        super().__init__()
        self.hf_dataset = hf_dataset
        self.block_size = block_size
        self.max_samples = max_samples

    def __iter__(self):
        yielded = 0
        for example in self.hf_dataset:
            if self.max_samples > 0 and yielded >= self.max_samples:
                break

            input_ids = example["input_ids"][: self.block_size]
            if len(input_ids) == 0:
                continue
            attention_mask = example.get("attention_mask", [1] * len(input_ids))[: self.block_size]
            labels = example.get("labels", input_ids.copy())[: self.block_size]

            yielded += 1
            yield {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }


def lm_collator(features: List[Dict[str, Any]], pad_token_id: int):
    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []

    for f in features:
        batch_input_ids.append(torch.tensor(f["input_ids"], dtype=torch.long))
        batch_attention_mask.append(torch.tensor(f["attention_mask"], dtype=torch.long))
        batch_labels.append(torch.tensor(f["labels"], dtype=torch.long))

    max_len_in_batch = max(x.size(0) for x in batch_input_ids)

    padded_input_ids = []
    padded_attention_mask = []
    padded_labels = []

    for input_ids, attention_mask, labels in zip(batch_input_ids, batch_attention_mask, batch_labels):
        pad_len = max_len_in_batch - input_ids.size(0)
        if pad_len > 0:
            input_ids = F.pad(input_ids, (0, pad_len), value=pad_token_id)
            attention_mask = F.pad(attention_mask, (0, pad_len), value=0)
            labels = F.pad(labels, (0, pad_len), value=-100)

        padded_input_ids.append(input_ids)
        padded_attention_mask.append(attention_mask)
        padded_labels.append(labels)

    return {
        "input_ids": torch.stack(padded_input_ids, dim=0),
        "attention_mask": torch.stack(padded_attention_mask, dim=0),
        "labels": torch.stack(padded_labels, dim=0),
    }


def build_training_dataset(args, tokenizer):
    source = load_training_source(args)

    tokenized_mode = args.tokenized_dataset or is_pretokenized_dataset(source, args.text_column)

    if tokenized_mode:
        print("[Data] Detected/forced pre-tokenized dataset mode.")
        dataset = PretokenizedTruncationIterableDataset(
            source,
            block_size=args.max_length,
            max_samples=args.max_train_samples,
        )
        approx_len = None if args.streaming else (min(len(source), args.max_train_samples) if args.max_train_samples > 0 else len(source))
        return dataset, approx_len, tokenized_mode

    print("[Data] Using raw-text dataset mode with online tokenization + packing.")
    dataset = PackedTextIterableDataset(
        source,
        tokenizer=tokenizer,
        text_column=args.text_column,
        block_size=args.max_length,
        add_eos_token=True,
        max_samples=args.max_train_samples,
    )
    return dataset, None, tokenized_mode


# =========================================================
# Main
# =========================================================
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    if args.max_length <= 1:
        raise ValueError("--max_length must be > 1.")
    if args.bf16 and args.fp16:
        raise ValueError("Choose only one of --bf16 and --fp16.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    raw_config = AutoConfig.from_pretrained(args.model_name_or_path)
    extra = ExtraVAEArgs(
        per_head_latent_size=args.per_head_latent_size,
        vae_hidden_size=args.vae_hidden_size,
        split_kv=args.split_kv,
        kl_weight=args.kl_weight,
        kl_warmup_steps=args.kl_warmup_steps,
        free_bits=args.free_bits,
        rec_weight=args.rec_weight,
        ntp_weight=args.ntp_weight,
        cos_weight=args.cos_weight,
        rel_l2_weight=args.rel_l2_weight,
        collect_kv_before_rope=True,
        sample_during_train=args.sample_during_train,
        deterministic_eval=args.deterministic_eval,
        use_vae=True,
        use_sdpa=args.use_sdpa,
        share_vae_across_layers=args.share_vae_across_layers,
        vae_group_size=args.vae_group_size,
        logvar_min=args.logvar_min,
        logvar_max=args.logvar_max,
        latent_stats_log_steps=args.latent_stats_log_steps,
        latent_hist_save_steps=args.latent_hist_save_steps,
        latent_hist_max_points=args.latent_hist_max_points,
        latent_hist_dirname=args.latent_hist_dirname,
        head_chunk_size=args.head_chunk_size,
    )
    config = inject_extra_config(raw_config, extra)
    config.output_dir = args.output_dir

    model = MistralForCausalLMVAE.from_pretrained(
        args.model_name_or_path,
        config=config,
        torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32),
    )
    model.config.use_cache = False

    print("=== Trainable parameters ===")
    for name, param in model.named_parameters():
        if "vae" in name or "shared_vae_pool" in name:
            param.requires_grad = True
            print(name)
        else:
            param.requires_grad = False
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable params: {total_trainable:,}")
    print("============================")

    # IMPORTANT:
    # This reconstruction-first dual-forward script uses cross-pass layer buffers
    # (raw teacher KV saved in raw pass, consumed in comp pass).
    # Gradient checkpointing recomputes layer forwards during backward and breaks
    # this assumption, causing "raw KV buffers are missing" errors.
    if args.gradient_checkpointing:
        print("[Warning] gradient_checkpointing is incompatible with this dual-forward KV reconstruction script.")
        print("[Warning] It is being disabled automatically.")
    args.gradient_checkpointing = False

    train_dataset, approx_len, tokenized_mode = build_training_dataset(args, tokenizer)

    print(f"[Data] dataset_path={args.dataset_path}")
    print(f"[Data] dataset_config_name={args.dataset_config_name}")
    print(f"[Data] dataset_split={args.dataset_split}")
    print(f"[Data] tokenized_mode={tokenized_mode}")
    print(f"[Data] streaming={args.streaming}")
    print(f"[Data] block_size(max_length)={args.max_length}")
    if approx_len is not None:
        print(f"[Data] approx_num_examples={approx_len}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=lambda features: lm_collator(features, pad_token_id=tokenizer.pad_token_id),
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.0,
    )

    if approx_len is not None:
        num_update_steps_per_epoch = math.ceil(
            approx_len / (args.per_device_train_batch_size * args.gradient_accumulation_steps)
        )
    else:
        if args.max_steps <= 0:
            raise ValueError("For streaming/raw packed datasets with unknown length, set --max_steps > 0.")
        num_update_steps_per_epoch = args.max_steps

    if args.max_steps > 0:
        total_update_steps = args.max_steps
        num_epochs = max(1, math.ceil(args.max_steps / max(1, num_update_steps_per_epoch)))
    else:
        num_epochs = args.num_train_epochs
        total_update_steps = num_epochs * num_update_steps_per_epoch

    warmup_steps = int(total_update_steps * args.warmup_ratio)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / max(1, warmup_steps)
        return max(0.0, float(total_update_steps - current_step) / max(1, total_update_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    use_autocast = (args.bf16 or args.fp16) and torch.cuda.is_available()
    autocast_dtype = torch.bfloat16 if args.bf16 else torch.float16

    if torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    vocab_size = model.config.vocab_size

    stop_training = False
    for epoch in range(num_epochs):
        for step, batch in enumerate(train_dataloader):
            batch = {k: v.to(device, non_blocking=torch.cuda.is_available()) for k, v in batch.items()}

            input_min = batch["input_ids"].min().item()
            input_max = batch["input_ids"].max().item()
            if input_min < 0 or input_max >= vocab_size:
                raise ValueError(f"input_ids out of range: min={input_min}, max={input_max}, vocab_size={vocab_size}")

            valid_labels = batch["labels"][batch["labels"] != -100]
            if valid_labels.numel() > 0:
                label_min = valid_labels.min().item()
                label_max = valid_labels.max().item()
                if label_min < 0 or label_max >= vocab_size:
                    raise ValueError(f"labels out of range: min={label_min}, max={label_max}, vocab_size={vocab_size}")

            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=use_autocast):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    use_cache=False,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
                loss = outputs.loss / args.gradient_accumulation_steps

            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % args.logging_steps == 0:
                    lr = scheduler.get_last_lr()[0]
                    print(f"[epoch {epoch + 1}] step={global_step} loss={loss.item() * args.gradient_accumulation_steps:.6f} lr={lr:.6e}")

                if global_step % args.save_steps == 0:
                    save_trainable_checkpoint(
                        model,
                        args.output_dir,
                        global_step,
                        extra_config=asdict(extra),
                    )

                if args.max_steps > 0 and global_step >= args.max_steps:
                    stop_training = True
                    break

        if stop_training:
            break

    save_trainable_checkpoint(
        model,
        args.output_dir,
        global_step,
        extra_config=asdict(extra),
    )

    with open(os.path.join(args.output_dir, "extra_vae_config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(extra), f, ensure_ascii=False, indent=2)

    print("Training finished.")


if __name__ == "__main__":
    main()

'''
nohup env PYTORCH_ALLOC_CONF=expandable_segments:True \
python train_mistral_kv_recon_vae.py \
  --model_name_or_path mistralai/mistral-7B-instruct-v0.2 \
  --dataset_path Salesforce/wikitext \
  --dataset_config_name wikitext-103-raw-v1 \
  --dataset_split train \
  --text_column text \
  --output_dir /home/ymz/SnapKV/SnapKV/experiments/LongBench/mistral_kv_recon_vae \
  --per_head_latent_size 32 \
  --vae_hidden_size 192 \
  --split_kv True \
  --kl_weight 1e-6 \
  --kl_warmup_steps 1000 \
  --free_bits 0.0 \
  --rec_weight 1.0 \
  --ntp_weight 1.0 \
  --cos_weight 0.25 \
  --rel_l2_weight 0.25 \
  --sample_during_train False \
  --deterministic_eval True \
  --share_vae_across_layers False \
  --vae_group_size 1 \
  --logvar_min -8.0 \
  --logvar_max -2.0 \
  --head_chunk_size 256 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-4 \
  --warmup_ratio 0.03 \
  --logging_steps 10 \
  --save_steps 500 \
  --bf16 True \
  --gradient_checkpointing False \
  --use_sdpa True \
  --max_length 512 \
  --max_steps 2000 \
  --latent_stats_log_steps 10 \
  --latent_hist_save_steps 100 \
> vae_train_20250429.log 2>&1 &
'''