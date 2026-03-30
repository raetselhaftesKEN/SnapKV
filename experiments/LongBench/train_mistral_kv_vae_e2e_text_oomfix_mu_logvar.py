# -*- coding: utf-8 -*-
"""
Mistral e2e dual-forward KV-VAE training (text-dataset version)
----------------------------------------------------------------
Supports raw text datasets such as:
- Salesforce/wikitext (e.g. wikitext-103-raw-v1)
- HuggingFaceFW/fineweb / fineweb-edu

Key changes compared with the old version:
1) Supports text-only datasets directly.
2) Uses the current model tokenizer online, so token ids always match the model vocab.
3) Supports optional streaming datasets.
4) Packs raw text into fixed-length LM training blocks on the fly, which is much more suitable
   for large corpora like FineWeb.
5) Still supports pre-tokenized datasets containing input_ids.

Design:
1) raw forward (no_grad): collect ground-truth pre-RoPE KV at every layer
2) compressed forward: insert VAE into pre-RoPE KV path, reconstruct KV, and use reconstructed KV in attention
3) total loss = ntp_weight * LM_loss + rec_weight * KV_MSE + kl_weight * KL
"""

import os
import math
import json
import argparse
import random
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Union, Tuple, Dict, Any, List, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

from datasets import load_from_disk, Dataset, DatasetDict, IterableDataset as HFIterableDataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    set_seed,
)

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
    kv_latent_size: int = 64
    vae_hidden_size: int = 512
    split_kv: bool = False

    kl_weight: float = 1e-4
    rec_weight: float = 1.0
    ntp_weight: float = 1.0

    collect_kv_before_rope: bool = True
    sample_during_train: bool = True
    deterministic_eval: bool = True

    use_vae: bool = True
    use_sdpa: bool = True

    latent_stats_log_steps: int = 10
    latent_hist_save_steps: int = 100
    latent_hist_max_points: int = 4096
    latent_hist_dirname: str = "latent_stats"


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
# VAE
# =========================================================
class TokenVAE(nn.Module):
    """
    Token-level VAE for a flattened KV vector.
    Input:  [B, T, D]
    Output: recon [B, T, D]
    """
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.enc1 = nn.Linear(input_dim, hidden_dim)
        self.enc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu_proj = nn.Linear(hidden_dim, latent_dim)
        self.logvar_proj = nn.Linear(hidden_dim, latent_dim)

        self.dec1 = nn.Linear(latent_dim, hidden_dim)
        self.dec2 = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = F.silu(self.enc1(x))
        h = F.silu(self.enc2(h))
        mu = self.mu_proj(h)
        logvar = self.logvar_proj(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor, training: bool = True) -> torch.Tensor:
        if not training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.dec1(z))
        h = F.silu(self.dec2(h))
        return self.out_proj(h)

    def forward(self, x: torch.Tensor, deterministic: bool = False):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar, training=(self.training and not deterministic))
        recon = self.decode(z)
        return recon, mu, logvar, z


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Return KL per token, shape [B, T]
    """
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return kl.sum(dim=-1)


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


def save_latent_snapshot(output_dir: str, dirname: str, step: int, mu_samples: List[torch.Tensor], logvar_samples: List[torch.Tensor], meta: Dict[str, Any]):
    save_dir = Path(output_dir) / dirname
    save_dir.mkdir(parents=True, exist_ok=True)
    mu = torch.cat(mu_samples, dim=0).numpy() if mu_samples else None
    logvar = torch.cat(logvar_samples, dim=0).numpy() if logvar_samples else None
    payload = {"meta": json.dumps(meta, ensure_ascii=False)}
    if mu is not None:
        payload["mu"] = mu
    if logvar is not None:
        payload["logvar"] = logvar
    import numpy as np
    np.savez_compressed(save_dir / f"latent_step_{step}.npz", **payload)


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
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)

        self.layer_idx = layer_idx
        self.collect_kv_before_rope = getattr(config, "collect_kv_before_rope", True)
        assert self.collect_kv_before_rope, "This script only supports pre-RoPE KV VAE."

        self.split_kv = getattr(config, "split_kv", False)
        self.sample_during_train = getattr(config, "sample_during_train", True)
        self.deterministic_eval = getattr(config, "deterministic_eval", True)
        self.scaling = self.head_dim ** -0.5
        self.use_sdpa = getattr(config, "use_sdpa", True)

        kv_dim = self.num_key_value_heads * self.head_dim

        if self.split_kv:
            self.k_vae = TokenVAE(
                input_dim=kv_dim,
                latent_dim=config.kv_latent_size,
                hidden_dim=config.vae_hidden_size,
            )
            self.v_vae = TokenVAE(
                input_dim=kv_dim,
                latent_dim=config.kv_latent_size,
                hidden_dim=config.vae_hidden_size,
            )
        else:
            self.kv_vae = TokenVAE(
                input_dim=2 * kv_dim,
                latent_dim=config.kv_latent_size,
                hidden_dim=config.vae_hidden_size,
            )

        self.buffer_raw_kv = None
        self.buffer_recon_kv = None
        self.buffer_kl = None
        self.buffer_mu_stats = None
        self.buffer_logvar_stats = None
        self.buffer_mu_sample = None
        self.buffer_logvar_sample = None

    def _vae_reconstruct(self, key_states_flat: torch.Tensor, value_states_flat: torch.Tensor):
        deterministic = (not self.training and self.deterministic_eval) or (
            self.training and not self.sample_during_train
        )

        sample_points = getattr(self.config, "latent_hist_max_points", 4096)

        if self.split_kv:
            recon_k, mu_k, logvar_k, _ = self.k_vae(key_states_flat, deterministic=deterministic)
            recon_v, mu_v, logvar_v, _ = self.v_vae(value_states_flat, deterministic=deterministic)

            mu_all = torch.cat([mu_k, mu_v], dim=-1)
            logvar_all = torch.cat([logvar_k, logvar_v], dim=-1)

            kl_k = kl_divergence(mu_k, logvar_k)
            kl_v = kl_divergence(mu_v, logvar_v)
            self.buffer_kl = (kl_k.mean() + kl_v.mean())
            recon_kv = torch.cat([recon_k, recon_v], dim=-1)
            self.buffer_recon_kv = recon_kv
            mu_stats = tensor_basic_stats(mu_all)
            mu_stats["numel"] = mu_all.numel()
            logvar_stats = tensor_basic_stats(logvar_all)
            logvar_stats["numel"] = logvar_all.numel()
            self.buffer_mu_stats = mu_stats
            self.buffer_logvar_stats = logvar_stats
            self.buffer_mu_sample = sample_flat_values(mu_all, sample_points)
            self.buffer_logvar_sample = sample_flat_values(logvar_all, sample_points)
            return recon_k, recon_v
        else:
            kv = torch.cat([key_states_flat, value_states_flat], dim=-1)
            recon_kv, mu, logvar, _ = self.kv_vae(kv, deterministic=deterministic)
            recon_k, recon_v = torch.split(recon_kv, key_states_flat.shape[-1], dim=-1)
            self.buffer_kl = kl_divergence(mu, logvar).mean()
            self.buffer_recon_kv = recon_kv
            mu_stats = tensor_basic_stats(mu)
            mu_stats["numel"] = mu.numel()
            logvar_stats = tensor_basic_stats(logvar)
            logvar_stats["numel"] = logvar.numel()
            self.buffer_mu_stats = mu_stats
            self.buffer_logvar_stats = logvar_stats
            self.buffer_mu_sample = sample_flat_values(mu, sample_points)
            self.buffer_logvar_sample = sample_flat_values(logvar, sample_points)
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

        self.buffer_raw_kv = torch.cat([key_states_flat, value_states_flat], dim=-1)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states_flat.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states_flat.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids
            )
        elif position_ids is not None:
            kv_seq_len = int(position_ids.max().item()) + 1
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids
            )
        else:
            raise ValueError("Need either position_embeddings or position_ids for Mistral attention.")

        if use_cache or past_key_value is not None:
            raise NotImplementedError("This e2e training script does not support KV cache / generation.")

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
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids
            )
        elif position_ids is not None:
            kv_seq_len = int(position_ids.max().item()) + 1
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids
            )
        else:
            raise ValueError("Need either position_embeddings or position_ids for Mistral attention.")

        if use_cache or past_key_value is not None:
            raise NotImplementedError("This e2e training script does not support KV cache / generation.")

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
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = MistralAttentionVAE(config=config, layer_idx=layer_idx)


class MistralModelVAE(MistralModel):
    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [MistralDecoderLayerVAE(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )
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
        kl_loss = None
        counted_layers = 0
        mu_stats_list = []
        logvar_stats_list = []
        mu_samples = []
        logvar_samples = []

        for mod in self.modules():
            if isinstance(mod, MistralAttentionVAE):
                if mod.buffer_raw_kv is None or mod.buffer_recon_kv is None or mod.buffer_kl is None:
                    continue
                layer_rec = self.mse(mod.buffer_recon_kv, mod.buffer_raw_kv)
                layer_kl = mod.buffer_kl
                rec_loss = layer_rec if rec_loss is None else (rec_loss + layer_rec)
                kl_loss = layer_kl if kl_loss is None else (kl_loss + layer_kl)
                counted_layers += 1
                if mod.buffer_mu_stats is not None:
                    mu_stats_list.append(mod.buffer_mu_stats)
                if mod.buffer_logvar_stats is not None:
                    logvar_stats_list.append(mod.buffer_logvar_stats)
                if mod.buffer_mu_sample is not None:
                    mu_samples.append(mod.buffer_mu_sample)
                if mod.buffer_logvar_sample is not None:
                    logvar_samples.append(mod.buffer_logvar_sample)
                mod.buffer_raw_kv = None
                mod.buffer_recon_kv = None
                mod.buffer_kl = None
                mod.buffer_mu_stats = None
                mod.buffer_logvar_stats = None
                mod.buffer_mu_sample = None
                mod.buffer_logvar_sample = None

        if counted_layers > 0:
            rec_loss = rec_loss / counted_layers
            kl_loss = kl_loss / counted_layers
        else:
            rec_loss = torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)
            kl_loss = torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)

        total_loss = (
            self.config.ntp_weight * lm_loss
            + self.config.rec_weight * rec_loss
            + self.config.kl_weight * kl_loss
        )

        mu_stats = merge_weighted_stats(mu_stats_list)
        logvar_stats = merge_weighted_stats(logvar_stats_list)

        CUR_STEP += 1
        latent_log_every = max(1, int(getattr(self.config, "latent_stats_log_steps", MODEL_LOG_STEPS)))
        if CUR_STEP % MODEL_LOG_STEPS == 0:
            base_msg = (
                f"[model step {CUR_STEP}] total={total_loss.item():.6f} "
                f"lm={lm_loss.item():.6f} rec={rec_loss.item():.6f} kl={kl_loss.item():.6f}"
            )
            if CUR_STEP % latent_log_every == 0:
                base_msg += (
                    f" mu_mean={mu_stats['mean']:.6f} mu_std={mu_stats['std']:.6f}"
                    f" logvar_mean={logvar_stats['mean']:.6f} logvar_std={logvar_stats['std']:.6f}"
                )
            print(base_msg)

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

    # data behavior
    parser.add_argument("--streaming", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_train_samples", type=int, default=-1)
    parser.add_argument("--tokenized_dataset", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--num_workers", type=int, default=0)

    # vae
    parser.add_argument("--kv_latent_size", type=int, default=64)
    parser.add_argument("--vae_hidden_size", type=int, default=512)
    parser.add_argument("--split_kv", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--kl_weight", type=float, default=1e-4)
    parser.add_argument("--rec_weight", type=float, default=1.0)
    parser.add_argument("--ntp_weight", type=float, default=1.0)
    parser.add_argument("--sample_during_train", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--deterministic_eval", type=lambda x: str(x).lower() == "true", default=True)

    # train
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--bf16", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--fp16", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--gradient_checkpointing", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--use_sdpa", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--latent_stats_log_steps", type=int, default=10)
    parser.add_argument("--latent_hist_save_steps", type=int, default=100)
    parser.add_argument("--latent_hist_max_points", type=int, default=4096)
    parser.add_argument("--latent_hist_dirname", type=str, default="latent_stats")
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
    """
    Turn a raw text dataset into fixed-length token blocks on the fly.

    Each yielded example is:
      {
        "input_ids": [...],
        "attention_mask": [...],
        "labels": [...],
      }
    with length == block_size.
    """
    def __init__(
        self,
        hf_dataset,
        tokenizer,
        text_column: str,
        block_size: int,
        add_eos_token: bool = True,
        max_samples: int = -1,
    ):
        super().__init__()
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.block_size = block_size
        self.add_eos_token = add_eos_token
        self.max_samples = max_samples

    def __iter__(self):
        token_buffer: List[int] = []
        yielded = 0

        eos_id = self.tokenizer.eos_token_id
        if eos_id is None:
            raise ValueError("Tokenizer must have eos_token_id for packed text training.")

        for example in self.hf_dataset:
            if self.max_samples > 0 and yielded >= self.max_samples:
                break

            if self.text_column not in example:
                raise KeyError(
                    f"text_column='{self.text_column}' not found in dataset example. "
                    f"Available keys: {list(example.keys())}"
                )

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
    """
    Wrap a pre-tokenized dataset and yield examples up to block_size.
    Useful for compatibility, but raw text datasets are preferred.
    """
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
        raise ValueError("--max_length must be > 1 for causal LM training.")
    if args.bf16 and args.fp16:
        raise ValueError("Choose only one of --bf16 and --fp16.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    raw_config = AutoConfig.from_pretrained(args.model_name_or_path)
    extra = ExtraVAEArgs(
        kv_latent_size=args.kv_latent_size,
        vae_hidden_size=args.vae_hidden_size,
        split_kv=args.split_kv,
        kl_weight=args.kl_weight,
        rec_weight=args.rec_weight,
        ntp_weight=args.ntp_weight,
        collect_kv_before_rope=True,
        sample_during_train=args.sample_during_train,
        deterministic_eval=args.deterministic_eval,
        use_vae=True,
        use_sdpa=args.use_sdpa,
        latent_stats_log_steps=args.latent_stats_log_steps,
        latent_hist_save_steps=args.latent_hist_save_steps,
        latent_hist_max_points=args.latent_hist_max_points,
        latent_hist_dirname=args.latent_hist_dirname,
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
    trainable_names = []
    for name, param in model.named_parameters():
        if "vae" in name:
            param.requires_grad = True
            trainable_names.append(name)
            print(name)
        else:
            param.requires_grad = False
    print(f"Total trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print("============================")

    if args.gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

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
            raise ValueError(
                "For streaming/raw packed datasets with unknown length, please set --max_steps > 0."
            )
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
        return max(
            0.0,
            float(total_update_steps - current_step) / max(1, total_update_steps - warmup_steps)
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    use_autocast = (args.bf16 or args.fp16) and torch.cuda.is_available()
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
    autocast_dtype = torch.bfloat16 if args.bf16 else torch.float16

    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    vocab_size = model.config.vocab_size

    stop_training = False
    for epoch in range(num_epochs):
        for step, batch in enumerate(train_dataloader):
            batch = {k: v.to(device, non_blocking=torch.cuda.is_available()) for k, v in batch.items()}

            # Hard range checks to catch tokenizer/data mismatch early.
            input_min = batch["input_ids"].min().item()
            input_max = batch["input_ids"].max().item()
            if input_min < 0 or input_max >= vocab_size:
                raise ValueError(
                    f"input_ids out of range: min={input_min}, max={input_max}, vocab_size={vocab_size}"
                )
            valid_labels = batch["labels"][batch["labels"] != -100]
            if valid_labels.numel() > 0:
                label_min = valid_labels.min().item()
                label_max = valid_labels.max().item()
                if label_min < 0 or label_max >= vocab_size:
                    raise ValueError(
                        f"labels out of range: min={label_min}, max={label_max}, vocab_size={vocab_size}"
                    )

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
                    print(
                        f"[epoch {epoch + 1}] step={global_step} "
                        f"loss={loss.item() * args.gradient_accumulation_steps:.6f} lr={lr:.6e}"
                    )

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

"""
PYTORCH_ALLOC_CONF=expandable_segments:True \
python train_mistral_kv_vae_e2e_text_oomfix_mu_logvar.py \
  --model_name_or_path mistralai/mistral-7B-instruct-v0.2 \
  --dataset_path Salesforce/wikitext \
  --dataset_config_name wikitext-103-raw-v1 \
  --dataset_split train \
  --text_column text \
  --output_dir /home/ymz/SnapKV/SnapKV/experiments/LongBench/mistral_kv_vae_e2e_wikitext_20260330 \
  --kv_latent_size 64 \
  --vae_hidden_size 512 \
  --split_kv False \
  --kl_weight 1e-5 \
  --rec_weight 1.0 \
  --ntp_weight 1.0 \
  --sample_during_train True \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-4 \
  --warmup_ratio 0.03 \
  --logging_steps 10 \
  --save_steps 500 \
  --bf16 True \
  --gradient_checkpointing True \
  --use_sdpa True \
  --max_length 768 \
  --max_steps 1000 \
  --latent_stats_log_steps 10 \
  --latent_hist_save_steps 100 \
  --latent_hist_max_points 4096 \
  --latent_hist_dirname latent_stats
"""
