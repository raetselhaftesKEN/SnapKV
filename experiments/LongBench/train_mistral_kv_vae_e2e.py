# -*- coding: utf-8 -*-
"""
Mistral e2e dual-forward KV-VAE training
---------------------------------------
Design:
1) raw forward (no_grad): collect ground-truth pre-RoPE KV at every layer
2) compressed forward: insert VAE into pre-RoPE KV path, reconstruct KV, and use reconstructed KV in attention
3) total loss = ntp_weight * LM_loss + rec_weight * KV_MSE + kl_weight * KL

This is the Mistral analogue of DeltaKV's qwen2_e2e training style,
but uses a VAE directly on original pre-RoPE KV instead of residual compression.
"""

import os
import math
import json
import argparse
from dataclasses import dataclass, asdict
from typing import Optional, Union, Tuple, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import load_from_disk, Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    #TrainingArguments,
    #Trainer,
    set_seed,
    default_data_collator,
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


def inject_extra_config(config, extra: ExtraVAEArgs):
    for k, v in asdict(extra).items():
        setattr(config, k, v)
    return config


# =========================================================
# Save trainable-only checkpoints
# =========================================================
def get_trainable_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    state = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            state[name] = param.detach().cpu()
    return state


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

        # encoder
        self.enc1 = nn.Linear(input_dim, hidden_dim)
        self.enc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu_proj = nn.Linear(hidden_dim, latent_dim)
        self.logvar_proj = nn.Linear(hidden_dim, latent_dim)

        # decoder
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


# =========================================================
# Simple eager attention for compatibility
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
    """
    query_states: [B, num_heads, Tq, Hd]
    key_states:   [B, num_heads, Tk, Hd]
    value_states: [B, num_heads, Tk, Hd]
    attention_mask: broadcastable to [B, num_heads, Tq, Tk]
    """
    attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2)) * scaling

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout_p, training=training)
    attn_output = torch.matmul(attn_weights, value_states)
    return attn_output, attn_weights


# =========================================================
# Mistral Attention with VAE
# =========================================================
class MistralAttentionVAE(MistralAttention):
    """
    Replace the original Mistral attention.
    We compress pre-RoPE KV with a VAE, reconstruct it, and use reconstructed KV for attention.
    """
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)

        self.layer_idx = layer_idx
        self.collect_kv_before_rope = getattr(config, "collect_kv_before_rope", True)
        assert self.collect_kv_before_rope, "This script only supports pre-RoPE KV VAE."

        self.split_kv = getattr(config, "split_kv", False)
        self.sample_during_train = getattr(config, "sample_during_train", True)
        self.deterministic_eval = getattr(config, "deterministic_eval", True)

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

        # buffers for loss
        self.buffer_raw_kv = None
        self.buffer_recon_kv = None
        self.buffer_kl = None

    def _vae_reconstruct(self, key_states_flat: torch.Tensor, value_states_flat: torch.Tensor):
        """
        key_states_flat:   [B, T, kv_dim]
        value_states_flat: [B, T, kv_dim]
        returns recon_k, recon_v with same shapes
        """
        deterministic = (not self.training and self.deterministic_eval) or (self.training and not self.sample_during_train)

        if self.split_kv:
            recon_k, mu_k, logvar_k, _ = self.k_vae(key_states_flat, deterministic=deterministic)
            recon_v, mu_v, logvar_v, _ = self.v_vae(value_states_flat, deterministic=deterministic)

            kl_k = kl_divergence(mu_k, logvar_k)  # [B, T]
            kl_v = kl_divergence(mu_v, logvar_v)  # [B, T]
            self.buffer_kl = (kl_k.mean() + kl_v.mean())
            recon_kv = torch.cat([recon_k, recon_v], dim=-1)
            self.buffer_recon_kv = recon_kv
            return recon_k, recon_v
        else:
            kv = torch.cat([key_states_flat, value_states_flat], dim=-1)  # [B, T, 2*kv_dim]
            recon_kv, mu, logvar, _ = self.kv_vae(kv, deterministic=deterministic)
            recon_k, recon_v = torch.split(recon_kv, key_states_flat.shape[-1], dim=-1)
            self.buffer_kl = kl_divergence(mu, logvar).mean()
            self.buffer_recon_kv = recon_kv
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

        # q/k/v projections
        query_states = self.q_proj(hidden_states)
        key_states_flat = self.k_proj(hidden_states)     # [B, T, kv_dim]
        value_states_flat = self.v_proj(hidden_states)   # [B, T, kv_dim]

        self.buffer_raw_kv = torch.cat([key_states_flat, value_states_flat], dim=-1)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states_flat.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states_flat.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # RoPE
        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        elif position_ids is not None:
            cos, sin = self.rotary_emb(value_states, position_ids)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        else:
            raise ValueError("Need either position_embeddings or position_ids for Mistral attention.")

        # training path assumes use_cache=False
        if use_cache or past_key_value is not None:
            raise NotImplementedError("This e2e training script does not support KV cache / generation.")

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_output, attn_weights = eager_mistral_attention(
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

        # compressed path
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states_flat = self.k_proj(hidden_states)     # [B, T, kv_dim]
        value_states_flat = self.v_proj(hidden_states)   # [B, T, kv_dim]

        recon_k_flat, recon_v_flat = self._vae_reconstruct(key_states_flat, value_states_flat)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = recon_k_flat.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = recon_v_flat.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        elif position_ids is not None:
            cos, sin = self.rotary_emb(value_states, position_ids)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        else:
            raise ValueError("Need either position_embeddings or position_ids for Mistral attention.")

        if use_cache or past_key_value is not None:
            raise NotImplementedError("This e2e training script does not support KV cache / generation.")

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_output, attn_weights = eager_mistral_attention(
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
        """
        Standard next-token loss.
        """
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        return loss

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

        #第一次前向，收集教师KV
        # 1) raw teacher forward (no grad) to collect ground-truth KV
        CURRENT_RUN_MODE = "raw"
        with torch.no_grad():
            _ = self.model(
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

        # 第二次前向，获得学生KV
        # 2) compressed student forward
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

        # 两次前向的loss
        lm_loss = self.compute_lm_loss(logits, labels)

        rec_loss = 0.0
        kl_loss = 0.0
        counted_layers = 0

        for mod in self.modules():
            if isinstance(mod, MistralAttentionVAE):
                if mod.buffer_raw_kv is None or mod.buffer_recon_kv is None or mod.buffer_kl is None:
                    continue
                rec_loss = rec_loss + self.mse(mod.buffer_recon_kv, mod.buffer_raw_kv)
                kl_loss = kl_loss + mod.buffer_kl
                counted_layers += 1

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

        CUR_STEP += 1
        if CUR_STEP % MODEL_LOG_STEPS == 0:
            print(
                f"[step {CUR_STEP}] total={total_loss.item():.6f} "
                f"lm={lm_loss.item():.6f} rec={rec_loss.item():.6f} kl={kl_loss.item():.6f}"
            )

        return CausalLMOutputWithPast(
            loss=total_loss,
            logits=logits,
            past_key_values=None,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# =========================================================
# Main
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser()

    # paths
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

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
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--bf16", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--fp16", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--gradient_checkpointing", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=-1)
    parser.add_argument("--max_train_samples", type=int, default=-1)

    return parser.parse_args()


def truncate_example(example, max_length: int):
    out = {}
    for k, v in example.items():
        if isinstance(v, list):
            out[k] = v[:max_length]
        else:
            out[k] = v
    return out

def dynamic_truncation_collator(features: List[Dict[str, Any]], max_length: int, pad_token_id: int):
    """
    Dynamically truncate each sample in the batch to max_length, then pad to batch max length.
    This avoids dataset.map(...) and therefore avoids writing a huge truncated Arrow cache to disk.
    """
    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []

    for f in features:
        input_ids = f["input_ids"][:max_length]
        attention_mask = f.get("attention_mask", [1] * len(input_ids))[:max_length]

        # labels default to input_ids for LM training
        labels = f.get("labels", input_ids.copy())[:max_length]

        batch_input_ids.append(torch.tensor(input_ids, dtype=torch.long))
        batch_attention_mask.append(torch.tensor(attention_mask, dtype=torch.long))
        batch_labels.append(torch.tensor(labels, dtype=torch.long))

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

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
    )
    config = inject_extra_config(raw_config, extra)

    model = MistralForCausalLMVAE.from_pretrained(
        args.model_name_or_path,
        config=config,
        torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32),
    )

    # freeze everything except VAE modules
    print("=== Trainable parameters ===")
    for name, param in model.named_parameters():
        if "vae" in name:
            param.requires_grad = True
            print(name)
        else:
            param.requires_grad = False
    print("============================")

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if os.path.exists(args.dataset_path):
        dataset = load_from_disk(args.dataset_path)
    else:
        dataset = load_dataset(args.dataset_path)

    if isinstance(dataset, Dataset):
        train_dataset = dataset
    elif "train" in dataset:
        train_dataset = dataset["train"]
    else:
        raise ValueError("Dataset must be a Dataset or a DatasetDict with 'train' split.")

    '''
    if args.max_length > 0:
        train_dataset = train_dataset.map(
            lambda x: truncate_example(x, args.max_length),
            num_proc=4,
            desc=f"truncate_to_{args.max_length}",
        )
    '''
    if args.max_train_samples > 0:
        max_n = min(args.max_train_samples, len(train_dataset))
        train_dataset = train_dataset.select(range(max_n))
        print(f"Using only the first {max_n} training samples.")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=lambda features: dynamic_truncation_collator(
            features,
            max_length=args.max_length if args.max_length > 0 else 10 ** 9,
            pad_token_id=tokenizer.pad_token_id,
        ),
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.0,
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_steps > 0:
        total_update_steps = args.max_steps
        num_epochs = math.ceil(args.max_steps / num_update_steps_per_epoch)
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

    use_autocast = args.bf16 or args.fp16
    autocast_dtype = torch.bfloat16 if args.bf16 else torch.float16

    global_step = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(num_epochs):
        for step, batch in enumerate(train_dataloader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

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
                        f"[epoch {epoch + 1}] step={global_step} loss={loss.item() * args.gradient_accumulation_steps:.6f} lr={lr:.6e}")

                if global_step % args.save_steps == 0:
                    save_trainable_checkpoint(
                        model,
                        args.output_dir,
                        global_step,
                        extra_config=asdict(extra),
                    )

                if args.max_steps > 0 and global_step >= args.max_steps:
                    break

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    save_trainable_checkpoint(
        model,
        args.output_dir,
        global_step,
        extra_config=asdict(extra),
    )
    print("Training finished.")

    # extra save
    with open(os.path.join(args.output_dir, "extra_vae_config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(extra), f, ensure_ascii=False, indent=2)

    print("Training finished.")


if __name__ == "__main__":
    main()

'''

python train_mistral_kv_vae_e2e.py \
  --model_name_or_path mistralai/mistral-7B-instruct-v0.2 \
  --dataset_path mikasenghaas/fineweb-edu-10bt-tokenized \
  --output_dir /home/ymz/SnapKV/SnapKV/experiments/LongBench/mistral_kv_vae_e2e \
  --kv_latent_size 64 \
  --vae_hidden_size 512 \
  --split_kv False \
  --kl_weight 1e-5 \
  --rec_weight 1.0 \
  --ntp_weight 1.0 \
  --sample_during_train True \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-4 \
  --warmup_ratio 0.03 \
  --logging_steps 10 \
  --save_steps 500 \
  --bf16 True \
  --max_length 2048
  
  
python train_mistral_kv_vae_e2e.py \
  --model_name_or_path mistralai/mistral-7B-instruct-v0.2 \
  --dataset_path mikasenghaas/fineweb-edu-10bt-tokenized \
  --output_dir /home/ymz/SnapKV/SnapKV/experiments/LongBench/mistral_kv_vae_e2e \
  --kv_latent_size 64 \
  --vae_hidden_size 512 \
  --split_kv False \
  --kl_weight 1e-5 \
  --rec_weight 1.0 \
  --ntp_weight 1.0 \
  --sample_during_train True \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-4 \
  --warmup_ratio 0.03 \
  --logging_steps 10 \
  --save_steps 500 \
  --bf16 True \
  --max_length 2048 \
  --max_train_samples 10000
  
'''