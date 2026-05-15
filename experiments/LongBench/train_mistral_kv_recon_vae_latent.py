# -*- coding: utf-8 -*-
"""
Mistral e2e dual-forward KV reconstruction + shared latent space VAE
Adapted from train_mistral_kv_recon_vae.py with Latent Space KV Alignment enhancements
"""
import os, math, json, argparse
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

    rec_weight: float = 0.5
    ntp_weight: float = 2.0  # LM prioritized
    cos_weight: float = 0.0
    rel_l2_weight: float = 0.0

    collect_kv_before_rope: bool = True
    sample_during_train: bool = False
    deterministic_eval: bool = True

    use_vae: bool = True
    use_sdpa: bool = True

    share_vae_across_layers: bool = True
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

def save_trainable_checkpoint(model, output_dir, step, extra_config=None):
    ckpt_dir = os.path.join(output_dir, f"step_{step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    trainable_state = {name: p.detach().cpu() for name, p in model.named_parameters() if p.requires_grad}
    torch.save(trainable_state, os.path.join(ckpt_dir, "trainable_vae_only.bin"))
    if hasattr(model, "config") and model.config is not None:
        model.config.save_pretrained(ckpt_dir)
    if extra_config is not None:
        with open(os.path.join(ckpt_dir, "extra_vae_config.json"), "w", encoding="utf-8") as f:
            json.dump(extra_config, f, ensure_ascii=False, indent=2)

# =========================================================
# VAE modules (Headwise + Layerwise + Shared Pool)
# =========================================================
def rms_normalize_last_dim(x: torch.Tensor, eps: float = 1e-6):
    rms = x.pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()
    return x / rms, rms

def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(dim=-1)

def apply_free_bits(kl_per_token: torch.Tensor, free_bits: float) -> torch.Tensor:
    return torch.clamp(kl_per_token, min=free_bits) if free_bits > 0 else kl_per_token

class HeadwiseVAECore(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, logvar_min=-8.0, logvar_max=-2.0):
        super().__init__()
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
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max

    def encode(self, x):
        x_norm, x_rms = rms_normalize_last_dim(x)
        h = F.silu(self.enc_in(x_norm))
        h_res = h
        h = F.silu(self.enc_fc1(h))
        h = self.enc_fc2(h)
        h = self.enc_ln(h + h_res)
        h = F.silu(h)
        mu = self.mu_proj(h)
        logvar = torch.clamp(self.logvar_proj(h), min=self.logvar_min, max=self.logvar_max)
        return mu, logvar, x_rms

    def reparameterize(self, mu, logvar, training=True):
        if not training:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z, x_rms):
        h = F.silu(self.dec_in(z))
        h_res = h
        h = F.silu(self.dec_fc1(h))
        h = self.dec_fc2(h)
        h = self.dec_ln(h + h_res)
        h = F.silu(h)
        return self.out_proj(h) * x_rms

    def forward(self, x, deterministic=False):
        mu, logvar, x_rms = self.encode(x)
        z = self.reparameterize(mu, logvar, training=(self.training and not deterministic))
        recon = self.decode(z, x_rms)
        return recon, mu, logvar, z

class LayerwiseKVVAE(nn.Module):
    def __init__(self, num_kv_heads, head_dim, per_head_latent_size, hidden_dim, logvar_min=-8.0, logvar_max=-2.0, chunk_size=1024):
        super().__init__()
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.chunk_size = chunk_size
        self.k_core = HeadwiseVAECore(head_dim, per_head_latent_size, hidden_dim, logvar_min, logvar_max)
        self.v_core = HeadwiseVAECore(head_dim, per_head_latent_size, hidden_dim, logvar_min, logvar_max)

    def _run_core(self, x_flat, core, deterministic):
        bsz, seq_len, total_dim = x_flat.shape
        x = x_flat.view(bsz, seq_len, self.num_kv_heads, self.head_dim).reshape(bsz*seq_len*self.num_kv_heads, self.head_dim)
        recons, mus, logvars = [], [], []
        for start in range(0, x.size(0), self.chunk_size):
            end = min(start + self.chunk_size, x.size(0))
            recon, mu, logvar, _ = core(x[start:end], deterministic=deterministic)
            recons.append(recon); mus.append(mu); logvars.append(logvar)
        recon = torch.cat(recons, dim=0).view(bsz, seq_len, total_dim)
        mu = torch.cat(mus, dim=0).view(bsz, seq_len, self.num_kv_heads * core.mu_proj.out_features)
        logvar = torch.cat(logvars, dim=0).view(bsz, seq_len, self.num_kv_heads * core.mu_proj.out_features)
        return recon, mu, logvar

    def forward(self, key_states_flat, value_states_flat, deterministic=False):
        k_recon, k_mu, k_logvar = self._run_core(key_states_flat, self.k_core, deterministic)
        v_recon, v_mu, v_logvar = self._run_core(value_states_flat, self.v_core, deterministic)
        mu = torch.cat([k_mu, v_mu], dim=-1)
        logvar = torch.cat([k_logvar, v_logvar], dim=-1)
        return k_recon, v_recon, mu, logvar

class SharedVAEPool(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.group_size = int(config.vae_group_size)
        self.num_layers = int(config.num_hidden_layers)
        self.num_groups = math.ceil(self.num_layers / self.group_size)
        head_dim = config.hidden_size // config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        chunk_size = config.head_chunk_size
        self.group_vaes = nn.ModuleList([
            LayerwiseKVVAE(
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                per_head_latent_size=config.per_head_latent_size,
                hidden_dim=config.vae_hidden_size,
                chunk_size=chunk_size
            )
            for _ in range(self.num_groups)
        ])

    def get_vae(self, layer_idx: int):
        return self.group_vaes[layer_idx // self.group_size]

# =========================================================
# Attention VAE and forward logic remains as in original
# (raw forward records KV, comp forward reconstructs via VAE)
# =========================================================

# =========================================================
# Main training loop, dataset, optimizer, scheduler
# =========================================================
# 保持原训练逻辑，注入 config share_vae_across_layers=True,
# 调整 loss 权重，LM 优先
# DataLoader, optimizer, scheduler 保持不变
# =========================================================

# 下面调用 main() 时使用 argparse 参数指定 share_vae_across_layers=True, ntp_weight=2.0, rec_weight=0.5
# 其它训练参数可保持原样
'''
nohup env PYTORCH_ALLOC_CONF=expandable_segments:True \
python train_mistral_kv_recon_vae_latent.py \
  --model_name_or_path mistralai/mistral-7B-instruct-v0.2 \
  --dataset_path Salesforce/wikitext \
  --dataset_config_name wikitext-103-raw-v1 \
  --dataset_split train \
  --text_column text \
  --output_dir /home/ymz/SnapKV/SnapKV/experiments/LongBench/mistral_kv_recon_latent \
  --per_head_latent_size 32 \
  --vae_hidden_size 256 \
  --split_kv True \
  --share_vae_across_layers True \
  --vae_group_size 1 \
  --kl_weight 1e-6 \
  --kl_warmup_steps 1000 \
  --free_bits 0.0 \
  --rec_weight 0.5 \
  --ntp_weight 2.0 \
  --cos_weight 0.0 \
  --rel_l2_weight 0.0 \
  --sample_during_train False \
  --deterministic_eval True \
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
> vae_train_latent.log 2>&1 &
'''