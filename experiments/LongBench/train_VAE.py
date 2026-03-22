#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train a conditional VAE on attention rows extracted from a Hugging Face causal LM,
e.g. mistralai/Mistral-7B-Instruct-v0.2.

Core idea:
- Freeze the LLM
- Run forward with output_attentions=True
- Convert each attention row (query -> past keys) into a fixed-length vector
- Train a conditional VAE to reconstruct that distribution

Author: ChatGPT
"""

import os
import math
import json
import random
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


# =========================
# Utils
# =========================

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def masked_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    s = x.sum(dim=-1, keepdim=True)
    return x / (s + eps)


def kl_anneal(step: int, warmup_steps: int, max_beta: float = 1.0) -> float:
    if warmup_steps <= 0:
        return max_beta
    return min(max_beta, max_beta * step / warmup_steps)


def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    p, q: (..., K), both assumed normalized
    returns mean JS over batch
    """
    p = torch.clamp(p, min=eps)
    q = torch.clamp(q, min=eps)
    m = 0.5 * (p + q)
    js = 0.5 * (p * (p.log() - m.log())).sum(dim=-1) + 0.5 * (q * (q.log() - m.log())).sum(dim=-1)
    return js.mean()


def topk_mass_loss(pred: torch.Tensor, target: torch.Tensor, topk: int = 16) -> torch.Tensor:
    """
    Encourage predicted mass on target top-k positions.
    pred, target: [B, K], normalized
    """
    k = min(topk, target.size(-1))
    idx = torch.topk(target, k=k, dim=-1).indices
    gathered_pred = torch.gather(pred, -1, idx).sum(dim=-1)
    gathered_tgt = torch.gather(target, -1, idx).sum(dim=-1)
    return F.mse_loss(gathered_pred, gathered_tgt)


# =========================
# Conditional VAE
# =========================

class CondAttentionVAE(nn.Module):
    def __init__(
        self,
        max_kv_len: int,
        num_layers: int,
        num_heads: int,
        latent_dim: int = 64,
        hidden_dim: int = 512,
        cond_dim: int = 64,
        query_pos_bins: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_kv_len = max_kv_len
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.latent_dim = latent_dim
        self.query_pos_bins = query_pos_bins

        self.layer_emb = nn.Embedding(num_layers, cond_dim)
        self.head_emb = nn.Embedding(num_heads, cond_dim)
        self.qpos_emb = nn.Embedding(query_pos_bins, cond_dim)

        cond_total_dim = cond_dim * 3

        # Input is transformed attention vector + condition
        self.encoder = nn.Sequential(
            nn.Linear(max_kv_len + cond_total_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.mu_proj = nn.Linear(hidden_dim, latent_dim)
        self.logvar_proj = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + cond_total_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, max_kv_len),
        )

    def make_cond(self, layer_ids, head_ids, qpos_bins):
        c = torch.cat([
            self.layer_emb(layer_ids),
            self.head_emb(head_ids),
            self.qpos_emb(qpos_bins),
        ], dim=-1)
        return c

    def encode(self, x, layer_ids, head_ids, qpos_bins):
        c = self.make_cond(layer_ids, head_ids, qpos_bins)
        h = self.encoder(torch.cat([x, c], dim=-1))
        mu = self.mu_proj(h)
        logvar = self.logvar_proj(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_logits(self, z, layer_ids, head_ids, qpos_bins):
        c = self.make_cond(layer_ids, head_ids, qpos_bins)
        logits = self.decoder(torch.cat([z, c], dim=-1))
        return logits

    def forward(self, x, layer_ids, head_ids, qpos_bins):
        mu, logvar = self.encode(x, layer_ids, head_ids, qpos_bins)
        z = self.reparameterize(mu, logvar)
        logits = self.decode_logits(z, layer_ids, head_ids, qpos_bins)
        recon = F.softmax(logits, dim=-1)
        return recon, logits, mu, logvar


# =========================
# Attention-row extraction
# =========================

@dataclass
class SampleConfig:
    max_seq_len: int = 1024
    max_kv_len: int = 512
    queries_per_head: int = 8
    layers_to_use: Optional[List[int]] = None
    heads_to_use: Optional[List[int]] = None
    min_valid_query_index: int = 32
    use_sqrt_transform: bool = True
    max_samples_per_sequence: int = 2048


def build_fixed_attention_row(
    full_row: torch.Tensor,
    q_index: int,
    max_kv_len: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    full_row: [seq_len] attention row over all positions (already causal-masked by model)
    q_index: current query position
    Return: fixed-length [max_kv_len] vector over recent history only, renormalized
    """
    valid = full_row[: q_index + 1]  # causal prefix
    if valid.numel() >= max_kv_len:
        vec = valid[-max_kv_len:]
    else:
        pad_len = max_kv_len - valid.numel()
        vec = F.pad(valid, (pad_len, 0), value=0.0)

    vec = vec / (vec.sum() + eps)
    return vec


def bucketize_query_pos(q_index: int, seq_len: int, num_bins: int) -> int:
    """
    Relative query position bucket, from older query to newer query.
    """
    if seq_len <= 1:
        return 0
    rel = q_index / (seq_len - 1)
    b = min(num_bins - 1, int(rel * num_bins))
    return b


class AttentionRowIterableDataset(IterableDataset):
    """
    Streams attention rows extracted online from a frozen LLM.
    """

    def __init__(
        self,
        hf_dataset,
        tokenizer,
        llm,
        device,
        text_field: str,
        sample_cfg: SampleConfig,
        batch_size_text: int = 1,
    ):
        super().__init__()
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.llm = llm
        self.device = device
        self.text_field = text_field
        self.cfg = sample_cfg
        self.batch_size_text = batch_size_text

        if self.cfg.layers_to_use is None:
            self.cfg.layers_to_use = list(range(llm.config.num_hidden_layers))

        # output attentions are generally on attention heads, not kv heads
        if self.cfg.heads_to_use is None:
            self.cfg.heads_to_use = list(range(llm.config.num_attention_heads))

    def _iter_texts(self):
        for ex in self.dataset:
            text = ex[self.text_field]
            if isinstance(text, list):
                text = " ".join([str(t) for t in text])
            elif not isinstance(text, str):
                text = str(text)
            if len(text.strip()) == 0:
                continue
            yield text

    @torch.no_grad()
    def _extract_from_text(self, text: str):
        tok = self.tokenizer(
            text,
            truncation=True,
            max_length=self.cfg.max_seq_len,
            return_tensors="pt",
        )
        input_ids = tok["input_ids"].to(self.device)
        attention_mask = tok["attention_mask"].to(self.device)

        outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            use_cache=False,
        )

        # List[num_layers], each: [B, num_heads, T, T]
        attentions = outputs.attentions
        if attentions is None:
            return

        seq_len = input_ids.shape[1]
        if seq_len <= self.cfg.min_valid_query_index + 1:
            return

        produced = 0
        for layer_id in self.cfg.layers_to_use:
            attn_l = attentions[layer_id][0]  # [H, T, T]
            H, T, _ = attn_l.shape

            head_ids = [h for h in self.cfg.heads_to_use if h < H]
            if len(head_ids) == 0:
                continue

            # Sample query positions with slight bias toward recent tokens
            possible_q = list(range(self.cfg.min_valid_query_index, T))
            recent_zone = possible_q[max(0, len(possible_q) // 2):]
            q_candidates = recent_zone if len(recent_zone) >= self.cfg.queries_per_head else possible_q

            for head_id in head_ids:
                chosen_q = random.sample(
                    q_candidates,
                    k=min(self.cfg.queries_per_head, len(q_candidates))
                )

                for q_idx in chosen_q:
                    row = attn_l[head_id, q_idx, :]  # [T]
                    vec = build_fixed_attention_row(row, q_idx, self.cfg.max_kv_len)

                    if self.cfg.use_sqrt_transform:
                        # softer geometry for simplex distributions
                        feat = torch.sqrt(torch.clamp(vec, min=1e-8))
                    else:
                        feat = vec

                    qbin = bucketize_query_pos(q_idx, T, num_bins=64)

                    yield {
                        "x": feat.cpu(),
                        "target": vec.cpu(),
                        "layer_id": torch.tensor(layer_id, dtype=torch.long),
                        "head_id": torch.tensor(head_id, dtype=torch.long),
                        "qpos_bin": torch.tensor(qbin, dtype=torch.long),
                    }

                    produced += 1
                    if produced >= self.cfg.max_samples_per_sequence:
                        return

    def __iter__(self):
        for text in self._iter_texts():
            for item in self._extract_from_text(text):
                yield item


def collate_rows(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    x = torch.stack([b["x"] for b in batch], dim=0)
    target = torch.stack([b["target"] for b in batch], dim=0)
    layer_id = torch.stack([b["layer_id"] for b in batch], dim=0)
    head_id = torch.stack([b["head_id"] for b in batch], dim=0)
    qpos_bin = torch.stack([b["qpos_bin"] for b in batch], dim=0)
    return {
        "x": x,
        "target": target,
        "layer_id": layer_id,
        "head_id": head_id,
        "qpos_bin": qpos_bin,
    }


# =========================
# Main training
# =========================

def parse_args():
    parser = argparse.ArgumentParser()

    # Model / dataset
    parser.add_argument("--model_name", type=str, required=True,
                        help="HF model name, e.g. mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--dataset_name", type=str, default=None,
                        help="HF dataset name, e.g. wikitext")
    parser.add_argument("--dataset_config", type=str, default=None,
                        help="HF dataset config, e.g. wikitext-103-raw-v1")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--text_field", type=str, default="text")

    # Optional local jsonl
    parser.add_argument("--local_jsonl", type=str, default=None,
                        help="Path to local JSONL with a text field")

    # Tokenization / extraction
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--max_kv_len", type=int, default=512)
    parser.add_argument("--queries_per_head", type=int, default=8)
    parser.add_argument("--min_valid_query_index", type=int, default=32)

    # Training
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--cond_dim", type=int, default=64)
    parser.add_argument("--batch_size_rows", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_steps", type=int, default=10000)
    parser.add_argument("--warmup_kl_steps", type=int, default=2000)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--recon_loss", type=str, default="js",
                        choices=["mse", "kl", "js"])
    parser.add_argument("--topk_aux", type=int, default=16)
    parser.add_argument("--topk_aux_weight", type=float, default=0.2)

    # System
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--attn_implementation", type=str, default="eager",
                        choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="./attn_vae_ckpt")
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=0)

    return parser.parse_args()


def get_dtype(name: str):
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def load_local_jsonl_dataset(path: str):
    return load_dataset("json", data_files=path, split="train")


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = get_dtype(args.dtype)

    print(f"[Info] Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[Info] Loading frozen LLM: {args.model_name}")
    llm = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
        device_map=None,
    )
    llm.to(device)
    llm.eval()
    for p in llm.parameters():
        p.requires_grad = False

    if args.local_jsonl is not None:
        dataset = load_local_jsonl_dataset(args.local_jsonl)
    else:
        if args.dataset_name is None:
            raise ValueError("Please specify either --dataset_name or --local_jsonl")
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config,
            split=args.dataset_split,
        )

    num_layers = llm.config.num_hidden_layers
    num_heads = llm.config.num_attention_heads

    print(f"[Info] Model layers={num_layers}, heads={num_heads}")

    sample_cfg = SampleConfig(
        max_seq_len=args.max_seq_len,
        max_kv_len=args.max_kv_len,
        queries_per_head=args.queries_per_head,
        min_valid_query_index=args.min_valid_query_index,
        use_sqrt_transform=True,
    )

    row_dataset = AttentionRowIterableDataset(
        hf_dataset=dataset,
        tokenizer=tokenizer,
        llm=llm,
        device=device,
        text_field=args.text_field,
        sample_cfg=sample_cfg,
    )

    loader = DataLoader(
        row_dataset,
        batch_size=args.batch_size_rows,
        collate_fn=collate_rows,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    vae = CondAttentionVAE(
        max_kv_len=args.max_kv_len,
        num_layers=num_layers,
        num_heads=num_heads,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        cond_dim=args.cond_dim,
        query_pos_bins=64,
    ).to(device)

    optimizer = torch.optim.AdamW(
        vae.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    global_step = 0
    loader_iter = iter(loader)

    while global_step < args.num_steps:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        x = batch["x"].to(device, non_blocking=True).float()
        target = batch["target"].to(device, non_blocking=True).float()
        layer_id = batch["layer_id"].to(device, non_blocking=True)
        head_id = batch["head_id"].to(device, non_blocking=True)
        qpos_bin = batch["qpos_bin"].to(device, non_blocking=True)

        recon, logits, mu, logvar = vae(x, layer_id, head_id, qpos_bin)

        # reconstruction loss
        if args.recon_loss == "mse":
            recon_loss = F.mse_loss(recon, target)
        elif args.recon_loss == "kl":
            # KL(target || recon)
            recon_loss = F.kl_div(
                torch.log(torch.clamp(recon, min=1e-8)),
                target,
                reduction="batchmean",
            )
        else:
            recon_loss = js_divergence(target, recon)

        aux_loss = topk_mass_loss(recon, target, topk=args.topk_aux)

        kl_loss = -0.5 * torch.mean(
            torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        )

        beta = kl_anneal(global_step, args.warmup_kl_steps, max_beta=1.0)
        loss = recon_loss + beta * kl_loss + args.topk_aux_weight * aux_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(vae.parameters(), args.grad_clip)
        optimizer.step()

        if global_step % args.log_every == 0:
            print(
                f"[Step {global_step:06d}] "
                f"loss={loss.item():.6f} "
                f"recon={recon_loss.item():.6f} "
                f"kl={kl_loss.item():.6f} "
                f"aux={aux_loss.item():.6f} "
                f"beta={beta:.4f}"
            )

        if global_step > 0 and global_step % args.save_every == 0:
            ckpt_path = os.path.join(args.save_dir, f"step_{global_step}.pt")
            torch.save({
                "step": global_step,
                "vae_state_dict": vae.state_dict(),
                "args": vars(args),
                "model_name": args.model_name,
                "num_layers": num_layers,
                "num_heads": num_heads,
                "max_kv_len": args.max_kv_len,
            }, ckpt_path)
            print(f"[Info] Saved checkpoint to: {ckpt_path}")

        global_step += 1

    final_path = os.path.join(args.save_dir, "final.pt")
    torch.save({
        "step": global_step,
        "vae_state_dict": vae.state_dict(),
        "args": vars(args),
        "model_name": args.model_name,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "max_kv_len": args.max_kv_len,
    }, final_path)
    print(f"[Info] Training complete. Final checkpoint: {final_path}")


if __name__ == "__main__":
    main()

'''
python train_VAE.py \
  --model_name mistralai/Mistral-7B-Instruct-v0.2 \
  --dataset_name wikitext \
  --dataset_config wikitext-103-raw-v1 \
  --dataset_split train \
  --text_field text \
  --max_seq_len 1024 \
  --max_kv_len 512 \
  --queries_per_head 8 \
  --latent_dim 64 \
  --hidden_dim 512 \
  --batch_size_rows 128 \
  --num_steps 10000 \
  --device cuda \
  --save_dir ./mistral_attn_vae
'''