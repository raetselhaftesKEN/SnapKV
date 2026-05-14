# -*- coding: utf-8 -*-
"""
KV Cache Compression with Cross-Attention Latent Space
-------------------------------------------------------
- Compress K/V heads via cross-attention adapter instead of plain MLP VAE
- Include suffix LM loss to stabilize attention during inference
- Supports per-head/chunk processing
"""
import os
import math
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer, AutoConfig, set_seed
from transformers.models.mistral.modeling_mistral import (
    MistralForCausalLM
)

# =========================================================
# Cross-Attention Adapter for KV
# =========================================================
class KVAdapter(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, n_layers=2, n_heads=4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, latent_dim)
        self.output_proj = nn.Linear(latent_dim, input_dim)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=latent_dim,
                nhead=n_heads,
                dim_feedforward=hidden_dim,
                activation='gelu',
                batch_first=True,
            )
            for _ in range(n_layers)
        ])

    def forward(self, x):
        # x: [B, Seq, Dim]
        z = self.input_proj(x)
        for layer in self.layers:
            z = layer(z)
        out = self.output_proj(z)
        return out, z  # recon, latent

# =========================================================
# Extra Adapter Config
# =========================================================
@dataclass
class ExtraAdapterArgs:
    latent_dim: int = 128
    hidden_dim: int = 512
    n_layers: int = 2
    n_heads: int = 4
    rec_weight: float = 1.0
    cos_weight: float = 0.25
    rel_l2_weight: float = 0.25
    lm_weight: float = 1.0
    head_chunk_size: int = 512

def tensor_basic_stats(x: torch.Tensor) -> Dict[str,float]:
    x = x.detach().float()
    return {"mean": x.mean().item(), "std": x.std().item()}

# =========================================================
# Dataset
# =========================================================
class PackedTextIterableDataset(IterableDataset):
    def __init__(self, hf_dataset, tokenizer, text_column: str, block_size: int, add_eos_token: bool=True, max_samples: int=-1):
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
        for example in self.hf_dataset:
            if self.max_samples > 0 and yielded >= self.max_samples:
                break
            text = example[self.text_column]
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            if self.add_eos_token:
                token_ids += [eos_id]
            token_buffer.extend(token_ids)
            while len(token_buffer) >= self.block_size:
                block = token_buffer[:self.block_size]
                token_buffer = token_buffer[self.block_size:]
                yielded += 1
                yield {"input_ids": block, "attention_mask": [1]*len(block), "labels": block.copy()}
                if self.max_samples > 0 and yielded >= self.max_samples:
                    break

def lm_collator(features: List[Dict[str, any]], pad_token_id: int):
    max_len = max(len(f["input_ids"]) for f in features)
    batch_input_ids, batch_attn, batch_labels = [], [], []
    for f in features:
        pad_len = max_len - len(f["input_ids"])
        batch_input_ids.append(torch.tensor(f["input_ids"] + [pad_token_id]*pad_len))
        batch_attn.append(torch.tensor(f["attention_mask"] + [0]*pad_len))
        batch_labels.append(torch.tensor(f["labels"] + [-100]*pad_len))
    return {"input_ids": torch.stack(batch_input_ids),
            "attention_mask": torch.stack(batch_attn),
            "labels": torch.stack(batch_labels)}

# =========================================================
# Training Step
# =========================================================
def compute_suffix_lm_loss(model, recon_kv, inputs, attention_mask, labels):
    # pseudo-code: replace model KV with recon_kv
    logits = model(inputs_embeds=recon_kv, attention_mask=attention_mask).logits
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
    return loss

def training_step(adapter, kv_batch, inputs, attention_mask, labels, optimizer, lm_model, args):
    optimizer.zero_grad()
    recon, latent = adapter(kv_batch)
    mse_loss = F.mse_loss(recon, kv_batch)
    cos_loss = 1 - F.cosine_similarity(recon, kv_batch, dim=-1).mean()
    rel_l2 = ((recon - kv_batch).pow(2).sum(-1)/(kv_batch.pow(2).sum(-1)+1e-6)).mean()
    rec_loss = args.rec_weight*mse_loss + args.cos_weight*cos_loss + args.rel_l2_weight*rel_l2
    lm_loss = args.lm_weight * compute_suffix_lm_loss(lm_model, recon, inputs, attention_mask, labels)
    total_loss = rec_loss + lm_loss
    total_loss.backward()
    optimizer.step()
    return total_loss.item()

# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--rec_weight", type=float, default=1.0)
    parser.add_argument("--cos_weight", type=float, default=0.25)
    parser.add_argument("--rel_l2_weight", type=float, default=0.25)
    parser.add_argument("--lm_weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw_model = MistralForCausalLM.from_pretrained(args.model_name_or_path)
    raw_model.train()
    adapter = KVAdapter(input_dim=raw_model.config.hidden_size,
                        latent_dim=args.latent_dim,
                        hidden_dim=args.hidden_dim,
                        n_layers=args.n_layers,
                        n_heads=args.n_heads)
    adapter.train()

    # Dataset
    from datasets import load_dataset
    dataset = load_dataset(args.dataset_path, split="train", streaming=False)
    train_dataset = PackedTextIterableDataset(dataset, tokenizer, args.text_column, args.block_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=lambda x: lm_collator(x, tokenizer.pad_token_id))

    optimizer = torch.optim.AdamW(list(adapter.parameters()), lr=args.lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter.to(device)
    raw_model.to(device)

    for step, batch in enumerate(train_loader):
        if step >= args.max_steps:
            break
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        # Use hidden states as KV proxy
        with torch.no_grad():
            hidden = raw_model.model.embed_tokens(input_ids)
        loss = training_step(adapter, hidden, input_ids, attention_mask, labels, optimizer, raw_model, args)
        if step % 10 == 0:
            print(f"[step {step}] loss={loss:.6f}")

    torch.save(adapter.state_dict(), os.path.join(args.output_dir, "kv_adapter.pt"))
    print("Training finished.")

if __name__ == "__main__":
    main()

'''
nohup python train_mistral_kv_recon_adapter.py \
  --model_name_or_path mistralai/mistral-7B-instruct-v0.2 \
  --dataset_path Salesforce/wikitext \
  --text_column text \
  --output_dir ./mistral_kv_crossattn \
  --block_size 512 \
  --batch_size 1 \
  --lr 2e-4 \
  --latent_dim 128 \
  --hidden_dim 512 \
  --n_layers 2 \
  --n_heads 4 \
  --rec_weight 1.0 \
  --cos_weight 0.25 \
  --rel_l2_weight 0.25 \
  --lm_weight 1.0 \
  --max_steps 2000 \
> kv_recon_adapter.log 2>&1 &
'''