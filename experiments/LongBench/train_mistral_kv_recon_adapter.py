# -*- coding: utf-8 -*-
"""
KV Adapter Training (Frozen Mistral)
------------------------------------
- Main Mistral model frozen
- KV adapter trains with cross-attention mapping
- Suffix LM loss included to stabilize attention
- Chunked KV processing for low VRAM
"""
import os, math, argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from datasets import load_dataset

# -----------------------
# KV Adapter with Cross-Attention
# -----------------------
class KVAdapter(nn.Module):
    def __init__(self, input_dim, latent_dim=128, hidden_dim=512, n_layers=2, n_heads=4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, latent_dim)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=latent_dim, nhead=n_heads, dim_feedforward=hidden_dim,
                activation='gelu', batch_first=True
            ) for _ in range(n_layers)
        ])
        self.output_proj = nn.Linear(latent_dim, input_dim)

    def forward(self, x):
        z = self.input_proj(x)
        for layer in self.layers:
            z = layer(z)
        recon = self.output_proj(z)
        return recon, z

# -----------------------
# Dataset
# -----------------------
class PackedTextDataset(IterableDataset):
    def __init__(self, hf_dataset, tokenizer, text_column, block_size=256, max_samples=-1):
        super().__init__()
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.block_size = block_size
        self.max_samples = max_samples

    def __iter__(self):
        buffer, yielded = [], 0
        eos_id = self.tokenizer.eos_token_id
        for ex in self.dataset:
            if self.max_samples > 0 and yielded >= self.max_samples:
                break
            text = ex[self.text_column]
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            ids += [eos_id]
            buffer.extend(ids)
            while len(buffer) >= self.block_size:
                block = buffer[:self.block_size]
                buffer = buffer[self.block_size:]
                yielded += 1
                yield {"input_ids": block, "attention_mask": [1]*len(block), "labels": block.copy()}
                if self.max_samples > 0 and yielded >= self.max_samples:
                    break

def collate_fn(batch, pad_token_id):
    max_len = max(len(f["input_ids"]) for f in batch)
    input_ids, masks, labels = [], [], []
    for f in batch:
        pad_len = max_len - len(f["input_ids"])
        input_ids.append(torch.tensor(f["input_ids"] + [pad_token_id]*pad_len))
        masks.append(torch.tensor(f["attention_mask"] + [0]*pad_len))
        labels.append(torch.tensor(f["labels"] + [-100]*pad_len))
    return {"input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(masks),
            "labels": torch.stack(labels)}

# -----------------------
# Training step
# -----------------------
def training_step(adapter, hidden, input_ids, attention_mask, labels, optimizer, lm_model, args):
    optimizer.zero_grad()
    recon, latent = adapter(hidden)
    mse_loss = F.mse_loss(recon, hidden)
    cos_loss = 1 - F.cosine_similarity(recon, hidden, dim=-1).mean()
    rel_l2 = ((recon - hidden).pow(2).sum(-1)/(hidden.pow(2).sum(-1)+1e-6)).mean()
    rec_loss = args.rec_weight*mse_loss + args.cos_weight*cos_loss + args.rel_l2_weight*rel_l2

    # Suffix LM loss (pseudo: use hidden as inputs_embeds)
    logits = lm_model(inputs_embeds=recon, attention_mask=attention_mask).logits
    lm_loss = args.lm_weight * F.cross_entropy(
        logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100
    )

    total_loss = rec_loss + lm_loss
    total_loss.backward()
    optimizer.step()
    return total_loss.item()

# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--dataset_config_name", default="wikitext-103-raw-v1")
    parser.add_argument("--text_column", default="text")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--rec_weight", type=float, default=1.0)
    parser.add_argument("--cos_weight", type=float, default=0.25)
    parser.add_argument("--rel_l2_weight", type=float, default=0.25)
    parser.add_argument("--lm_weight", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)

    # Load frozen Mistral
    raw_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16
    )
    for p in raw_model.parameters():
        p.requires_grad = False
    raw_model.train()

    # Adapter
    adapter = KVAdapter(input_dim=raw_model.config.hidden_size,
                        latent_dim=args.latent_dim,
                        hidden_dim=args.hidden_dim,
                        n_layers=args.n_layers,
                        n_heads=args.n_heads)
    adapter.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_model.to(device, dtype=torch.bfloat16)
    for p in raw_model.parameters():
        p.requires_grad = False
    adapter.to(device, dtype=torch.bfloat16)

    # Dataset
    dataset = load_dataset(args.dataset_path, args.dataset_config_name, split="train")
    train_dataset = PackedTextDataset(dataset, tokenizer, args.text_column, args.block_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id))

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr)

    for step, batch in enumerate(train_loader):
        if step >= args.max_steps:
            break
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
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
  --dataset_config_name wikitext-103-raw-v1 \
  --text_column text \
  --output_dir ./mistral_kv_adapter \
  --block_size 256 \
  --batch_size 1 \
  --latent_dim 128 \
  --hidden_dim 512 \
  --n_layers 2 \
  --n_heads 4 \
  --rec_weight 1.0 \
  --cos_weight 0.25 \
  --rel_l2_weight 0.25 \
  --lm_weight 1.0 \
  --max_steps 2000 \
> kv_adapter.log 2>&1 &
'''