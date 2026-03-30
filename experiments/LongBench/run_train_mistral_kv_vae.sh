#!/bin/bash

set -euo pipefail

# =========================
# Basic paths
# =========================
LOG_DIR="/home/ymz/SnapKV/SnapKV/experiments/LongBench/log"

# =========================
# Environment
# =========================
export PYTORCH_ALLOC_CONF="expandable_segments:True"

# 如果你要固定GPU，可以取消下面注释并修改编号
# export CUDA_VISIBLE_DEVICES=0

# =========================
# Prepare log dir
# =========================
mkdir -p "${LOG_DIR}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/train_mistral_kv_vae_e2e_${TIMESTAMP}.log"
PID_FILE="${LOG_DIR}/train_mistral_kv_vae_e2e_${TIMESTAMP}.pid"

echo "Starting training..."
echo "Log file: ${LOG_FILE}"

nohup python train_mistral_kv_vae_e2e_text_oomfix_mu_logvar.py \
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
  > "${LOG_FILE}" 2>&1 &

echo $! > "${PID_FILE}"

echo "Training started in background."
echo "PID: $(cat "${PID_FILE}")"
echo "PID file: ${PID_FILE}"
echo "Log file: ${LOG_FILE}"