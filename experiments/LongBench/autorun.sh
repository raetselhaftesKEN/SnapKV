#!/usr/bin/env bash
set -euo pipefail


CMD="python pred_snap.py --model mistral-7B-instruct-v0.2 --compress_args_path"
DATE_TAG="20251221"

run_one () {
  local json="$1"
  local log="$2"
  echo "[$(date '+%F %T')] START: $json" | tee -a "$log"
  $CMD "$json" >> "$log" 2>&1
  echo "[$(date '+%F %T')] DONE : $json" | tee -a "$log"
}

run_one "ablation_c1024_w32_b16_k7_maxpool.json"   "${DATE_TAG}c1024w32b16.log"
run_one "ablation_c1024_w32_b32_k7_maxpool.json"   "${DATE_TAG}c1024w32b32.log"
run_one "ablation_c1024_w32_b64_k7_maxpool.json"   "${DATE_TAG}c1024w32b64.log"
run_one "ablation_c1024_w32_b128_k7_maxpool.json"  "${DATE_TAG}c1024w32b128.log"
run_one "ablation_c1024_w32_b256_k7_maxpool.json"  "${DATE_TAG}c1024w32b256.log"
run_one "ablation_c1024_w32_b512_k7_maxpool.json"  "${DATE_TAG}c1024w32b512.log"
run_one "ablation_c1024_w32_b1024_k7_maxpool.json" "${DATE_TAG}c1024w32b1024.log"

echo "[$(date '+%F %T')] ALL DONE"
