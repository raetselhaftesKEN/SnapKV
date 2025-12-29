#!/usr/bin/env bash
set -u

CMD="python pred_snap.py --model mistral-7B-instruct-v0.2 --compress_args_path"
DATE_TAG="20251224"

run_one () {
  local json="$1"
  local log="$2"
  echo "[$(date '+%F %T')] START: $json" | tee -a "$log"
  if $CMD "$json" >> "$log" 2>&1; then
    echo "[$(date '+%F %T')] DONE : $json" | tee -a "$log"
    return 0
  else
    echo "[$(date '+%F %T')] FAIL : $json (exit=$?)" | tee -a "$log"
    return 1
  fi
}


run_one "ablation_c256_w32_k7_maxpool.json"   "${DATE_TAG}c256w32.log"   || true
run_one "ablation_c128_w32_k7_maxpool.json"   "${DATE_TAG}c128w32.log"   || true


echo "[$(date '+%F %T')] ALL DONE (continue-on-fail mode)"
