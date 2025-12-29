#!/usr/bin/env bash
set -u

CMD="python pred_snap.py --model mistral-7B-instruct-v0.2 --compress_args_path"
DATE_TAG="20251229"

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


run_one "ablation_c512_w32_k7_maxpool_sw.json"   "${DATE_TAG}c512w32sw.log"   || true
run_one "ablation_c1024_w32_k7_maxpool_sw.json"   "${DATE_TAG}c1024w32sw.log"   || true
run_one "ablation_c2048_w32_k7_maxpool_sw.json"   "${DATE_TAG}c2048w32sw.log"   || true
run_one "ablation_c4096_w32_k7_maxpool_sw.json"   "${DATE_TAG}c4096w32sw.log"   || true


echo "[$(date '+%F %T')] ALL DONE (continue-on-fail mode)"
