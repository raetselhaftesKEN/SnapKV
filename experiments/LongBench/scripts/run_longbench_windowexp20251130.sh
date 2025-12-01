export https_proxy=http://127.0.0.1:7897


nohup python pred_snap.py --model mistral-7B-instruct-v0.2 --compress_args_path ablation_c4096_w16_k7_maxpool.json > 20251130w16.log 2>&1 &

nohup python pred_snap.py --model mistral-7B-instruct-v0.2 --compress_args_path ablation_c4096_w32_k7_maxpool.json > 20251130w32.log 2>&1 &

nohup python pred_snap.py --model mistral-7B-instruct-v0.2 --compress_args_path ablation_c4096_w64_k7_maxpool.json > 20251130w64.log 2>&1 &

nohup python pred_snap.py --model mistral-7B-instruct-v0.2 --compress_args_path ablation_c1024_w32_k7_maxpool.json > 20251201c1024.log 2>&1 &

nohup python pred_snap.py --model mistral-7B-instruct-v0.2 --compress_args_path ablation_c2048_w32_k7_maxpool.json > 20251201c2048.log 2>&1 &

nohup python pred_snap.py --model mistral-7B-instruct-v0.2 --compress_args_path ablation_c512_w32_k7_maxpool.json > 20251201c512.log 2>&1 &

nohup python pred_snap.py --model mistral-7B-instruct-v0.2 > 20251201nocompress.log 2>&1 &

nohup python pred_snap_nokv.py --model mistral-7B-instruct-v0.2 > 20251201nokv.log 2>&1 &
