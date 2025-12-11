export https_proxy=http://127.0.0.1:7897


# block
nohup python pred_snap.py --model mistral-7B-instruct-v0.2 --compress_args_path ablation_c4096_w64_b64_k7_maxpool.json > 20251210w64b64.log 2>&1 &

nohup python pred_snap.py --model mistral-7B-instruct-v0.2 --compress_args_path ablation_c4096_w64_b256_k7_maxpool.json > 20251210w64b256.log 2>&1 &

nohup python pred_snap.py --model mistral-7B-instruct-v0.2 --compress_args_path ablation_c4096_w16_b64_k7_maxpool.json > 20251210w16b64.log 2>&1 &

nohup python pred_snap.py --model mistral-7B-instruct-v0.2 --compress_args_path ablation_c4096_w64_b1024_k7_maxpool.json > 20251210w64b1024.log 2>&1 &

nohup python pred_snap.py --model mistral-7B-instruct-v0.2 --compress_args_path ablation_c4096_w64_b4096_k7_maxpool.json > 20251210w64b4096.log 2>&1 &

nohup python pred_snap.py --model mistral-7B-instruct-v0.2 --compress_args_path ablation_c1024_w64_b64_k7_maxpool.json > 20251210c1024w64b64.log 2>&1 &