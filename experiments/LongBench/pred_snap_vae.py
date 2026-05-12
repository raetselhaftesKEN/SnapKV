# python pred_snap_vae.py --model mistral-7B-instruct-v0.2 --compress_args_path ablation_c4096_w32_k7_maxpool.json
import os
import json
import random
import torch
import numpy as np
from tqdm import tqdm
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from snapkv.monkeypatch.monkeypatch import replace_llama, replace_mistral, replace_mixtral

def seed_everything(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--compress_args_path', type=str, default=None)
    parser.add_argument('--e', action='store_true')
    parser.add_argument('--dataset', type=str, default='qasper')
    parser.add_argument('--block_size', type=int, default=64)
    return parser.parse_args(args)

def build_chat(tokenizer, prompt, model_name):
    # 保留原来的 prompt 构造逻辑
    if "mistral" in model_name or "mixtral" in model_name:
        return prompt
    return prompt  # 其它模型保持不变

def post_process(response, model_name):
    if "xgen" in model_name:
        return response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        return response.split("<eoa>")[0]
    return response

@torch.inference_mode()
def get_pred_single_gpu(data, max_length, max_gen, prompt_format, dataset, model_name, model2path, out_path, compress=False, **compress_args):
    model, tokenizer = load_model_and_tokenizer(model2path[model_name], model_name, device="cuda", compress=compress)
    device = model.device

    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = max_length // 2
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        prompt = build_chat(tokenizer, prompt, model_name)

        if "chatglm3" in model_name:
            input_data = prompt.to(device)
        else:
            input_data = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)

        context_length = input_data.input_ids.shape[-1]

        if dataset == "samsum":
            output = model.generate(
                **input_data,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length + 1,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
            )[0]
        else:
            output = model.generate(
                **input_data,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length + 1,
            )[0]

        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name)

        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({"pred": pred, "answers": json_obj.get("answers", []), "all_classes": json_obj.get("all_classes", None), "length": json_obj["length"]}, f, ensure_ascii=False)
            f.write("\n")

def load_model_and_tokenizer(path, model_name, device, compress=False):
    tokenizer = AutoTokenizer.from_pretrained(path, padding_side="right", use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        use_cache=True,
        use_flash_attention_2=True
    )

    if "mistral" in model_name or "mixtral" in model_name:
        # =========================
        # VAE 配置改成 reconstruction-first 训练结构
        # =========================
        # VAE 配置修改为训练一致
        model.config.use_kv_vae = True
        model.config.vae_ckpt_path = "/home/ymz/SnapKV/SnapKV/experiments/LongBench/mistral_kv_recon_vae/step_2000"  # 指向训练完成的 VAE checkpoint
        model.config.kv_vae_deterministic = True
        model.config.kv_vae_apply_on_decode_only = False

        model.config.share_vae_across_layers = False
        model.config.vae_group_size = 1

        model.config.per_head_latent_size = 32
        model.config.vae_hidden_size = 192  # 必须与训练一致
        model.config.logvar_min = -8.0
        model.config.logvar_max = -2.0
        model.config.head_chunk_size = 256

        # 可选混合策略
        model.config.use_kv_vae_hybrid = False
        model.config.kv_vae_keep_original_ratio = 0.5
        model.config.kv_vae_mix_seed = 42
        model.config.kv_vae_hybrid_prefill_only = False
        model.config.kv_vae_hybrid_decode_only = False
        model.config.kv_vae_debug = False

    model = model.eval()
    model.to(device)
    return model, tokenizer

if __name__ == "__main__":
    seed_everything(42)
    args = parse_args()
    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))

    model_name = args.model
    max_length = min(model2maxlen[model_name], 16384)
    block_size = args.block_size
    dataset = args.dataset
    out_dir = "pred_e" if args.e else "pred"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/{model_name}/{dataset}.jsonl"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    data = load_dataset('THUDM/LongBench', dataset+"_e" if args.e else dataset, split='test', trust_remote_code=True)
    data_all = [x for x in data]
    prompt_format = dataset2prompt[dataset]

    compress_args = None
    if args.compress_args_path:
        compress_args = json.load(open(os.path.join('config', args.compress_args_path), "r"))
        replace_llama()
        replace_mistral()
        replace_mixtral()

    get_pred_single_gpu(
        data_all,
        max_length,
        dataset2maxlen[dataset],
        prompt_format,
        dataset,
        model_name,
        model2path,
        out_path,
        compress=(compress_args is not None),
        **(compress_args or {})
    )