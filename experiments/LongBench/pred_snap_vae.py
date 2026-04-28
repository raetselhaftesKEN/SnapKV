# python pred_snap_vae.py --model mistral-7B-instruct-v0.2 --compress_args_path ablation_c4096_w32_k7_maxpool.json --use_kv_vae_cache --vae_ckpt_path /path/to/step_2000
import os
import json
import random
import argparse

import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from snapkv.monkeypatch.monkeypatch import replace_llama, replace_mistral, replace_mixtral


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, choices=[
        "llama2-7b-chat-4k", "longchat-v1.5-7b-32k", "xgen-7b-8k",
        "internlm-7b-8k", "chatglm2-6b", "chatglm2-6b-32k", "chatglm3-6b-32k", "vicuna-v1.5-7b-16k",
        "mistral-7B-instruct-v0.2", "mistral-7B-instruct-v0.1", "llama-2-7B-32k-instruct",
        "mixtral-8x7B-instruct-v0.1", "lwm-text-chat-1m", "lwm-text-1m"
    ])
    parser.add_argument('--compress_args_path', type=str, default=None, help="Path to the compress args json under config/")
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument('--dataset', type=str, default='qasper', help="Dataset to evaluate on")
    parser.add_argument('--block_size', type=int, default=64, help="Comp. block size")

    # =========================
    # VAE dropped-KV cache args
    # =========================
    parser.add_argument('--use_kv_vae_cache', action='store_true',
                        help='Enable hybrid SnapKV + VAE latent cache for dropped KV tokens')
    parser.add_argument('--vae_ckpt_path', type=str, default=None,
                        help='Path to VAE checkpoint directory or trainable_vae_only.bin')
    parser.add_argument('--kv_vae_deterministic', action='store_true',
                        help='Use deterministic latent (z=mu) when restoring dropped KV')
    parser.add_argument('--vae_group_size', type=int, default=4)
    parser.add_argument('--kv_latent_size', type=int, default=32)
    parser.add_argument('--vae_hidden_size', type=int, default=256)
    parser.add_argument('--logvar_min', type=float, default=-4.0)
    parser.add_argument('--logvar_max', type=float, default=1.0)

    # optional debug
    parser.add_argument('--kv_vae_debug', action='store_true',
                        help='Enable VAE debug stats inside hijack module')

    return parser.parse_args(args)


# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    if "chatglm3" in model_name:
        print('chatglm3')
        prompt = tokenizer.build_chat_input(prompt)
    elif "chatglm" in model_name:
        print('chatglm')
        prompt = tokenizer.build_prompt(prompt)
    elif "longchat" in model_name or "vicuna" in model_name:
        print('longchat')
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "llama2" in model_name or "llama-2" in model_name or "lwm" in model_name:
        print('llama2', model_name)
        prompt = f"[INST]{prompt}[/INST]"
    elif "xgen" in model_name:
        print('xgen')
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        print('internlm')
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    elif "mistral" in model_name or "mixtral" in model_name:
        print('mistral')
        prompt = prompt
    return prompt


def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response


@torch.inference_mode()
def get_pred_single_gpu(
    data,
    max_length,
    max_gen,
    prompt_format,
    dataset,
    model_name,
    model2path,
    out_path,
    args,
    compress=False,
    window_sizes=None,
    max_capacity_prompts=None,
    kernel_sizes=None,
    pooling=None,
    block_size=64,
):
    model, tokenizer = load_model_and_tokenizer(
        model2path[model_name],
        model_name,
        device="cuda",
        compress=compress,
        args=args,
    )
    device = model.device
    printed = False

    for json_obj in tqdm(data):
        ############################################################################################################
        # load compress args
        if compress:
            layers = len(model.model.layers)

            if not isinstance(window_sizes, list):
                window_sizes = [window_sizes] * layers
            if not isinstance(max_capacity_prompts, list):
                max_capacity_prompts = [max_capacity_prompts] * layers
            if not isinstance(kernel_sizes, list):
                kernel_sizes = [kernel_sizes] * layers

            for i in range(layers):
                attn_cfg = model.model.layers[i].self_attn.config
                attn_cfg.window_size = window_sizes[i]
                attn_cfg.max_capacity_prompt = max_capacity_prompts[i]
                attn_cfg.kernel_size = kernel_sizes[i]
                attn_cfg.pooling = pooling
                attn_cfg.block_size = block_size

                # pass VAE cache configs to every layer attention config
                attn_cfg.use_kv_vae_cache = bool(args.use_kv_vae_cache)
                attn_cfg.vae_ckpt_path = args.vae_ckpt_path
                attn_cfg.kv_vae_deterministic = bool(args.kv_vae_deterministic)
                attn_cfg.share_vae_across_layers = True
                attn_cfg.vae_group_size = int(args.vae_group_size)
                attn_cfg.kv_latent_size = int(args.kv_latent_size)
                attn_cfg.vae_hidden_size = int(args.vae_hidden_size)
                attn_cfg.logvar_min = float(args.logvar_min)
                attn_cfg.logvar_max = float(args.logvar_max)
                attn_cfg.kv_vae_debug = bool(args.kv_vae_debug)
        ############################################################################################################

        prompt = prompt_format.format(**json_obj)

        # truncate to fit max_length
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if "chatglm3" in model_name:
            tokenized_prompt = tokenizer(
                prompt, truncation=False, return_tensors="pt", add_special_tokens=False
            ).input_ids[0]

        if len(tokenized_prompt) > max_length:
            half = int(max_length / 2)
            prompt = (
                tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)
                + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
            )

        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
            prompt = build_chat(tokenizer, prompt, model_name)

        if "chatglm3" in model_name:
            input_data = prompt.to(device)
        else:
            input_data = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)

        context_length = input_data.input_ids.shape[-1]

        if not printed:
            print(prompt)
            printed = True

        if dataset == "samsum":
            output = model.generate(
                **input_data,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length + 1,
                eos_token_id=[
                    tokenizer.eos_token_id,
                    tokenizer.encode("\n", add_special_tokens=False)[-1]
                ],
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

        torch.cuda.synchronize()
        print(
            f"[MEM] allocated={torch.cuda.memory_allocated() / 1024 ** 2:.1f}MB | "
            f"reserved={torch.cuda.memory_reserved() / 1024 ** 2:.1f}MB | "
            f"max_alloc={torch.cuda.max_memory_allocated() / 1024 ** 2:.1f}MB"
        )

        # optional VAE debug stats
        if "mistral" in model_name and args.kv_vae_debug:
            try:
                dbg = getattr(model.model.layers[0].self_attn, "_kv_vae_last_stats", None)
                if dbg is not None:
                    print("[KV-VAE DEBUG]", dbg)
            except Exception:
                pass

        with open(out_path, "a", encoding="utf-8") as f:
            json.dump(
                {
                    "pred": pred,
                    "answers": json_obj["answers"],
                    "all_classes": json_obj["all_classes"],
                    "length": json_obj["length"],
                },
                f,
                ensure_ascii=False,
            )
            f.write('\n')


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(path, model_name, device, compress=False, args=None):
    if "chatglm" in model_name or "internlm" in model_name or "xgen" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            path, trust_remote_code=True, torch_dtype=torch.bfloat16
        ).to(device)

    elif "llama2" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to(device)

    elif "longchat" in model_name or "vicuna" in model_name:
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
            use_cache=True,
            use_flash_attention_2=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)

    elif "llama-2" in model_name or "lwm" in model_name:
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
            use_cache=True,
            use_flash_attention_2=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)

    elif "mistral" in model_name:
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
            use_cache=True,
            use_flash_attention_2=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            path,
            padding_side="right",
            use_fast=False,
        )

        # =========================
        # Hybrid SnapKV + VAE latent-cache config
        # =========================
        if args is not None:
            model.config.use_kv_vae_cache = bool(args.use_kv_vae_cache)
            model.config.vae_ckpt_path = args.vae_ckpt_path
            model.config.kv_vae_deterministic = bool(args.kv_vae_deterministic)

            # keep consistent with training config
            model.config.share_vae_across_layers = True
            model.config.vae_group_size = int(args.vae_group_size)
            model.config.kv_latent_size = int(args.kv_latent_size)
            model.config.vae_hidden_size = int(args.vae_hidden_size)
            model.config.logvar_min = float(args.logvar_min)
            model.config.logvar_max = float(args.logvar_max)
            model.config.kv_vae_debug = bool(args.kv_vae_debug)

    elif "mixtral" in model_name:
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
            use_cache=True,
            use_flash_attention_2=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(path)

    else:
        raise ValueError(f"Model {model_name} not supported!")

    model = model.eval()
    return model, tokenizer


if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()

    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))

    model_name = args.model
    max_length = model2maxlen[model_name]
    block_size = args.block_size

    if args.e:
        datasets = [
            "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news",
            "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"
        ]
    else:
        datasets = [
            "narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique",
            "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum",
            "passage_count", "passage_retrieval_en", "lcc", "repobench-p"
        ]

    if args.dataset not in datasets:
        raise ValueError(f"Dataset {args.dataset} not found in datasets")

    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))

    max_length = min(max_length, 16384)

    if not os.path.exists("pred"):
        os.makedirs("pred")
    if not os.path.exists("pred_e"):
        os.makedirs("pred_e")

    dataset = args.dataset

    if args.compress_args_path:
        compress_args = json.load(open(os.path.join('config', args.compress_args_path), "r"))
        compress = True
        write_model_name = model_name + args.compress_args_path.split(".")[0]

        # IMPORTANT:
        # replace_mistral() must point to your modified mistral_hijack_4_37_vae.py
        replace_llama()
        replace_mistral()
        replace_mixtral()
    else:
        compress = False
        compress_args = None
        write_model_name = model_name

    if args.use_kv_vae_cache:
        write_model_name += "_vae_cache"

    if args.e:
        data = load_dataset('THUDM/LongBench', f"{dataset}_e", split='test', trust_remote_code=True)
        if not os.path.exists(f"pred_e/{write_model_name}"):
            os.makedirs(f"pred_e/{write_model_name}")
        out_path = f"pred_e/{write_model_name}/{dataset}.jsonl"
    else:
        data = load_dataset('THUDM/LongBench', dataset, split='test', trust_remote_code=True)
        if not os.path.exists(f"pred_e/{write_model_name}"):
            os.makedirs(f"pred_e/{write_model_name}")
        out_path = f"pred_e/{write_model_name}/{dataset}.jsonl"

    prompt_format = dataset2prompt[dataset]
    max_gen = dataset2maxlen[dataset]
    data_all = [data_sample for data_sample in data]

    if compress_args is not None:
        get_pred_single_gpu(
            data_all,
            max_length,
            max_gen,
            prompt_format,
            dataset,
            model_name,
            model2path,
            out_path,
            args=args,
            compress=compress,
            **compress_args,
        )
    else:
        get_pred_single_gpu(
            data_all,
            max_length,
            max_gen,
            prompt_format,
            dataset,
            model_name,
            model2path,
            out_path,
            args=args,
            compress=compress,
        )

'''
python pred_snap_vae.py \
  --model mistral-7B-instruct-v0.2 \
  --dataset qasper \
  --compress_args_path ablation_c4096_w32_k7_maxpool.json \
  --use_kv_vae_cache \
  --vae_ckpt_path /home/ymz/SnapKV/SnapKV/experiments/LongBench/mistral_kv_predictor_friendly/step_2000 \
  --kv_vae_deterministic
'''