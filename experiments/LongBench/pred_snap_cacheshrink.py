# python pred_snap_cacheshrink.py \
#   --model mistral-7B-instruct-v0.2 \
#   --compress_args_path ablation_c4096_w32_k7_maxpool.json \
#   --use_cacheshrink \
#   --cacheshrink_ratio 2.0 \
#   --cacheshrink_method auto \
#   --dataset qasper

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
        "internlm-7b-8k", "chatglm2-6b", "chatglm2-6b-32k", "chatglm3-6b-32k",
        "vicuna-v1.5-7b-16k", "mistral-7B-instruct-v0.2", "mistral-7B-instruct-v0.1",
        "llama-2-7B-32k-instruct", "mixtral-8x7B-instruct-v0.1", "lwm-text-chat-1m",
        "lwm-text-1m"
    ])
    parser.add_argument('--compress_args_path', type=str, default=None, help="Path to the SnapKV compress args")
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument('--dataset', type=str, default='qasper', help="Dataset to evaluate on")
    parser.add_argument('--block_size', type=int, default=64, help="SnapKV comp. block size")

    # CacheShrink / MLA options. These are intentionally thin wrappers over convert_to_mla(),
    # so this script stays close to the pip-install quick-start usage.
    parser.add_argument('--use_cacheshrink', action='store_true', help='Enable CacheShrink MLA compression')
    parser.add_argument('--cacheshrink_ratio', type=float, default=2.0, help='Target KV compression ratio for CacheShrink')
    parser.add_argument('--cacheshrink_method', type=str, default='auto',
                        choices=['auto', 'separate', 'xkv', 'joint', 'decoupled_rope'],
                        help='CacheShrink compression method')
    parser.add_argument('--cacheshrink_group_size', type=int, default=4,
                        help='xKV cross-layer group size; only used by xKV/auto on GQA models')
    parser.add_argument('--cacheshrink_skip_early_layers', type=int, default=0,
                        help='xKV: number of early layers to leave out of xKV groups')
    parser.add_argument('--cacheshrink_keep_early_original', action='store_true',
                        help='xKV: keep skipped early layers as original attention')
    parser.add_argument('--cacheshrink_use_calibration', action='store_true',
                        help='Use CacheShrink calibration/SVD init. Slower but usually better.')
    parser.add_argument('--cacheshrink_calib_samples', type=int, default=128,
                        help='Number of calibration samples for CacheShrink')
    parser.add_argument('--cacheshrink_calib_len', type=int, default=512,
                        help='Max calibration length for CacheShrink')
    parser.add_argument('--cacheshrink_dtype', type=str, default='bfloat16', choices=['bfloat16', 'float16', 'float32'],
                        help='dtype passed to CacheShrink convert_to_mla. README recommends bfloat16.')
    parser.add_argument('--cacheshrink_verbose', action='store_true', help='Print CacheShrink conversion details')
    return parser.parse_args(args)


def str_to_torch_dtype(dtype_name: str):
    if dtype_name == 'bfloat16':
        return torch.bfloat16
    if dtype_name == 'float16':
        return torch.float16
    if dtype_name == 'float32':
        return torch.float32
    raise ValueError(f'Unsupported dtype: {dtype_name}')


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


def apply_snapkv_config(model, window_sizes=None, max_capacity_prompts=None,
                        kernel_sizes=None, pooling=None, block_size=64):
    """Write SnapKV runtime config to every attention layer.

    CacheShrink may replace the attention module class. Therefore this function is defensive:
    if a converted layer no longer exposes the same config path, we skip it instead of crashing.
    """
    if not hasattr(model, 'model') or not hasattr(model.model, 'layers'):
        print('[WARN] Cannot find model.model.layers; skip SnapKV runtime config.')
        return

    layers = len(model.model.layers)
    if not isinstance(window_sizes, list):
        window_sizes = [window_sizes] * layers
    if not isinstance(max_capacity_prompts, list):
        max_capacity_prompts = [max_capacity_prompts] * layers
    if not isinstance(kernel_sizes, list):
        kernel_sizes = [kernel_sizes] * layers

    applied = 0
    skipped = 0
    for i in range(layers):
        attn = getattr(model.model.layers[i], 'self_attn', None)
        cfg = getattr(attn, 'config', None)
        if cfg is None:
            skipped += 1
            continue
        cfg.window_size = window_sizes[i]
        cfg.max_capacity_prompt = max_capacity_prompts[i]
        cfg.kernel_size = kernel_sizes[i]
        cfg.pooling = pooling
        cfg.block_size = block_size
        applied += 1
    print(f'[SnapKV] Runtime config applied to {applied} layers; skipped {skipped} layers.')


@torch.inference_mode()
def get_pred_single_gpu(data, max_length, max_gen,
                        prompt_format, dataset, model_name,
                        model2path, out_path,
                        compress=False,
                        window_sizes=None,
                        max_capacity_prompts=None,
                        kernel_sizes=None,
                        pooling=None,
                        block_size=64,
                        use_cacheshrink=False,
                        cacheshrink_ratio=2.0,
                        cacheshrink_method='auto',
                        cacheshrink_group_size=4,
                        cacheshrink_skip_early_layers=0,
                        cacheshrink_keep_early_original=False,
                        cacheshrink_use_calibration=False,
                        cacheshrink_calib_samples=128,
                        cacheshrink_calib_len=512,
                        cacheshrink_dtype='bfloat16',
                        cacheshrink_verbose=False):
    model, tokenizer = load_model_and_tokenizer(
        model2path[model_name],
        model_name,
        device='cuda',
        compress=compress,
        use_cacheshrink=use_cacheshrink,
        cacheshrink_ratio=cacheshrink_ratio,
        cacheshrink_method=cacheshrink_method,
        cacheshrink_group_size=cacheshrink_group_size,
        cacheshrink_skip_early_layers=cacheshrink_skip_early_layers,
        cacheshrink_keep_early_original=cacheshrink_keep_early_original,
        cacheshrink_use_calibration=cacheshrink_use_calibration,
        cacheshrink_calib_samples=cacheshrink_calib_samples,
        cacheshrink_calib_len=cacheshrink_calib_len,
        cacheshrink_dtype=cacheshrink_dtype,
        cacheshrink_verbose=cacheshrink_verbose,
    )
    device = model.device

    # SnapKV config only needs to be set once after the model is loaded/converted.
    if compress:
        apply_snapkv_config(
            model,
            window_sizes=window_sizes,
            max_capacity_prompts=max_capacity_prompts,
            kernel_sizes=kernel_sizes,
            pooling=pooling,
            block_size=block_size,
        )

    printed = False
    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)

        # Truncate to fit max_length. We truncate in the middle, since both ends may contain key instructions.
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors='pt').input_ids[0]
        if "chatglm3" in model_name:
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors='pt', add_special_tokens=False).input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length / 2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + \
                     tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)

        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
            prompt = build_chat(tokenizer, prompt, model_name)

        if "chatglm3" in model_name:
            input = prompt.to(device)
        else:
            input = tokenizer(prompt, truncation=False, return_tensors='pt').to(device)

        context_length = input.input_ids.shape[-1]
        if not printed:
            print(prompt)
            printed = True

        if dataset == "samsum":
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length + 1,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
            )[0]
        else:
            output = model.generate(
                **input,
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

        with open(out_path, 'a', encoding='utf-8') as f:
            json.dump({
                "pred": pred,
                "answers": json_obj["answers"],
                "all_classes": json_obj["all_classes"],
                "length": json_obj["length"],
            }, f, ensure_ascii=False)
            f.write('\n')


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(path, model_name, device, compress=False,
                             use_cacheshrink=False,
                             cacheshrink_ratio=2.0,
                             cacheshrink_method='auto',
                             cacheshrink_group_size=4,
                             cacheshrink_skip_early_layers=0,
                             cacheshrink_keep_early_original=False,
                             cacheshrink_use_calibration=False,
                             cacheshrink_calib_samples=128,
                             cacheshrink_calib_len=512,
                             cacheshrink_dtype='bfloat16',
                             cacheshrink_verbose=False):
    if use_cacheshrink:
        # README quick-start path: pip install cacheshrink, then convert_to_mla(...).
        # Important: call replace_* before this function in __main__ when SnapKV is enabled,
        # so AutoModelForCausalLM.from_pretrained inside CacheShrink sees the SnapKV-patched classes.
        from cacheshrink import convert_to_mla

        dtype = str_to_torch_dtype(cacheshrink_dtype)
        print(
            f"[CacheShrink] convert_to_mla(path={path}, ratio={cacheshrink_ratio}, "
            f"method={cacheshrink_method}, dtype={cacheshrink_dtype}, "
            f"calibration={cacheshrink_use_calibration})"
        )
        # Some Mistral/LLaMA tokenizer.json files cannot be parsed by older
        # tokenizers wheels when AutoTokenizer defaults to the fast tokenizer.
        # CacheShrink internally calls AutoTokenizer.from_pretrained(), so we
        # temporarily force use_fast=False only during convert_to_mla().
        from transformers import AutoTokenizer as _HF_AutoTokenizer

        _orig_from_pretrained = _HF_AutoTokenizer.from_pretrained

        def _from_pretrained_force_slow(*args, **kwargs):
            kwargs["use_fast"] = False
            return _orig_from_pretrained(*args, **kwargs)

        _HF_AutoTokenizer.from_pretrained = _from_pretrained_force_slow
        try:
            model, tokenizer = convert_to_mla(
                path,
                compression_ratio=cacheshrink_ratio,
                compression_method=cacheshrink_method,
                cross_layer_group_size=cacheshrink_group_size,
                xkv_skip_early_layers=cacheshrink_skip_early_layers,
                keep_early_layers_original=cacheshrink_keep_early_original,
                device=device,
                dtype=dtype,
                use_calibration=cacheshrink_use_calibration,
                num_calibration_samples=cacheshrink_calib_samples,
                max_calibration_length=cacheshrink_calib_len,
                store_original_weights=False,
                verbose=cacheshrink_verbose,
            )
        finally:
            _HF_AutoTokenizer.from_pretrained = _orig_from_pretrained

        # Ensure the tokenizer used later in LongBench evaluation is also the
        # slow tokenizer, matching the original SnapKV script's Mistral branch.
        tokenizer_kwargs = {
            "use_fast": False,
            "trust_remote_code": ("chatglm" in model_name or "internlm" in model_name or "xgen" in model_name),
        }
        if "mistral" in model_name:
            tokenizer_kwargs["padding_side"] = "right"
        tokenizer = AutoTokenizer.from_pretrained(path, **tokenizer_kwargs)
        if "mistral" in model_name:
            tokenizer.padding_side = "right"
        model = model.eval()
        return model, tokenizer

    if "chatglm" in model_name or "internlm" in model_name or "xgen" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
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
        tokenizer = AutoTokenizer.from_pretrained(path, padding_side="right", use_fast=False)
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

    # 减小 maxlen，以免 OOM。可以按需改大/改小。
    max_length = min(max_length, 16384)

    os.makedirs("pred", exist_ok=True)
    os.makedirs("pred_e", exist_ok=True)

    dataset = args.dataset
    if args.compress_args_path:
        compress_args = json.load(open(os.path.join('config', args.compress_args_path), "r"))
        compress = True
        # SnapKV monkeypatch must happen before model construction. This also happens before
        # CacheShrink's convert_to_mla loads the model.
        replace_llama()
        replace_mistral()
        replace_mixtral()
    else:
        compress = False
        compress_args = None

    name_parts = [model_name]
    if args.compress_args_path:
        name_parts.append(args.compress_args_path.split(".")[0])
    if args.use_cacheshrink:
        name_parts.append(
            f"cacheshrink_{args.cacheshrink_method}_r{args.cacheshrink_ratio}"
            f"_g{args.cacheshrink_group_size}_skip{args.cacheshrink_skip_early_layers}"
        )
    write_model_name = "_".join(name_parts)

    if args.e:
        data = load_dataset('THUDM/LongBench', f"{dataset}_e", split='test', trust_remote_code=True)
        out_dir = f"pred_e/{write_model_name}"
    else:
        data = load_dataset('THUDM/LongBench', dataset, split='test', trust_remote_code=True)
        # Keep your original behavior: normal LongBench also writes into pred_e/.
        # Change to pred/ here if you want to separate non-E outputs.
        out_dir = f"pred_e/{write_model_name}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/{dataset}.jsonl"

    prompt_format = dataset2prompt[dataset]
    max_gen = dataset2maxlen[dataset]
    data_all = [data_sample for data_sample in data]

    common_kwargs = dict(
        use_cacheshrink=args.use_cacheshrink,
        cacheshrink_ratio=args.cacheshrink_ratio,
        cacheshrink_method=args.cacheshrink_method,
        cacheshrink_group_size=args.cacheshrink_group_size,
        cacheshrink_skip_early_layers=args.cacheshrink_skip_early_layers,
        cacheshrink_keep_early_original=args.cacheshrink_keep_early_original,
        cacheshrink_use_calibration=args.cacheshrink_use_calibration,
        cacheshrink_calib_samples=args.cacheshrink_calib_samples,
        cacheshrink_calib_len=args.cacheshrink_calib_len,
        cacheshrink_dtype=args.cacheshrink_dtype,
        cacheshrink_verbose=args.cacheshrink_verbose,
    )

    if compress_args is not None:
        get_pred_single_gpu(
            data_all, max_length, max_gen, prompt_format, dataset, model_name,
            model2path, out_path, compress, block_size=block_size,
            **compress_args, **common_kwargs
        )
    else:
        get_pred_single_gpu(
            data_all, max_length, max_gen, prompt_format, dataset, model_name,
            model2path, out_path, compress, block_size=block_size,
            **common_kwargs
        )
