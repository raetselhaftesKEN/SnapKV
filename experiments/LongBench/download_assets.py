"""
download_assets.py

功能：
- 在有网络的环境下，预先从 HuggingFace 下载指定的模型和 LongBench 数据集子任务，
  以便之后把缓存目录拷到无网服务器上使用 pred_snap.py。

用法示例：
    # 只下载 mistral-7B-instruct-v0.2 + qasper
    python download_assets.py --model mistral-7B-instruct-v0.2 --dataset qasper

    # 下载 mistral-7B-instruct-v0.2 + 多个数据集
    python download_assets.py --model mistral-7B-instruct-v0.2 --dataset qasper hotpotqa gov_report
"""

import os
import json
import argparse

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "llama2-7b-chat-4k", "longchat-v1.5-7b-32k", "xgen-7b-8k",
            "internlm-7b-8k", "chatglm2-6b", "chatglm2-6b-32k", "chatglm3-6b-32k",
            "vicuna-v1.5-7b-16k",
            "mistral-7B-instruct-v0.2", "mistral-7B-instruct-v0.1",
            "llama-2-7B-32k-instruct", "mixtral-8x7B-instruct-v0.1",
            "lwm-text-chat-1m", "lwm-text-1m"
        ],
        help="和 pred_snap.py 中 --model 一致的名字"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="+",
        default=["qasper"],
        help="需要预先下载的 LongBench 子数据集名，例如 qasper hotpotqa gov_report"
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="若指定，则下载 LongBench-E (后缀 _e)，否则下载普通 LongBench"
    )
    parser.add_argument(
        "--model2path",
        type=str,
        default="config/model2path.json",
        help="model2path.json 的路径（默认与 pred_snap.py 相同）"
    )
    return parser.parse_args()


def download_model_and_tokenizer(path: str, model_name: str):
    """
    仿照 pred_snap.py 中 load_model_and_tokenizer 的分支逻辑，
    只负责触发 from_pretrained 下载，不做推理。
    """
    print(f"[INFO] 开始下载模型和 tokenizer: {model_name} (path={path})")

    if "chatglm" in model_name or "internlm" in model_name or "xgen" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            path,
            trust_remote_code=True,
            torch_dtype="auto"
        )

    elif "llama2" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype="auto")

    elif "longchat" in model_name or "vicuna" in model_name:
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            device_map=None,
            use_cache=True,
            use_flash_attention_2=False,  # 这里只是下载，不需要真跑
        )
        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)

    elif "llama-2" in model_name or "lwm" in model_name:
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            device_map=None,
            use_cache=True,
            use_flash_attention_2=False,
        )
        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)

    elif "mistral" in model_name:
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            device_map=None,
            use_cache=True,
            use_flash_attention_2=False,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            path,
            padding_side="right",
            use_fast=False,
        )

    elif "mixtral" in model_name:
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            device_map=None,
            use_cache=True,
            use_flash_attention_2=False,
        )
        tokenizer = AutoTokenizer.from_pretrained(path)

    else:
        raise ValueError(f"Model {model_name} not supported!")

    # 释放内存，只保留本地缓存的权重文件
    del model
    del tokenizer

    print(f"[INFO] 模型 {model_name} 下载完成（已写入 HuggingFace 缓存目录）")


def download_longbench_subsets(datasets_list, eval_flag=False):
    """
    下载指定的 LongBench 子数据集。pred_snap.py 里只用到了 split='test'。
    """
    for ds in datasets_list:
        config_name = f"{ds}_e" if eval_flag else ds
        print(f"[INFO] 开始下载 LongBench 子数据集: config='{config_name}', split='test'")
        _ = load_dataset("THUDM/LongBench", config_name, split="test", trust_remote_code=True)
        print(f"[INFO] LongBench/{config_name} 下载完成（已写入 HuggingFace 缓存目录）")


def main():
    args = parse_args()

    # 1. 读 model2path.json，拿到对应的 HuggingFace repo / 本地路径
    if not os.path.exists(args.model2path):
        raise FileNotFoundError(f"找不到 {args.model2path}，请确认路径是否正确。")

    with open(args.model2path, "r", encoding="utf-8") as f:
        model2path = json.load(f)

    if args.model not in model2path:
        raise KeyError(f"model2path.json 中没有键：{args.model}")

    path = model2path[args.model]

    # 2. 下载模型和 tokenizer
    download_model_and_tokenizer(path, args.model)

    # 3. 下载 LongBench 数据集子任务
    download_longbench_subsets(args.dataset, eval_flag=args.eval)

    print("\n[DONE] 所有指定模型和数据集已下载到本机 HuggingFace 缓存目录。")
    print("之后可以将 ~/.cache/huggingface 整个目录打包上传到服务器，在服务器上即可离线运行 pred_snap.py。")


if __name__ == "__main__":
    main()
