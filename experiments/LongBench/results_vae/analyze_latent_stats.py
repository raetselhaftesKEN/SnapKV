import os
import re
import json
import glob
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.stats import gaussian_kde
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def extract_step_from_filename(path):
    m = re.search(r"latent_step_(\d+)\.npz$", os.path.basename(path))
    if m is None:
        return None
    return int(m.group(1))


def load_latent_npz(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    out = {}

    if "mu" in data:
        out["mu"] = np.asarray(data["mu"]).reshape(-1)
    else:
        raise KeyError(f"{npz_path} 中没有 mu")

    if "logvar" in data:
        out["logvar"] = np.asarray(data["logvar"]).reshape(-1)
    else:
        raise KeyError(f"{npz_path} 中没有 logvar")

    meta = None
    if "meta" in data:
        try:
            meta = data["meta"].item()
        except Exception:
            meta = data["meta"]
    out["meta"] = meta
    out["step"] = extract_step_from_filename(npz_path)

    return out


def summarize_array(x, prefix):
    x = np.asarray(x).reshape(-1)
    return {
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_std": float(np.std(x)),
        f"{prefix}_min": float(np.min(x)),
        f"{prefix}_max": float(np.max(x)),
        f"{prefix}_p01": float(np.percentile(x, 1)),
        f"{prefix}_p05": float(np.percentile(x, 5)),
        f"{prefix}_p25": float(np.percentile(x, 25)),
        f"{prefix}_p50": float(np.percentile(x, 50)),
        f"{prefix}_p75": float(np.percentile(x, 75)),
        f"{prefix}_p95": float(np.percentile(x, 95)),
        f"{prefix}_p99": float(np.percentile(x, 99)),
    }


def build_summary_table(npz_files):
    rows = []
    for f in npz_files:
        d = load_latent_npz(f)
        row = {"step": d["step"], "file": os.path.basename(f)}
        row.update(summarize_array(d["mu"], "mu"))
        row.update(summarize_array(d["logvar"], "logvar"))
        rows.append(row)

    if not rows:
        raise ValueError("没有找到可用的 latent_step_*.npz 文件")

    df = pd.DataFrame(rows).sort_values("step").reset_index(drop=True)
    return df


def plot_histogram(x, title, xlabel, out_path, bins=80, density=True):
    plt.figure(figsize=(8, 5))
    plt.hist(x, bins=bins, density=density, alpha=0.8)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Density" if density else "Count")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_kde_or_density_line(x, title, xlabel, out_path, bins=120):
    x = np.asarray(x).reshape(-1)

    plt.figure(figsize=(8, 5))
    if HAS_SCIPY and len(x) >= 10:
        xs = np.linspace(np.min(x), np.max(x), 512)
        kde = gaussian_kde(x)
        ys = kde(xs)
        plt.plot(xs, ys)
        plt.fill_between(xs, ys, alpha=0.25)
        plt.ylabel("Density")
    else:
        hist, edges = np.histogram(x, bins=bins, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        plt.plot(centers, hist)
        plt.fill_between(centers, hist, alpha=0.25)
        plt.ylabel("Approx. Density")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_compare_steps(npz_files, key, out_path, max_steps_to_plot=6, bins=120):
    step_and_file = []
    for f in npz_files:
        step = extract_step_from_filename(f)
        if step is not None:
            step_and_file.append((step, f))
    step_and_file = sorted(step_and_file, key=lambda x: x[0])

    if len(step_and_file) == 0:
        return

    if len(step_and_file) > max_steps_to_plot:
        idx = np.linspace(0, len(step_and_file) - 1, max_steps_to_plot).round().astype(int)
        selected = [step_and_file[i] for i in idx]
    else:
        selected = step_and_file

    plt.figure(figsize=(9, 6))
    for step, f in selected:
        d = load_latent_npz(f)
        x = d[key]
        if HAS_SCIPY and len(x) >= 10:
            xs = np.linspace(np.min(x), np.max(x), 512)
            kde = gaussian_kde(x)
            ys = kde(xs)
            plt.plot(xs, ys, label=f"step {step}")
        else:
            hist, edges = np.histogram(x, bins=bins, density=True)
            centers = 0.5 * (edges[:-1] + edges[1:])
            plt.plot(centers, hist, label=f"step {step}")

    plt.title(f"{key} distribution comparison across steps")
    plt.xlabel(key)
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_stats_curve(df_summary, out_dir):
    # mu mean/std
    plt.figure(figsize=(9, 5))
    plt.plot(df_summary["step"], df_summary["mu_mean"], label="mu_mean")
    plt.plot(df_summary["step"], df_summary["mu_std"], label="mu_std")
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.title("Mu statistics across steps")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mu_stats_curve.png"), dpi=220)
    plt.close()

    # logvar mean/std
    plt.figure(figsize=(9, 5))
    plt.plot(df_summary["step"], df_summary["logvar_mean"], label="logvar_mean")
    plt.plot(df_summary["step"], df_summary["logvar_std"], label="logvar_std")
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.title("Logvar statistics across steps")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "logvar_stats_curve.png"), dpi=220)
    plt.close()

    # mu quantiles
    plt.figure(figsize=(9, 5))
    for c in ["mu_p01", "mu_p05", "mu_p25", "mu_p50", "mu_p75", "mu_p95", "mu_p99"]:
        plt.plot(df_summary["step"], df_summary[c], label=c)
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.title("Mu quantiles across steps")
    plt.legend(ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mu_quantiles_curve.png"), dpi=220)
    plt.close()

    # logvar quantiles
    plt.figure(figsize=(9, 5))
    for c in ["logvar_p01", "logvar_p05", "logvar_p25", "logvar_p50", "logvar_p75", "logvar_p95", "logvar_p99"]:
        plt.plot(df_summary["step"], df_summary[c], label=c)
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.title("Logvar quantiles across steps")
    plt.legend(ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "logvar_quantiles_curve.png"), dpi=220)
    plt.close()


def parse_step_log(log_path):
    pattern = re.compile(
        r"\[model step\s+(\d+)\]\s+"
        r"total=([-\d.eE+]+)\s+"
        r"lm=([-\d.eE+]+)\s+"
        r"rec=([-\d.eE+]+)\s+"
        r"kl=([-\d.eE+]+)"
        r"(?:\s+mu_mean=([-\d.eE+]+)\s+mu_std=([-\d.eE+]+)\s+logvar_mean=([-\d.eE+]+)\s+logvar_std=([-\d.eE+]+))?"
    )

    rows = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            m = pattern.search(line.strip())
            if not m:
                continue
            rows.append({
                "step": int(m.group(1)),
                "total": float(m.group(2)),
                "lm": float(m.group(3)),
                "rec": float(m.group(4)),
                "kl": float(m.group(5)),
                "mu_mean": None if m.group(6) is None else float(m.group(6)),
                "mu_std": None if m.group(7) is None else float(m.group(7)),
                "logvar_mean": None if m.group(8) is None else float(m.group(8)),
                "logvar_std": None if m.group(9) is None else float(m.group(9)),
            })

    if len(rows) == 0:
        raise ValueError("没有匹配到任何 [model step ...] 行")

    return pd.DataFrame(rows).sort_values("step").reset_index(drop=True)


def plot_training_curves(df_log, out_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(df_log["step"], df_log["total"], label="total")
    plt.plot(df_log["step"], df_log["lm"], label="lm")
    plt.plot(df_log["step"], df_log["rec"], label="rec")
    plt.plot(df_log["step"], df_log["kl"], label="kl")
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.title("Training curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_curves_from_log.png"), dpi=220)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(df_log["step"], df_log["kl"])
    plt.xlabel("Step")
    plt.ylabel("KL")
    plt.title("KL curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "kl_curve_from_log.png"), dpi=220)
    plt.close()

    if df_log["mu_mean"].notna().any():
        plt.figure(figsize=(9, 5))
        plt.plot(df_log["step"], df_log["mu_mean"], label="mu_mean")
        plt.plot(df_log["step"], df_log["mu_std"], label="mu_std")
        plt.xlabel("Step")
        plt.ylabel("Value")
        plt.title("Mu stats from log")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "mu_stats_from_log.png"), dpi=220)
        plt.close()

        plt.figure(figsize=(9, 5))
        plt.plot(df_log["step"], df_log["logvar_mean"], label="logvar_mean")
        plt.plot(df_log["step"], df_log["logvar_std"], label="logvar_std")
        plt.xlabel("Step")
        plt.ylabel("Value")
        plt.title("Logvar stats from log")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "logvar_stats_from_log.png"), dpi=220)
        plt.close()


def save_json_report(df_summary, out_dir):
    report = {
        "num_checkpoints": int(len(df_summary)),
        "step_min": int(df_summary["step"].min()),
        "step_max": int(df_summary["step"].max()),
        "mu_mean_range": [float(df_summary["mu_mean"].min()), float(df_summary["mu_mean"].max())],
        "mu_std_range": [float(df_summary["mu_std"].min()), float(df_summary["mu_std"].max())],
        "logvar_mean_range": [float(df_summary["logvar_mean"].min()), float(df_summary["logvar_mean"].max())],
        "logvar_std_range": [float(df_summary["logvar_std"].min()), float(df_summary["logvar_std"].max())],
    }
    with open(os.path.join(out_dir, "latent_summary_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_dir", type=str, required=True, help="保存 latent_step_*.npz 的目录")
    parser.add_argument("--out_dir", type=str, default="./latent_analysis", help="分析结果输出目录")
    parser.add_argument("--log_file", type=str, default=None, help="可选，训练日志或裁剪后的 step 日志")
    parser.add_argument("--bins", type=int, default=80)
    parser.add_argument("--max_compare_steps", type=int, default=6)
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    npz_files = sorted(glob.glob(os.path.join(args.latent_dir, "latent_step_*.npz")))
    if len(npz_files) == 0:
        raise ValueError(f"在目录 {args.latent_dir} 下没有找到 latent_step_*.npz")

    # 汇总表
    df_summary = build_summary_table(npz_files)
    df_summary.to_csv(os.path.join(args.out_dir, "latent_summary.csv"), index=False)
    save_json_report(df_summary, args.out_dir)

    # 取最早、中间、最晚三个 step 分别出单图
    step_file_pairs = [(extract_step_from_filename(f), f) for f in npz_files]
    step_file_pairs = [(s, f) for s, f in step_file_pairs if s is not None]
    step_file_pairs = sorted(step_file_pairs, key=lambda x: x[0])

    selected = []
    if len(step_file_pairs) >= 1:
        selected.append(step_file_pairs[0])
    if len(step_file_pairs) >= 3:
        selected.append(step_file_pairs[len(step_file_pairs) // 2])
    if len(step_file_pairs) >= 2:
        selected.append(step_file_pairs[-1])

    seen = set()
    selected_unique = []
    for s, f in selected:
        if s not in seen:
            selected_unique.append((s, f))
            seen.add(s)

    for step, f in selected_unique:
        d = load_latent_npz(f)

        plot_histogram(
            d["mu"],
            title=f"Mu histogram (step {step})",
            xlabel="mu",
            out_path=os.path.join(args.out_dir, f"mu_hist_step_{step}.png"),
            bins=args.bins,
            density=True,
        )
        plot_histogram(
            d["logvar"],
            title=f"Logvar histogram (step {step})",
            xlabel="logvar",
            out_path=os.path.join(args.out_dir, f"logvar_hist_step_{step}.png"),
            bins=args.bins,
            density=True,
        )

        plot_kde_or_density_line(
            d["mu"],
            title=f"Mu KDE/density (step {step})",
            xlabel="mu",
            out_path=os.path.join(args.out_dir, f"mu_kde_step_{step}.png"),
            bins=max(args.bins, 100),
        )
        plot_kde_or_density_line(
            d["logvar"],
            title=f"Logvar KDE/density (step {step})",
            xlabel="logvar",
            out_path=os.path.join(args.out_dir, f"logvar_kde_step_{step}.png"),
            bins=max(args.bins, 100),
        )

    # 多 step 对比
    plot_compare_steps(
        npz_files,
        key="mu",
        out_path=os.path.join(args.out_dir, "mu_compare_steps.png"),
        max_steps_to_plot=args.max_compare_steps,
        bins=max(args.bins, 100),
    )
    plot_compare_steps(
        npz_files,
        key="logvar",
        out_path=os.path.join(args.out_dir, "logvar_compare_steps.png"),
        max_steps_to_plot=args.max_compare_steps,
        bins=max(args.bins, 100),
    )

    # 统计量曲线
    plot_stats_curve(df_summary, args.out_dir)

    # 可选日志曲线
    if args.log_file is not None and os.path.exists(args.log_file):
        df_log = parse_step_log(args.log_file)
        df_log.to_csv(os.path.join(args.out_dir, "parsed_step_log.csv"), index=False)
        plot_training_curves(df_log, args.out_dir)

    print(f"分析完成，结果已保存到: {args.out_dir}")
    print(f"共读取 {len(npz_files)} 个 latent npz 文件")
    print(df_summary[[
        "step", "mu_mean", "mu_std", "logvar_mean", "logvar_std"
    ]].head())


if __name__ == "__main__":
    main()

'''
python analyze_latent_stats.py \
  --latent_dir ./mistral_kv_vae_e2e_wikitext_20260330/latent_stats \
  --out_dir ./latent_analysis_20260330
  
python analyze_latent_stats.py --latent_dir ./mistral_kv_vae_e2e_wikitext_20260330/latent_stats   --out_dir ./latent_analysis_20260330   --log_file ./train_mistral_kv_vae_e2e_20260330.log
  
  
python analyze_latent_stats.py --latent_dir ./mistral_kv_predictor_friendly/latent_stats   --out_dir ./latent_analysis_20260407   --log_file ./mistral_kv_predictor_friendly/train_mistral_kv_vae_e2e_20260407.log
''' 