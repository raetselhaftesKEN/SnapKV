import re
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def parse_step_log(log_path):
    """
    支持两种日志格式：

    1) 当前已有格式：
       [model step 10] total=9.581302 lm=7.440800 rec=2.140499 kl=0.189391

    2) 如果以后你加了 mu/logvar 统计，也支持：
       [model step 10] total=... lm=... rec=... kl=... mu_mean=... mu_std=... logvar_mean=... logvar_std=...
    """
    pattern_basic = re.compile(
        r"\[model step\s+(\d+)\]\s+"
        r"total=([-\d.eE+]+)\s+"
        r"lm=([-\d.eE+]+)\s+"
        r"rec=([-\d.eE+]+)\s+"
        r"kl=([-\d.eE+]+)"
    )

    pattern_extra = re.compile(
        r"\[model step\s+(\d+)\]\s+"
        r"total=([-\d.eE+]+)\s+"
        r"lm=([-\d.eE+]+)\s+"
        r"rec=([-\d.eE+]+)\s+"
        r"kl=([-\d.eE+]+)\s+"
        r"mu_mean=([-\d.eE+]+)\s+"
        r"mu_std=([-\d.eE+]+)\s+"
        r"logvar_mean=([-\d.eE+]+)\s+"
        r"logvar_std=([-\d.eE+]+)"
    )

    rows = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            m_extra = pattern_extra.search(line)
            if m_extra:
                rows.append({
                    "step": int(m_extra.group(1)),
                    "total": float(m_extra.group(2)),
                    "lm": float(m_extra.group(3)),
                    "rec": float(m_extra.group(4)),
                    "kl": float(m_extra.group(5)),
                    "mu_mean": float(m_extra.group(6)),
                    "mu_std": float(m_extra.group(7)),
                    "logvar_mean": float(m_extra.group(8)),
                    "logvar_std": float(m_extra.group(9)),
                })
                continue

            m_basic = pattern_basic.search(line)
            if m_basic:
                rows.append({
                    "step": int(m_basic.group(1)),
                    "total": float(m_basic.group(2)),
                    "lm": float(m_basic.group(3)),
                    "rec": float(m_basic.group(4)),
                    "kl": float(m_basic.group(5)),
                })

    if not rows:
        raise ValueError("没有在日志中匹配到任何 [model step ...] 行，请检查输入文件格式。")

    df = pd.DataFrame(rows).sort_values("step").reset_index(drop=True)
    return df


def plot_losses(df, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. KL 单独曲线
    plt.figure(figsize=(8, 5))
    plt.plot(df["step"], df["kl"], marker="o", markersize=2)
    plt.xlabel("Step")
    plt.ylabel("KL")
    plt.title("KL Curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "kl_curve.png", dpi=200)
    plt.close()

    # 2. total / lm / rec / kl 四条曲线
    plt.figure(figsize=(10, 6))
    plt.plot(df["step"], df["total"], label="total")
    plt.plot(df["step"], df["lm"], label="lm")
    plt.plot(df["step"], df["rec"], label="rec")
    plt.plot(df["step"], df["kl"], label="kl")
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.title("Training Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "training_curves.png", dpi=200)
    plt.close()

    # 3. KL 对数坐标，更方便看前期增长
    plt.figure(figsize=(8, 5))
    plt.plot(df["step"], df["kl"], marker="o", markersize=2)
    plt.yscale("log")
    plt.xlabel("Step")
    plt.ylabel("KL (log scale)")
    plt.title("KL Curve (Log Scale)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "kl_curve_log.png", dpi=200)
    plt.close()


def plot_mu_logvar_stats(df, out_dir):
    out_dir = Path(out_dir)

    needed = ["mu_mean", "mu_std", "logvar_mean", "logvar_std"]
    if not all(col in df.columns for col in needed):
        print("日志中没有 mu/logvar 统计项，跳过 mu/logvar 分布图。")
        return

    # 这里只能画“统计量随 step 的变化”，不是严格意义上的原始分布直方图
    plt.figure(figsize=(10, 6))
    plt.plot(df["step"], df["mu_mean"], label="mu_mean")
    plt.plot(df["step"], df["mu_std"], label="mu_std")
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.title("Mu Statistics")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "mu_stats.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(df["step"], df["logvar_mean"], label="logvar_mean")
    plt.plot(df["step"], df["logvar_std"], label="logvar_std")
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.title("Logvar Statistics")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "logvar_stats.png", dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", type=str, required=True, help="裁剪后的 step 日志 txt 文件")
    parser.add_argument("--out_dir", type=str, default="./step_plots", help="输出图片目录")
    args = parser.parse_args()

    df = parse_step_log(args.log_file)
    print(df.head())
    print(f"\n共解析到 {len(df)} 个 step 点")
    print(f"step 范围: {df['step'].min()} ~ {df['step'].max()}")
    print(f"KL 范围: {df['kl'].min():.6f} ~ {df['kl'].max():.6f}")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(args.out_dir) / "parsed_steps.csv", index=False)

    plot_losses(df, args.out_dir)
    plot_mu_logvar_stats(df, args.out_dir)

    print(f"\n结果已保存到: {args.out_dir}")


if __name__ == "__main__":
    main()

'''
python plot_step_log.py --log_file log/train_mistral_kv_vae_e2e_20260327_150923_cut.log --out_dir log/plots_train_mistral_kv_vae_e2e_20260327_150923_cut
'''