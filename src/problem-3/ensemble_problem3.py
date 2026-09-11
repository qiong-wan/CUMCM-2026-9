"""Problem 3: stochastic plateau ensemble for the ambient boundary.

The deterministic model freezes the ambient temperature and moisture at their
last-30-minute means for ``t > 14400 s``.  This driver instead draws an AR(1)
(Ornstein-Uhlenbeck) fluctuation around that mean, with the amplitude and
correlation estimated from the detrended observation window, and solves an
ensemble of realizations to obtain a distribution of the critical drying time
``t_*``.  Outputs are written to the categorized ``output/problem-3/stochastic``
tree.

Run (trial)::

    python src/problem-3/ensemble_problem3.py --n 400 --factor 1 --members 8
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from solve_problem3 import (OUT, STOCHASTIC_DIR, Environment, audit, dump_json,
                            plateau_fluctuation_stats, simulate)

DATA_DIR = STOCHASTIC_DIR / "data"
ENSEMBLE_DIR = STOCHASTIC_DIR / "ensemble"
FIG_DIR = STOCHASTIC_DIR / "figures"
REPORT_DIR = STOCHASTIC_DIR / "reports"


def _import_matplotlib():
    """Import Matplotlib with an output-local config directory and CJK font."""
    os.environ["MPLCONFIGDIR"] = str(OUT / ".mplconfig")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def _summary(values):
    """Return descriptive statistics of an array."""
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "std": float(values.std(ddof=1)),
            "min": float(values.min()), "max": float(values.max()),
            "p05": float(np.percentile(values, 5)), "p25": float(np.percentile(values, 25)),
            "median": float(np.percentile(values, 50)),
            "p75": float(np.percentile(values, 75)), "p95": float(np.percentile(values, 95)),
            "count": int(values.size)}


def plot_plateau_paths(values, args, seeds):
    """Draw sample stochastic plateau realizations for both channels."""
    plt = _import_matplotlib()
    t = np.arange(14400.0, 14400.0 + 24 * 3600.0, 60.0)
    tail_mean = values[values[:, 0] >= 12600, 1:].mean(axis=0)
    fig, ax = plt.subplots(1, 2, figsize=(11.4, 4.1), layout="constrained")
    for seed in seeds:
        env = Environment(values, "fluct", seed=seed, tau_s=args.tau, sigma_scale=args.sigma_scale)
        ambient = np.array([env(s) for s in t])
        ax[0].plot(t / 3600, ambient[:, 0], linewidth=.9, label=f"种子 {seed}")
        ax[1].plot(t / 3600, ambient[:, 1], linewidth=.9, label=f"种子 {seed}")
    for a, label, level in [(ax[0], "空气温度 (°C)", tail_mean[0]),
                            (ax[1], "空气水分浓度 (kg/kg)", tail_mean[1])]:
        a.axhline(level, color="#444444", linestyle="--", linewidth=1, label="均值")
        a.set(xlabel="自烘干开始的时间 (h)", ylabel=label)
        a.grid(alpha=.2)
    ax[0].set_title("随机平台期：温度")
    ax[1].set_title("随机平台期：水分浓度")
    ax[0].legend(fontsize=7, ncol=2)
    fig.savefig(FIG_DIR / "plateau_paths.svg", bbox_inches="tight")
    fig.savefig(FIG_DIR / "plateau_paths.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_distribution(hours, reference_hours, args):
    """Draw the histogram of critical times with reference lines."""
    plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(7.2, 4.4), layout="constrained")
    ax.hist(hours, bins=min(20, max(4, len(hours))), color="#176c9b", alpha=.75,
            edgecolor="white", label="随机集合")
    ax.axvline(hours.mean(), color="#c23b34", linewidth=2, label=f"集合均值 {hours.mean():.3f} h")
    if reference_hours is not None:
        ax.axvline(reference_hours, color="#247c56", linestyle="--", linewidth=2,
                   label=f"确定性参考 {reference_hours:.3f} h")
    ax.set(xlabel="临界烘干时间 (h)", ylabel="频数",
           title=f"临界时间分布（{len(hours)} 个实现）")
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    fig.savefig(FIG_DIR / "critical_time_distribution.svg", bbox_inches="tight")
    fig.savefig(FIG_DIR / "critical_time_distribution.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    """Run a stochastic plateau ensemble and write categorized outputs."""
    parser = argparse.ArgumentParser(description="Problem 3 stochastic plateau ensemble.")
    parser.add_argument("--n", type=int, default=400)
    parser.add_argument("--factor", type=float, default=1.)
    parser.add_argument("--members", type=int, default=8)
    parser.add_argument("--seed0", type=int, default=0)
    parser.add_argument("--tau", type=float, default=None,
                        help="AR(1) correlation time [s]; default estimates the lag-1 "
                             "coefficient from the observed 60 s record.")
    parser.add_argument("--sigma-scale", type=float, default=1.,
                        help="Multiplier on the estimated fluctuation std.")
    parser.add_argument("--no-reference", action="store_true",
                        help="Skip the deterministic mean30 reference solve.")
    args = parser.parse_args()
    for folder in [DATA_DIR, ENSEMBLE_DIR, FIG_DIR, REPORT_DIR]:
        folder.mkdir(parents=True, exist_ok=True)

    values, _ = audit()
    stats = plateau_fluctuation_stats(values)
    tail = values[values[:, 0] >= 12600, 1:].mean(axis=0)
    ar1_phi = {key: (None if args.tau is not None else
                     float(np.clip(stats[key]["acf1"], 0.0, 0.95))) for key in ("T", "C")}
    dump_json(DATA_DIR / "tail_model.json", {
        "window_s": [12600, 14400], "tail_mean": tail.tolist(),
        "fluctuation_stats": stats, "tau_s": args.tau, "dt_env_s": 60.0,
        "ar1_phi": ar1_phi, "sigma_scale": args.sigma_scale,
        "model": "T_inf(t)=mean_T+sigma_T*z_T(t); C_inf(t)=mean_C+sigma_C*z_C(t); "
                 "z is a clipped AR(1) with unit variance; phi is the observed lag-1 "
                 "autocorrelation at the 60 s cadence and the innovations are bootstrapped "
                 "from the standardized observed residual, matching both frequency and amplitude",
        "detrending": "linear fit over the last observed 30 min"})

    started = time.perf_counter()
    records = []
    for member in range(args.members):
        seed = args.seed0 + member
        print(f"--- ensemble member {member + 1}/{args.members} (seed {seed}) ---", flush=True)
        meta = simulate(args.n, args.factor, "fluct", f"ens{member:03d}", values=values,
                        seed=seed, tau_s=args.tau, sigma_scale=args.sigma_scale,
                        save=False, verbose=False, progress=True)
        records.append({
            "member": member, "seed": seed,
            "critical_time_s": meta["event"]["estimate_s"],
            "strict_end_s": meta["event"]["end_s"],
            "end_max_C": meta["event"]["g_end"] + 0.15,
            "steps": meta["steps"], "rejected_steps": meta["rejected_steps"],
            "mass_relative_defect": meta["mass_relative_defect"],
            "heat_relative_defect": meta["heat_relative_defect"],
            "max_scaled_residual": meta["max_scaled_residual"],
            "moisture_min": meta["moisture_range"][0],
            "temperature_min": meta["temperature_range"][0],
            "temperature_max": meta["temperature_range"][1],
            "runtime_s": meta["runtime_s"]})
        print(f"member {member:02d} seed {seed}: t*={meta['event']['estimate_s']/3600:.4f} h", flush=True)

    reference = None
    if not args.no_reference:
        ref = simulate(args.n, args.factor, "mean30", "reference_mean30", values=values,
                       save=False, verbose=False)
        reference = {"mode": "mean30", "critical_time_s": ref["event"]["estimate_s"],
                     "critical_time_h": ref["event"]["estimate_s"] / 3600.0}
        print(f"reference mean30: t*={reference['critical_time_h']:.4f} h", flush=True)

    t_hours = np.array([r["critical_time_s"] for r in records]) / 3600.0
    summary = {"members": args.members, "N": args.n, "factor": args.factor,
               "tau_s": args.tau, "sigma_scale": args.sigma_scale,
               "critical_time_h": _summary(t_hours),
               "reference": reference,
               "wall_s": time.perf_counter() - started,
               "tail_model": (DATA_DIR / "tail_model.json").name}
    if reference is not None:
        summary["mean_minus_reference_h"] = float(t_hours.mean() - reference["critical_time_h"])
    dump_json(ENSEMBLE_DIR / "summary.json", summary)

    header = ["member", "seed", "critical_time_s", "critical_time_h", "strict_end_s",
              "end_max_C", "steps", "rejected_steps", "mass_relative_defect",
              "heat_relative_defect", "max_scaled_residual", "moisture_min",
              "temperature_min", "temperature_max", "runtime_s"]
    lines = [",".join(header)]
    for r in records:
        lines.append(",".join(str(r[k]) if k != "critical_time_h" else f"{r['critical_time_s']/3600.0:.12f}"
                              for k in header))
    (ENSEMBLE_DIR / "realizations.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    np.savez_compressed(ENSEMBLE_DIR / "realizations.npz",
                        seeds=np.array([r["seed"] for r in records]),
                        critical_time_s=np.array([r["critical_time_s"] for r in records]),
                        strict_end_s=np.array([r["strict_end_s"] for r in records]),
                        end_max_C=np.array([r["end_max_C"] for r in records]))

    plot_plateau_paths(values, args, [args.seed0 + k for k in range(min(5, args.members))])
    plot_distribution(t_hours, None if reference is None else reference["critical_time_h"], args)

    report = [
        "# 第三问平台期随机波动集合报告", "",
        f"对 {args.members} 个实现，每个实现用不同随机种子生成平台期（t>14400 s）的环境温度与水分浓度 AR(1) 波动，"
        f"网格 N={args.n}、时间倍率 {args.factor}，从同一均匀初值独立求解到全域达标。", "",
        "## 平台期波动模型", "",
        f"- 均值：温度 {tail[0]:.6f}°C，水分浓度 {tail[1]:.6f} kg/kg（12600–14400 s 算术均值）。",
        f"- 残差标准差（去线性趋势后）：温度 {stats['T']['sigma']:.6f}°C，水分 {stats['C']['sigma']:.8f} kg/kg。",
        f"- 一阶自相关：温度 {stats['T']['acf1']:.6f}，水分 {stats['C']['acf1']:.6f}。",
        f"- AR(1) 系数 phi：温度 {ar1_phi['T'] if ar1_phi['T'] is not None else '由 tau 给出'}，"
        f"水分 {ar1_phi['C'] if ar1_phi['C'] is not None else '由 tau 给出'}；"
        f"环境网格 60 s，波动放大系数 {args.sigma_scale:g}，标准差截断到 ±4。",
        f"- 残差幅度：峰度 温度 {stats['T']['kurtosis']:.2f}、水分 {stats['C']['kurtosis']:.2f}；"
        f"最大 |z| 温度 {stats['T']['max_abs_z']:.2f}、水分 {stats['C']['max_abs_z']:.2f}。"
        f"新息按该经验分布自助重采样，避免高斯尾尖峰。", "",
        "## 结果", "",
        f"- 临界时间均值 {summary['critical_time_h']['mean']:.6f} h，标准差 {summary['critical_time_h']['std']:.6f} h。",
        f"- 5%/50%/95% 分位：{summary['critical_time_h']['p05']:.6f} / "
        f"{summary['critical_time_h']['median']:.6f} / {summary['critical_time_h']['p95']:.6f} h。",
        f"- 最小/最大：{summary['critical_time_h']['min']:.6f} / {summary['critical_time_h']['max']:.6f} h。"]
    if reference is not None:
        report.append(f"- 确定性 mean30 参考：{reference['critical_time_h']:.6f} h；集合均值与其差 "
                      f"{summary['mean_minus_reference_h']:+.6f} h。")
    report += ["",
        "## 文件", "",
        "- `data/tail_model.json`：平台期均值、残差标准差、自相关与模型说明。",
        "- `ensemble/realizations.csv`、`ensemble/realizations.npz`：逐实现结果。",
        "- `ensemble/summary.json`：集合统计。",
        "- `figures/plateau_paths.svg|png`、`figures/critical_time_distribution.svg|png`：样本路径与临界时间分布。", "",
        "## 限制", "",
        "AR(1) 参数由 4 h 观测的末段估计，不能替代对真实烘房长期波动的实测标定；"
        "该集合只描述给定波动模型下的时长分布，不含模型结构误差，也不构成工艺安全余量。"]
    (REPORT_DIR / "stochastic_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary["critical_time_h"], ensure_ascii=False, indent=2))
    print(f"Wrote stochastic outputs under {STOCHASTIC_DIR}")


if __name__ == "__main__":
    main()
