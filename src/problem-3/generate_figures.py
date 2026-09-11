"""Problem 3: standalone scientific figure generation.

Plotting is deliberately separated from the solver so that the numerical run
does not need Matplotlib and so that figures can be regenerated from saved
cases without recomputing.  All figure files (SVG plus QA raster) are written
to ``output/problem-3/image``.

This program is tolerant of a trial run: the three production-based figures are
always drawn, the mean-boundary curve is added when ``mean12800.npz`` exists,
and the convergence figure is added only when ``verification.json`` carries the
spatial/temporal study.  Pass ``--require-full`` to demand the complete,
PASS verification instead.

When the stochastic ensemble data exists (``stochastic/ensemble`` and
``stochastic/data``), one ``temperature_and_environment_seed{seed}`` figure is
also written per seed, reconstructing the seeded AR(1) air temperature and
moisture from the stored tail model without re-solving.  Use
``--no-stochastic`` to skip them.

Run::

    python src/problem-3/generate_figures.py
    python src/problem-3/generate_figures.py --require-full
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from plot_style import configure_chinese

from solve_problem3 import (CASES_DIR, FIGURES_DIR, OUT, ROOT, STOCHASTIC_DIR,
                            VALIDATION_DIR, Environment, audit, load_case)


def _import_matplotlib():
    """Import Matplotlib with an output-local config directory (lazy import)."""
    import os
    os.environ["MPLCONFIGDIR"] = str(VALIDATION_DIR / ".mplconfig")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def save_figure(fig, name):
    """Save Chinese SVG/PNG figures and a JPEG visual-check preview."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    previews = FIGURES_DIR / "previews"
    previews.mkdir(exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.svg", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=140, bbox_inches="tight")
    fig.savefig(previews / f"{name}.jpg", dpi=115, bbox_inches="tight")
    plt = _import_matplotlib()
    plt.close(fig)


def figures(data, mean_data, report, values):
    """Draw the available figures and return the names that were written.

    ``data`` is the production case; ``mean_data`` may be ``None`` when the
    sensitivity case was skipped; ``report`` may be empty when verification was
    skipped.  The three production figures are always produced.
    """
    plt = _import_matplotlib()
    configure_chinese(plt)
    made = []
    t = data["times"] / 3600
    fig, ax = plt.subplots(1, 2, figsize=(11.4, 4.1), layout="constrained")
    ax[0].plot(t, data["outputs"][:, 1, 0], label="中心／全域最大值", color="#176c9b")
    ax[0].plot(data["history"][:, 0] / 3600, data["history"][:, 3], label="体积加权平均值", color="#aa662e")
    ax[0].plot(t, data["outputs"][:, 1, -1], label="表面", color="#247c56")
    ax[0].axhline(.15, color="#9f3546", linestyle="--", label="达标阈值 0.15")
    ax[0].set(xlabel="烘干时间（小时）", ylabel="干基含水率（kg/kg）", title="含水率变化过程")
    ax[0].legend(fontsize=8)
    ax[1].plot(t, data["outputs"][:, 1, 0], color="#176c9b", label="观测末值延拓")
    if mean_data is not None:
        ax[1].plot(mean_data["times"] / 3600, mean_data["outputs"][:, 1, 0],
                   color="#a45178", label="末段30分钟均值延拓")
    ax[1].axhline(.15, color="#9f3546", linestyle="--")
    ax[1].set(xlim=(55.5, 58), ylim=(.148, .153), xlabel="烘干时间（小时）",
              ylabel="全域最大含水率（kg/kg）", title="结束时刻与边界敏感性")
    ax[1].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=.2)
    save_figure(fig, "drying_history")
    made.append("drying_history")

    fig, ax = plt.subplots(1, 2, figsize=(11.4, 4.1), layout="constrained")
    choose = [21600, 43200, 64800, 86400, 129600, 172800, data["full_times"][-1]]
    for t_value, color in zip(choose, ["#176c9b", "#aa662e", "#247c56", "#a45178", "#536d35", "#444444", "#c23b34"]):
        idx = int(np.argmin(abs(data["full_times"] - t_value)))
        C = data["full_states"][idx, 1]
        label = f"{t_value / 3600:.2f} 小时" if t_value == choose[-1] else f"{t_value / 3600:.0f} 小时"
        for a in ax:
            a.plot(data["r"] * 100, C, color=color, label=label)
    ax[0].set(xlabel="径向距离（cm）", ylabel="含水率（kg/kg）", title="径向含水率分布", xlim=(0, 2))
    ax[1].set(xlabel="径向距离（cm）", ylabel="含水率（kg/kg）", title="表面附近含水率分布", xlim=(1.97, 2), ylim=(.045, .21))
    ax[0].legend(ncol=2, fontsize=8)
    for a in ax:
        a.grid(alpha=.2)
        a.axhline(.15, color="#888888", linestyle="--", linewidth=.8)
    save_figure(fig, "radial_profiles")
    made.append("radial_profiles")

    fig, ax = plt.subplots(1, 2, figsize=(11.4, 4.1), layout="constrained")
    selected = t <= 8
    ax[0].plot(t[selected], data["outputs"][selected, 0, 0], label="药材中心", color="#176c9b")
    ax[0].plot(t[selected], data["outputs"][selected, 0, -1], label="药材表面", color="#247c56")
    boundary_t = np.r_[values[:, 0], 8 * 3600]
    ax[0].plot(boundary_t / 3600, np.r_[values[:, 1], values[-1, 1]], label="烘房温度（末值延拓）", color="#aa662e", linewidth=1)
    ax[0].set(xlabel="时间（小时）", ylabel="温度（°C）", title="药材与烘房温度变化")
    ax[0].legend(fontsize=8)
    ax[1].plot(values[:, 0] / 3600, values[:, 2], color="#444444", label="环境水分浓度观测")
    for mode, color in [("last", "#176c9b"), ("mean30", "#a45178")]:
        env = Environment(values, mode)
        ax[1].plot([4, 8], [env.tail[1], env.tail[1]], color=color, label={"last": "观测末值延拓", "mean30": "末段均值延拓"}[mode])
    ax[1].set(xlabel="时间（小时）", ylabel="环境水分浓度（kg/kg）", title="环境水分浓度与延拓方案")
    ax[1].legend(fontsize=8)
    for a in ax:
        a.axvline(4, color="#888888", linestyle=":")
        a.grid(alpha=.2)
    save_figure(fig, "temperature_and_environment")
    made.append("temperature_and_environment")

    spatial, temporal = report.get("space_convergence"), report.get("time_convergence")
    if spatial and temporal:
        fig, ax = plt.subplots(1, 2, figsize=(11.4, 4.1), layout="constrained")
        ax[0].loglog([v["fine_N"] for v in spatial], [abs(v["event_difference_s"]) for v in spatial], "o-", color="#176c9b")
        ax[0].set(xlabel="加密网格径向区间数 N", ylabel="临界时间变化（秒）", title="空间加密（时间策略固定）")
        ax[1].loglog([v["factors"][1] for v in temporal], [v["all_saved_fields"]["C_max_kg_kg"] for v in temporal], "o-", color="#a45178")
        ax[1].set(xlabel="加密时间步倍率", ylabel="全场含水率最大差（kg/kg）", title="时间加密（N=12800）")
        for a in ax:
            a.grid(which="both", alpha=.2)
        save_figure(fig, "convergence")
        made.append("convergence")
    return made


def stochastic_air_figures(values, seeds, tau_s, sigma_scale, end_s, dt_env=60.0):
    """Draw one air temperature/moisture figure per stochastic seed.

    The observed interval is the measured record; beyond 14400 s the curve is
    the seeded AR(1) plateau reconstructed from the stored tail model, so no
    re-solve is needed.  The sampling matches the 60 s observation cadence so
    the high-frequency fluctuation is not aliased.  Figures are named
    ``temperature_and_environment_seed{seed}``.
    """
    plt = _import_matplotlib()
    configure_chinese(plt)
    grid = np.arange(0.0, end_s + dt_env, dt_env)
    mean = values[values[:, 0] >= 12600, 1:].mean(axis=0)
    last = values[-1, 1:]
    made = []
    for seed in seeds:
        env = Environment(values, "fluct", seed=seed, tau_s=tau_s,
                          sigma_scale=sigma_scale, dt_env=dt_env)
        ambient = np.array([env(t) for t in grid])
        fig, ax = plt.subplots(1, 2, figsize=(11.4, 4.1), layout="constrained")
        for a in ax:
            a.axvspan(0, 4, color="#9f3546", alpha=.07, label="观测区间（0～4小时）")
            a.axvline(4, color="#888888", linestyle=":", linewidth=1)
            a.grid(alpha=.2)
        ax[0].plot(grid / 3600, ambient[:, 0], color="#176c9b", linewidth=1.0,
                   label="烘房温度")
        ax[0].axhline(mean[0], color="#aa662e", linestyle="--", linewidth=1, label="末段30分钟均值")
        ax[0].axhline(last[0], color="#247c56", linestyle=":", linewidth=1, label="观测末值")
        ax[0].set(xlabel="烘干时间（小时）", ylabel="烘房温度（°C）",
                  title=f"种子{seed}：烘房温度")
        ax[0].legend(fontsize=8)
        ax[1].plot(grid / 3600, ambient[:, 1], color="#247c56", linewidth=1.0,
                   label="环境水分浓度")
        ax[1].axhline(mean[1], color="#aa662e", linestyle="--", linewidth=1, label="末段30分钟均值")
        ax[1].axhline(last[1], color="#176c9b", linestyle=":", linewidth=1, label="观测末值")
        ax[1].set(xlabel="烘干时间（小时）", ylabel="环境水分浓度（kg/kg）",
                  title=f"种子{seed}：环境水分浓度")
        ax[1].legend(fontsize=8)
        name = f"temperature_and_environment_seed{seed}"
        save_figure(fig, name)
        made.append(name)
    return made


def main():
    """Regenerate figures from saved cases; partial data is allowed by default."""
    parser = argparse.ArgumentParser(description="Generate Problem 3 figures from saved cases.")
    parser.add_argument("--require-full", action="store_true",
                        help="Require a PASS verification with spatial/temporal convergence.")
    parser.add_argument("--no-stochastic", action="store_true",
                        help="Skip the per-seed fluct air figures even when data exists.")
    args = parser.parse_args()
    report = {}
    if (VALIDATION_DIR / "verification.json").exists():
        report = json.loads((VALIDATION_DIR / "verification.json").read_text(encoding="utf-8"))
    if args.require_full and (report.get("status") != "PASS"
                              or not report.get("space_convergence")
                              or not report.get("time_convergence")):
        raise RuntimeError("full verification with convergence is required; "
                           "run the complete --all flow first")
    data, meta = load_case("production")
    mean_data = load_case("mean12800")[0] if (CASES_DIR / "mean12800.npz").exists() else None
    values, _ = audit()
    made = figures(data, mean_data, report, values)
    skipped = [n for n in ("drying_history", "radial_profiles",
                           "temperature_and_environment", "convergence") if n not in made]
    stochastic = []
    npz_path = STOCHASTIC_DIR / "ensemble" / "realizations.npz"
    tail_path = STOCHASTIC_DIR / "data" / "tail_model.json"
    if not args.no_stochastic and npz_path.exists() and tail_path.exists():
        seeds = np.load(npz_path)["seeds"].tolist()
        tail = json.loads(tail_path.read_text(encoding="utf-8"))
        stochastic = stochastic_air_figures(values, seeds, tail.get("tau_s"), tail["sigma_scale"],
                                           float(data["times"][-1]), tail.get("dt_env_s", 60.0))
    print(f"Wrote {len(made) + len(stochastic)} figure(s) to {FIGURES_DIR}: "
          + ", ".join(made + stochastic)
          + (f". Skipped (missing data): {', '.join(skipped)}." if skipped else "")
          + (f" Per-seed fluct figures: {len(stochastic)}." if stochastic else ""))


if __name__ == "__main__":
    main()
