"""Problem 3: standalone scientific figure generation.

Plotting is deliberately separated from the solver so that the numerical run
does not need Matplotlib and so that figures can be regenerated from saved
cases without recomputing.  All figure files (SVG plus QA raster) are written
to ``output/problem-3``.

This program is tolerant of a trial run: the three production-based figures are
always drawn, the mean-boundary curve is added when ``mean12800.npz`` exists,
and the convergence figure is added only when ``verification.json`` carries the
spatial/temporal study.  Pass ``--require-full`` to demand the complete,
PASS verification instead.

When the stochastic ensemble data exists (``stochastic/ensemble`` and
``stochastic/data``), a ``temperature_and_environment_seed0`` figure is written
for seed 0 by default (``--all-seeds`` generates every seed), reconstructing
the seeded AR(1) air temperature and moisture from the stored tail model
without re-solving.  Use ``--no-stochastic`` to skip it.

Four analytical figures are also produced from the saved data: a space-time
moisture heatmap with the 0.15 contour, a temperature-moisture phase portrait,
a numerical-error/boundary-scenario budget, and solver balance/boundary
diagnostics.  The spatial convergence figure has five mesh points and the
temporal one now shows all three time-step levels (1x, 0.5x, 0.25x).

Run::

    python src/problem-3/generate_figures.py
    python src/problem-3/generate_figures.py --require-full
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from solve_problem3 import (CASES_DIR, FIGURES_DIR, OUT, ROOT, STOCHASTIC_DIR,
                            VALIDATION_DIR, Environment, audit, case_environment,
                            compare, load_case)


def _import_matplotlib():
    """Import Matplotlib with an output-local config directory and CJK font."""
    import os
    os.environ["MPLCONFIGDIR"] = str(ROOT / "output/problem-3/.mplconfig")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def save_figure(fig, name):
    """Save an editable SVG and a QA raster, both inside output/problem-3."""
    fig.savefig(FIGURES_DIR / f"{name}.svg", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=140, bbox_inches="tight")
    plt = _import_matplotlib()
    plt.close(fig)


def _case_event_h(name):
    """Return a case's critical-time estimate in hours."""
    meta = json.loads((CASES_DIR / f"{name}.json").read_text(encoding="utf-8"))
    return meta["event"]["estimate_s"] / 3600.0


def figures(data, mean_data, report, values):
    """Draw the available figures and return the names that were written.

    ``data`` is the production case; ``mean_data`` may be ``None`` when the
    sensitivity case was skipped; ``report`` may be empty when verification was
    skipped.  The three production figures are always produced.
    """
    from matplotlib.ticker import FormatStrFormatter
    plt = _import_matplotlib()
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none"})
    made = []
    t = data["times"] / 3600
    fig, ax = plt.subplots(1, 2, figsize=(12.0, 4.3), layout="constrained")
    curves = [
        (t, data["outputs"][:, 1, 0], "#176c9b", "中心/全域最大值"),
        (data["history"][:, 0] / 3600, data["history"][:, 3], "#aa662e", "体积加权平均"),
        (t, data["outputs"][:, 1, -1], "#247c56", "表面"),
    ]
    for tt, yy, color, label in curves:
        ax[0].plot(tt, yy, color=color, label=label)
    ax[0].axhline(.15, color="#9f3546", linestyle="--", label="阈值 0.15")

    def cross_hour(tt, yy, level=0.15):
        """First crossing time (h) of a curve with the threshold."""
        idx = np.where((yy - level)[:-1] * (yy - level)[1:] <= 0)[0]
        if not len(idx):
            return None
        i = int(idx[0])
        return float(tt[i] + (level - yy[i]) / (yy[i + 1] - yy[i]) * (tt[i + 1] - tt[i]))

    for tt, yy, color, label in curves:
        tc = cross_hour(tt, yy)
        if tc is None:
            continue
        ax[0].plot([tc], [.15], "o", ms=9, mfc="none", mec=color, mew=2, zorder=5)
        ax[0].vlines(tc, 0.0, .15, colors=color, linestyles="--", linewidth=1, alpha=.85)
        ax[0].annotate(f"{tc:.2f} h", (tc, 0.0), textcoords="offset points",
                       xytext=(0, -2), ha="center", va="top", fontsize=7,
                       color=color, clip_on=False)
    ax[0].set(xlabel="自烘干开始的时间 (h)", ylabel="干基含水率 (kg/kg)",
              title="含水率轨迹（标注与 0.15 的交汇时间）")
    ax[0].set_ylim(bottom=0.0)
    ax[0].legend(fontsize=8)

    prod_meta = json.loads((CASES_DIR / "production.json").read_text(encoding="utf-8"))
    blue_label = {"last": "末值边界", "mean30": "末 30 min 均值边界"}.get(
        prod_meta.get("mode", "last"), f"随机波动（种子 {prod_meta.get('seed')}）")
    ax[1].plot(t, data["outputs"][:, 1, 0], color="#176c9b", label=blue_label)
    if mean_data is not None:
        ax[1].plot(mean_data["times"] / 3600, mean_data["outputs"][:, 1, 0],
                   color="#a45178", label="末 30 min 均值边界")
    ax[1].axhline(.15, color="#9f3546", linestyle="--")
    # Zoom into the threshold-crossing window; here the two curves separate
    # (they differ by only ~5e-6 kg/kg over the full endpoint range).
    xlo, xhi = 57.42, 57.48
    m2 = (t >= xlo) & (t <= xhi)
    vals = [data["outputs"][m2, 1, 0]]
    if mean_data is not None:
        tm = mean_data["times"] / 3600
        mm2 = (tm >= xlo) & (tm <= xhi)
        vals.append(mean_data["outputs"][mm2, 1, 0])
    allv = np.concatenate(vals)
    lo, hi = float(allv.min()), float(allv.max())
    pad = (hi - lo) * 0.12
    ax[1].set(xlim=(xlo, xhi), ylim=(lo - pad, hi + pad), xlabel="自烘干开始的时间 (h)",
              ylabel="最大含水率 (kg/kg)", title="终点与边界敏感性（交汇区放大）")
    ax[1].ticklabel_format(axis="y", style="plain", useOffset=False)
    ax[1].yaxis.set_major_formatter(FormatStrFormatter("%.6f"))
    ax[1].legend(fontsize=8, loc="upper left")
    tc_blue = cross_hour(t, data["outputs"][:, 1, 0])
    ax[1].plot([tc_blue], [.15], "o", ms=9, mfc="none", mec="#176c9b", mew=2, zorder=5)
    if mean_data is not None:
        tc_pink = cross_hour(mean_data["times"] / 3600, mean_data["outputs"][:, 1, 0])
        ax[1].plot([tc_pink], [.15], "o", ms=9, mfc="none", mec="#a45178", mew=2, zorder=5)
        ax[1].annotate("交汇时间差 %.2f s" % (abs(tc_pink - tc_blue) * 3600),
                       xy=(0.5 * (tc_blue + tc_pink), .15), xytext=(0.05, 0.14),
                       textcoords="axes fraction", fontsize=8, color="#333333",
                       arrowprops=dict(arrowstyle="->", color="#333333"))
    for a in ax:
        a.grid(alpha=.2)
    save_figure(fig, "drying_history")
    made.append("drying_history")

    fig, ax = plt.subplots(1, 2, figsize=(11.4, 4.1), layout="constrained")
    choose = [21600, 43200, 64800, 86400, 129600, 172800, data["full_times"][-1]]
    for t_value, color in zip(choose, ["#176c9b", "#aa662e", "#247c56", "#a45178", "#536d35", "#444444", "#c23b34"]):
        idx = int(np.argmin(abs(data["full_times"] - t_value)))
        C = data["full_states"][idx, 1]
        label = f"{t_value / 3600:.2f} h" if t_value == choose[-1] else f"{t_value / 3600:.0f} h"
        for a in ax:
            a.plot(data["r"] * 100, C, color=color, label=label)
    ax[0].set(xlabel="半径 (cm)", ylabel="含水率 (kg/kg)", title="完整径向场", xlim=(0, 2))
    ax[1].set(xlabel="半径 (cm)", ylabel="含水率 (kg/kg)", title="表面薄层放大", xlim=(1.97, 2), ylim=(.045, .21))
    ax[0].legend(ncol=2, fontsize=8)
    for a in ax:
        a.grid(alpha=.2)
        a.axhline(.15, color="#888888", linestyle="--", linewidth=.8)
    save_figure(fig, "radial_profiles")
    made.append("radial_profiles")

    prod_meta = json.loads((CASES_DIR / "production.json").read_text(encoding="utf-8"))
    env = case_environment(values, prod_meta)
    grid = np.arange(0.0, 8 * 3600 + 60, 60.0)
    ambient = np.array([env(s) for s in grid])
    tail_mean = Environment(values, "mean30").tail
    tail_last = Environment(values, "last").tail
    mode_name = {"last": "末值", "mean30": "末 30 min 均值"}.get(
        prod_meta.get("mode", "last"), f"随机波动（种子 {prod_meta.get('seed')}）")

    fig, ax = plt.subplots(1, 2, figsize=(12.4, 4.5), layout="constrained")
    sel = t <= 8
    ax[0].plot(t[sel], data["outputs"][sel, 0, 0], color="#176c9b", label="材料中心（求解）")
    ax[0].plot(t[sel], data["outputs"][sel, 0, -1], color="#247c56", label="材料表面（求解）")
    ax[0].plot(grid / 3600, ambient[:, 0], color="#aa662e", linewidth=1.3,
               label=f"空气边界：{mode_name}")
    ax[0].axhline(tail_mean[0], color="#aa662e", linestyle="--", linewidth=1,
                  label="末 30 min 均值（随机边界均值）")
    ax[0].axhline(tail_last[0], color="#9f3546", linestyle=":", linewidth=1,
                  label="末值延拓（仅对照，未用于求解）")
    ax[0].set(xlabel="自烘干开始的时间 (h)", ylabel="温度 (°C)",
              title="温度：材料与空气边界")
    ax[0].legend(fontsize=8, loc="lower right")
    ax[0].text(0.03, 0.12,
               "0–4 h：附件 1 实测分段线性插值\n4 h 后：AR(1) 随机波动边界\n"
               "（种子 0，均值 = 末 30 min 均值）",
               transform=ax[0].transAxes, fontsize=8, color="#333333",
               bbox=dict(boxstyle="round", fc="#f7f7f7", ec="#cccccc"))

    ax[1].plot(values[:, 0] / 3600, values[:, 2], color="#444444",
               label="实测空气水分浓度（0–4 h）")
    ax[1].plot(grid / 3600, ambient[:, 1], color="#aa662e", linewidth=1.3,
               label=f"空气边界：{mode_name}")
    ax[1].axhline(tail_mean[1], color="#aa662e", linestyle="--", linewidth=1,
                  label="末 30 min 均值（随机边界均值）")
    ax[1].axhline(tail_last[1], color="#9f3546", linestyle=":", linewidth=1,
                  label="末值延拓（仅对照，未用于求解）")
    ax[1].set(xlabel="自烘干开始的时间 (h)", ylabel="空气水分浓度 (kg/kg)",
              title="空气水分浓度边界")
    ax[1].legend(fontsize=8, loc="lower right")
    ax[1].text(0.03, 0.12,
               "观测区间保持不变；\n随机波动均值 = 末 30 min 均值",
               transform=ax[1].transAxes, fontsize=8, color="#333333",
               bbox=dict(boxstyle="round", fc="#f7f7f7", ec="#cccccc"))
    for a in ax:
        a.axvline(4, color="#888888", linestyle=":")
        a.grid(alpha=.2)
    save_figure(fig, "temperature_and_environment")
    made.append("temperature_and_environment")

    spatial, temporal = report.get("space_convergence"), report.get("time_convergence")
    if spatial and temporal:
        fig, ax = plt.subplots(1, 2, figsize=(13.2, 5.0), layout="constrained")
        # --- spatial: critical time as slim bars (primary axis); the adjacent
        #     pair difference as points at the pair midpoints (secondary axis) ---
        mesh_N = [400, 800, 1600, 3200, 6400, 12800]
        xpos = np.arange(len(mesh_N), dtype=float)
        xmid = xpos[:-1] + 0.5
        tstar = np.array([_case_event_h(f"space{n}") for n in mesh_N])
        dts = np.array([abs(v["event_difference_s"]) for v in spatial])
        order = float(np.log2(dts[-2] / dts[-1]))
        ax0 = ax[0]
        base = float(tstar.min()) - 0.005
        ax0.bar(xpos, tstar - base, bottom=base, width=0.4, color="#aa662e", alpha=.85)
        for x, y in zip(xpos, tstar):
            ax0.annotate(f"{y:.4f}", (x, y), textcoords="offset points",
                         xytext=(0, 4), ha="center", fontsize=8, color="#7a4a1e")
        ax0.set_ylim(base, float(tstar.max()) + 0.022)
        ax0.set_xticks(xpos)
        ax0.set_xticklabels([str(n) for n in mesh_N], fontsize=8)
        ax0.set(xlabel="径向区间数 N（差值点位于相邻网格对之间）",
                ylabel="临界时间 t* (h)",
                title=f"空间网格收敛（实测阶 p≈{order:.2f}，时间步固定 30 s）")
        ax0.grid(axis="y", which="both", alpha=.2)
        ax0b = ax0.twinx()
        ax0b.plot(xmid, dts, "o-", color="#176c9b")
        for i, (x, y) in enumerate(zip(xmid, dts)):
            last = i == len(xmid) - 1
            ax0b.annotate(f"{y:.3f} s", (x, y), textcoords="offset points",
                          xytext=(15, 6) if last else (8, 0),
                          ha="right" if last else "left",
                          va="bottom" if last else "center",
                          fontsize=8, color="#176c9b")
        ax0b.set_yscale("log")
        ax0b.set_ylabel("相邻网格对的临界时长差 |Δt*| (s)", color="#176c9b")
        ax0b.tick_params(axis="y", colors="#176c9b")
        ax0.annotate("交付采用 N=12800", xy=(xpos[-1], tstar[-1]),
                     xytext=(0.60, 0.84), textcoords="axes fraction",
                     color="#c23b34", fontsize=9,
                     arrowprops=dict(arrowstyle="->", color="#c23b34"))
        # --- temporal: relative field difference vs actual internal time step ---
        steps = np.array([30.0, 15.0, 7.5])
        factors = [1.0, 0.5, 0.25]
        names = ["space12800", "production", "time12800quarter"]
        diffs = np.array([
            0.0 if name == "time12800quarter"
            else compare(name, "time12800quarter")["all_saved_fields"]["C_max_kg_kg"]
            for name in names]) * 1e6
        ax[1].plot(steps, diffs, "o-", color="#a45178", label="相对 7.5 s 解的全场最大差")
        for x, y in zip(steps, diffs):
            ax[1].annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                           xytext=(0, 9), ha="center", fontsize=8, color="#a45178")
        ax[1].plot([steps[1]], [diffs[1]], "o", ms=12, mfc="none", mec="#c23b34", mew=2)
        ax[1].annotate("交付采用 15 s（0.5×）", xy=(steps[1], diffs[1]),
                       xytext=(0.55, 0.70), textcoords="axes fraction",
                       color="#c23b34", fontsize=9,
                       arrowprops=dict(arrowstyle="->", color="#c23b34"))
        ax[1].invert_xaxis()
        ax[1].set_xticks(steps)
        ax[1].set_xticklabels([f"{s:g} s\n({f:g}×)" for s, f in zip(steps, factors)])
        ax[1].set(xlabel="内部时间步上限（交付间隔 60 s）",
                  ylabel=r"相对 7.5 s 解的全场含水率最大差 ($10^{-6}$ kg/kg)",
                  title="时间步收敛（固定 N=12800，三个时间步）")
        ax[1].grid(alpha=.2)
        ax[1].text(0.03, 0.72, "网格固定为 N=12800；\n以最细 7.5 s 解为参考",
                   transform=ax[1].transAxes, fontsize=8, color="#444444",
                   bbox=dict(boxstyle="round", fc="#f5f5f5", ec="#cccccc"))
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
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none"})
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
            a.axvspan(0, 4, color="#9f3546", alpha=.07, label="观测段 0–4 h")
            a.axvline(4, color="#888888", linestyle=":", linewidth=1)
            a.grid(alpha=.2)
        ax[0].plot(grid / 3600, ambient[:, 0], color="#176c9b", linewidth=1.0,
                   label="空气温度")
        ax[0].axhline(mean[0], color="#aa662e", linestyle="--", linewidth=1, label="末 30 min 均值")
        ax[0].axhline(last[0], color="#247c56", linestyle=":", linewidth=1, label="末值")
        ax[0].set(xlabel="自烘干开始的时间 (h)", ylabel="空气温度 (°C)",
                  title=f"随机种子 {seed}：空气温度")
        ax[0].legend(fontsize=8)
        ax[1].plot(grid / 3600, ambient[:, 1], color="#247c56", linewidth=1.0,
                   label="空气水分浓度")
        ax[1].axhline(mean[1], color="#aa662e", linestyle="--", linewidth=1, label="末 30 min 均值")
        ax[1].axhline(last[1], color="#176c9b", linestyle=":", linewidth=1, label="末值")
        ax[1].set(xlabel="自烘干开始的时间 (h)", ylabel="空气水分浓度 (kg/kg)",
                  title=f"随机种子 {seed}：空气水分浓度")
        ax[1].legend(fontsize=8)
        name = f"temperature_and_environment_seed{seed}"
        save_figure(fig, name)
        made.append(name)
    return made


def moisture_space_time(data):
    """Space-time heatmap of the moisture field with the 0.15 contour."""
    plt = _import_matplotlib()
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none"})
    import math
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    from matplotlib.ticker import NullFormatter
    from scipy.interpolate import RectBivariateSpline
    t = data["times"] / 3600.0
    r = np.linspace(0.0, 2.0, data["outputs"].shape[2])
    C = data["outputs"][:, 1, :]
    end = float(t[-1])
    # Refine the coarse 21-node radial field before plotting so the sharp 0.15
    # colour transition does not show the radial grid as stair steps.
    r_fine = np.linspace(0.0, 2.0, 201)
    C_fine = np.clip(RectBivariateSpline(r, t, C.T, kx=3, ky=1)(r_fine, t), 0.0, 2.55)
    lo, hi, ctr = 0.05, 2.55, 0.15
    # Smooth diverging palette with the turning point at 0.15: orange-red below,
    # blue above, blended through a pale band.  The turn is placed at the
    # log-scale position of 0.15 so the colour-bar spacing stays logarithmic
    # while its labels are unchanged.
    pos = (math.log(ctr) - math.log(lo)) / (math.log(hi) - math.log(lo))
    turn = pos * 0.985
    cmap = LinearSegmentedColormap.from_list("dry_wet", [
        (0.0, "#7f0000"), (turn * 0.45, "#d73027"), (turn * 0.80, "#f46d43"),
        (turn, "#fdae61"), (min(turn + 0.008, 1.0), "#c6dbef"),
        (turn + (1 - turn) * 0.12, "#9ecae1"), (turn + (1 - turn) * 0.24, "#6baed6"),
        (turn + (1 - turn) * 0.50, "#3182bd"), (turn + (1 - turn) * 0.75, "#08519c"),
        (1.0, "#08306b")])
    norm = LogNorm(vmin=lo, vmax=hi)
    yticks = np.round(np.arange(0.0, 2.01, 0.1), 1)
    ylabels = ["0（中心）"] + [f"{y:.1f}" for y in yticks[1:]]
    fig, ax = plt.subplots(1, 2, figsize=(12.4, 5.0), layout="constrained")
    image = None
    extent = [t[0], t[-1], r_fine[0], r_fine[-1]]
    for a, title in [(ax[0], "含水率时空演化（全程）"), (ax[1], "含水率时空演化（末 8 h）")]:
        # Interpolated image instead of cell shading to avoid a visible mesh.
        image = a.imshow(C_fine, extent=extent, origin="lower", aspect="auto",
                         cmap=cmap, norm=norm, interpolation="bilinear")
        a.contour(t, r_fine, C_fine, levels=[ctr], colors="#111111", linewidths=1.4, linestyles="--")
        a.axvline(end, color="#111111", linewidth=1.2)
        a.set(xlabel="自烘干开始的时间 (h)", ylabel="半径 (cm)", title=title)
        a.set_yticks(yticks)
        a.set_yticklabels(ylabels, fontsize=7)
    ax[1].set_xlim(end - 8, end)
    # Repeat the text cues on the full-history panel as well.
    ax[0].annotate("0.15 等值线", xy=(end * 0.50, 1.30), color="#111111", fontsize=9)
    ax[0].annotate("中心最后达标", xy=(end, 0.0), xytext=(end * 0.50, 0.45),
                   color="#111111", fontsize=9,
                   arrowprops=dict(arrowstyle="->", color="#111111"))
    ax[1].annotate("0.15 等值线", xy=(end - 4, 0.5), color="#111111", fontsize=9)
    ax[1].annotate("中心最后达标", xy=(end, 0.0), xytext=(end - 7.6, 0.4),
                   color="#111111", fontsize=9,
                   arrowprops=dict(arrowstyle="->", color="#111111"))
    cb = fig.colorbar(image, ax=ax, location="right", shrink=0.92, pad=0.015)
    cb.set_label("干基含水率 (kg/kg)")
    cb.set_ticks([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 2.55])
    cb.ax.set_yticklabels(["0.05", "0.1", "0.2", "0.5", "1.0", "2.0", "2.55"])
    cb.ax.axhline(ctr, color="#111111", linewidth=1.2)
    cb.ax.text(0.5, ctr, "0.15", transform=cb.ax.get_yaxis_transform(),
               ha="center", va="bottom", color="#111111", fontsize=8)
    cb.ax.yaxis.set_minor_formatter(NullFormatter())
    save_figure(fig, "moisture_space_time")
    return "moisture_space_time"


def phase_portrait(data):
    """Temperature-moisture phase trajectories coloured by elapsed time."""
    plt = _import_matplotlib()
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none"})
    t = data["times"] / 3600.0
    fig, ax = plt.subplots(1, 2, figsize=(11.6, 4.4), layout="constrained")
    sc = None
    for a, idx, name in [(ax[0], 0, "中心"), (ax[1], -1, "表面")]:
        T, C = data["outputs"][:, 0, idx], data["outputs"][:, 1, idx]
        sc = a.scatter(T, C, c=t, cmap="plasma", s=7, linewidths=0, rasterized=True)
        a.axhline(0.15, color="#9f3546", linestyle="--", linewidth=1.2)
        a.annotate("起点", (T[0], C[0]), textcoords="offset points", xytext=(6, 4), fontsize=9)
        a.annotate("终点", (T[-1], C[-1]), textcoords="offset points", xytext=(-28, 6), fontsize=9)
        a.set(xlabel="温度 (°C)", ylabel="干基含水率 (kg/kg)", title=f"{name}相轨迹（温度–含水率）")
        a.grid(alpha=.2)
    cb = fig.colorbar(sc, ax=ax, location="right", shrink=0.92, pad=0.015)
    cb.set_label("自烘干开始的时间 (h)")
    save_figure(fig, "phase_portrait")
    return "phase_portrait"


def error_budget(report):
    """Log-scale bars comparing numerical error and scenario shift."""
    plt = _import_matplotlib()
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none"})
    e = report.get("empirical_error", {})
    s = report.get("boundary_sensitivity", {})
    items = [
        ("空间最细两级时长差", abs(e.get("space_last_difference_s", 0.0)), "#176c9b"),
        ("空间外推剩余估计", abs(e.get("space_fine_residual_time_estimate_s", 0.0)), "#5aa0c8"),
        ("时间最细两级时长差", abs(e.get("time_last_difference_s", 0.0)), "#247c56"),
        ("保守数值误差尺度", abs(e.get("conservative_numeric_allowance_s", 0.0)), "#aa662e"),
        ("边界情景差（末值 vs 末 30 min 均值）", abs(s.get("difference_s", 0.0)), "#c23b34"),
    ]
    items = items[::-1]
    labels = [x[0] for x in items]
    vals = [max(x[1], 1e-5) for x in items]
    colors = [x[2] for x in items]
    fig, ax = plt.subplots(figsize=(9.4, 4.4), layout="constrained")
    bars = ax.barh(labels, vals, color=colors, alpha=.88)
    ax.set_xscale("log")
    ax.set(xlabel="时长尺度 (s，对数坐标)", title="数值误差与边界情景差异量级")
    ax.grid(axis="x", which="both", alpha=.25)
    for bar, val in zip(bars, vals):
        ax.text(val * 1.18, bar.get_y() + bar.get_height() / 2, f"{val:.4g} s",
                va="center", fontsize=9)
    save_figure(fig, "error_budget")
    return "error_budget"


def solver_diagnostics(report):
    """Discrete-balance defects and boundary reconstruction convergence."""
    plt = _import_matplotlib()
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none"})
    names = ["space400", "space800", "space1600", "space3200", "space6400",
             "space12800", "production"]
    metas = [json.loads((CASES_DIR / f"{n}.json").read_text(encoding="utf-8")) for n in names]
    x = np.arange(len(names))
    width = 0.38
    fig, ax = plt.subplots(1, 2, figsize=(12.8, 4.4), layout="constrained")
    ax[0].bar(x - width / 2, [m["mass_relative_defect"] for m in metas], width,
              label="水分独立平衡缺陷", color="#176c9b")
    ax[0].bar(x + width / 2, [m["heat_relative_defect"] for m in metas], width,
              label="变热容热平衡缺陷", color="#aa662e")
    ax[0].axhline(1e-7, color="#c23b34", linestyle="--", linewidth=1, label="验收阈值 1e-7")
    ax[0].set_yscale("log")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([n.replace("space", "N=") for n in names], rotation=30, ha="right")
    ax[0].set(ylabel="相对缺陷（对数）", title="离散平衡缺陷（各案例）")
    ax[0].legend(fontsize=8)
    ax[0].grid(axis="y", which="both", alpha=.2)

    bc = report.get("boundary_convergence", [])
    Nvals = np.array([400, 800, 1600, 3200, 6400, 12800], dtype=float)
    RT = np.array([b["delivery_abs_max_RT_RC_symT_symC"][0] for b in bc])
    RC = np.array([b["delivery_abs_max_RT_RC_symT_symC"][1] for b in bc])
    rt_tol = report.get("production_boundaries", {}).get("rt_tolerance_W_m2", 1e-3)
    rc_tol = 1e-9
    ax[1].loglog(Nvals, RT, "o-", color="#176c9b", label="热 Robin 残差 (W/m²)")
    ax[1].loglog(Nvals, RC, "s-", color="#247c56", label="湿 Robin 残差 [(kg/kg)·m/s]")
    ax[1].axhline(rt_tol, color="#176c9b", linestyle=":", linewidth=1)
    ax[1].axhline(rc_tol, color="#247c56", linestyle=":", linewidth=1)
    ax[1].text(Nvals[0] * 1.05, rt_tol * 1.3,
               f"热 Robin 验收阈值 $10^{{{int(round(np.log10(rt_tol)))}}}$",
               color="#176c9b", fontsize=8, va="bottom")
    ax[1].text(Nvals[-1] * 0.98, rc_tol * 1.3,
               f"湿 Robin 验收阈值 $10^{{{int(round(np.log10(rc_tol)))}}}$",
               color="#247c56", fontsize=8, va="bottom", ha="right")
    order = float(-np.polyfit(np.log(Nvals), np.log(RT), 1)[0])
    ax[1].annotate(f"热 Robin 实测阶 ≈ {order:.2f}", xy=(Nvals[-1], RT[-1]),
                   xytext=(0.42, 0.60), textcoords="axes fraction", fontsize=9,
                   color="#176c9b", arrowprops=dict(arrowstyle="->", color="#176c9b"))
    ax[1].set(xlabel="径向区间数 N", ylabel="边界重构残差（对数）", title="边界重构收敛")
    ax[1].legend(fontsize=8)
    ax[1].grid(which="both", alpha=.2)
    save_figure(fig, "solver_diagnostics")
    return "solver_diagnostics"


def main():
    """Regenerate figures from saved cases; partial data is allowed by default."""
    parser = argparse.ArgumentParser(description="Generate Problem 3 figures from saved cases.")
    parser.add_argument("--require-full", action="store_true",
                        help="Require a PASS verification with spatial/temporal convergence.")
    parser.add_argument("--no-stochastic", action="store_true",
                        help="Skip the per-seed fluct air figures even when data exists.")
    parser.add_argument("--all-seeds", action="store_true",
                        help="Generate one fluct figure per seed; default is seed 0 only.")
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
    made.append(moisture_space_time(data))
    made.append(phase_portrait(data))
    if "empirical_error" in report:
        made.append(error_budget(report))
    if report.get("boundary_convergence"):
        made.append(solver_diagnostics(report))
    skipped = [n for n in ("drying_history", "radial_profiles",
                           "temperature_and_environment", "convergence",
                           "moisture_space_time", "phase_portrait",
                           "error_budget", "solver_diagnostics") if n not in made]
    stochastic = []
    npz_path = STOCHASTIC_DIR / "ensemble" / "realizations.npz"
    tail_path = STOCHASTIC_DIR / "data" / "tail_model.json"
    if not args.no_stochastic and npz_path.exists() and tail_path.exists():
        seeds = np.load(npz_path)["seeds"].tolist()
        if not args.all_seeds:
            seeds = seeds[:1]  # by default only seed 0
        tail = json.loads(tail_path.read_text(encoding="utf-8"))
        stochastic = stochastic_air_figures(values, seeds, tail.get("tau_s"), tail["sigma_scale"],
                                           float(data["times"][-1]), tail.get("dt_env_s", 60.0))
    print(f"Wrote {len(made) + len(stochastic)} figure(s) to {FIGURES_DIR}: "
          + ", ".join(made + stochastic)
          + (f". Skipped (missing data): {', '.join(skipped)}." if skipped else "")
          + (f" Per-seed fluct figures: {len(stochastic)}." if stochastic else ""))


if __name__ == "__main__":
    main()
