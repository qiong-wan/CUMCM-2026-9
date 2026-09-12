"""Generate Chinese scientific figures from verified Problem 4 data only.

No solver import, PDE integration or mutation of numerical case files occurs.
PNG files are written but never read. JPEG previews come directly from Figures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")

from matplotlib import font_manager, patheffects, pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.text import Text
import numpy as np
import openpyxl


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/problem-4"
IMAGES = OUT / "image"
PREVIEWS = ROOT.parent / "tmp/problem4_figure_previews"
COLORS = ("#0072B2", "#009E73", "#E69F00", "#D55E00", "#CC79A7", "#56B4E9")
FIGURE_NAMES = (
    "01_inputs_and_radius",
    "02_moisture_history",
    "03_actual_radius_profiles",
    "04_moisture_space_time",
    "05_temperature_and_diffusion",
    "06_numerical_convergence",
    "07_scenarios_and_sensitivity",
    "08_ambient_fluctuation",
)
LABELS = {
    "appendix3_fixed": "附录 3 · 固定半径",
    "appendix3_shrink": "附录 3 · 实测收缩",
    "appendix4_fixed": "附录 4 · 固定半径",
    "appendix4_last": "附录 4 · 末值延拓",
    "appendix4_mean30": "附录 4 · 末段均值",
    "appendix4_fluct": "附录 4 · 随机波动",
}


def sha256(path: Path) -> str:
    """Read a non-PNG input or SVG file in bounded blocks for provenance."""
    if path.suffix.lower() == ".png":
        raise ValueError("Reading PNG output is forbidden")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    """Read a saved numerical or figure manifest."""
    return json.loads(path.read_text(encoding="utf-8"))


def configure_style() -> str:
    """Use a verified CJK font and the established preceding-question publication palette."""
    installed = {item.name for item in font_manager.fontManager.ttflist}
    candidates = ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC")
    family = next((name for name in candidates if name in installed), None)
    if family is None:
        raise RuntimeError("No supported Chinese font installed")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [family, "DejaVu Sans"],
            "axes.unicode_minus": False,
            "mathtext.fontset": "stix",
            "svg.fonttype": "path",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.edgecolor": "#3D4652",
            "axes.labelcolor": "#20242A",
            "text.color": "#20242A",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.formatter.use_mathtext": True,
        }
    )
    warnings.filterwarnings("error", message="Glyph .* missing from font.*")
    return family


def load_verified(selected: set[str] | None = None) -> tuple:
    """Verify delivery and the numerical files required by the selected plots."""
    requested = set(FIGURE_NAMES) if selected is None else selected
    manifest = read_json(OUT / "delivery_manifest.json")
    if manifest["status"] != "PASS":
        raise RuntimeError("Delivery is not verified")
    solver = ROOT / "src/problem-4/solve_problem4.py"
    if sha256(solver) != manifest["source_sha256"]:
        raise RuntimeError("Solver source changed after verification")
    numerical_hashes = {str(solver): manifest["source_sha256"]}
    for name, digest in manifest["inputs"].items():
        if sha256(ROOT / name) != digest:
            raise RuntimeError(f"Input changed: {name}")
        numerical_hashes[str(ROOT / name)] = digest
    for name, digest in manifest["files"].items():
        if sha256(OUT / name) != digest:
            raise RuntimeError(f"Delivery changed: {name}")
        numerical_hashes[str(OUT / name)] = digest
    if sha256(OUT / "review/verification.json") != manifest["verification_sha256"]:
        raise RuntimeError("Verification changed")
    report = read_json(OUT / "review/verification.json")
    if report["status"] != "PASS":
        raise RuntimeError("Numerical prerequisite checks not passed")
    numerical_hashes[str(OUT / "review/verification.json")] = manifest[
        "verification_sha256"
    ]
    case_paths = [report["main"]]
    scenario_relatives = {case["path"] for case in report["scenarios"]}
    if {"07_scenarios_and_sensitivity", "08_ambient_fluctuation"} & requested:
        case_paths.extend(case["path"] for case in report["scenarios"])
    for relative in case_paths:
        path = OUT / relative
        meta = read_json(path / "metadata.json")
        expected_meta = report["case_metadata_sha256"][relative]
        if sha256(path / "metadata.json") != expected_meta:
            raise RuntimeError("Case metadata altered")
        numerical_hashes[str(path / "metadata.json")] = expected_meta
        for filename, digest in meta["files"].items():
            keep = selected is None
            if selected is not None:
                keep = (
                    relative == report["main"]
                    and ("04_moisture_space_time" in requested or filename == "summary.npz")
                ) or (
                    "08_ambient_fluctuation" in requested
                    and filename == "summary.npz"
                    and relative in scenario_relatives
                )
            if not keep:
                continue
            source = path / filename
            if sha256(source) != digest:
                raise RuntimeError(f"Case data altered: {source}")
            numerical_hashes[str(source)] = digest
    if "06_numerical_convergence" in requested:
        for item in (*report["space"], *report["time"]):
            for key in ("coarse", "fine"):
                relative = item[key]
                path = OUT / relative / "metadata.json"
                digest = report["case_metadata_sha256"][relative]
                if sha256(path) != digest:
                    raise RuntimeError(f"Convergence metadata altered: {relative}")
                numerical_hashes[str(path)] = digest
    main_path = OUT / report["main"]
    main_meta = read_json(main_path / "metadata.json")
    with np.load(main_path / "summary.npz", allow_pickle=False) as data:
        summary = {name: data[name].copy() for name in data.files}
    env = np.loadtxt(OUT / "data/ambient_observed.csv", delimiter=",", skiprows=1)
    radius = np.loadtxt(OUT / "data/radius_observed_SI.csv", delimiter=",", skiprows=1)
    for original, derived, relative, conversion in (
        ("附件1.xlsx", env, "data/ambient_observed.csv", 1.0),
        ("附件2.xlsx", radius, "data/radius_observed_SI.csv", 0.01),
    ):
        book = openpyxl.load_workbook(
            ROOT / "data" / original, read_only=True, data_only=True
        )
        observed = np.asarray(
            list(book["Sheet1"].iter_rows(min_row=2, values_only=True))
        )
        book.close()
        observed[:, 1:] *= conversion
        if not np.array_equal(observed, derived):
            raise RuntimeError(
                f"Plot input copy differs from audited observations: {relative}"
            )
        numerical_hashes[str(OUT / relative)] = sha256(OUT / relative)
    return report, main_meta, summary, env, radius, numerical_hashes


def finish_axes(ax, xlabel: str, ylabel: str, title: str = "") -> None:
    """Apply quiet scientific axes with visible units and light major grids."""
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.grid(True, color="#D8DEE6", alpha=0.7, linewidth=0.55)
    ax.set_axisbelow(True)


def save_figure(fig, name: str, sources: list, scope: str, manifest: list) -> None:
    """Check canvas and text bounds, then export SVG/PNG and a directly rendered JPEG."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    outside = []
    hidden_ticks = set()
    for ax in fig.axes:
        for axis, limits in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            low, high = sorted(limits)
            for tick in (*axis.get_major_ticks(), *axis.get_minor_ticks()):
                if tick.get_loc() < low or tick.get_loc() > high:
                    hidden_ticks.update((id(tick.label1), id(tick.label2)))
    for artist in fig.findobj(match=Text):
        if (
            id(artist) in hidden_ticks
            or not artist.get_visible()
            or not artist.get_text().strip()
        ):
            continue
        box = artist.get_window_extent(renderer)
        if (
            box.width > 0
            and box.height > 0
            and (
                box.x0 < bounds.x0 - 2
                or box.y0 < bounds.y0 - 2
                or box.x1 > bounds.x1 + 2
                or box.y1 > bounds.y1 + 2
            )
        ):
            outside.append(artist.get_text())
    if outside:
        raise RuntimeError(f"Figure text outside canvas: {name}: {outside}")
    pixels = np.asarray(fig.canvas.buffer_rgba())
    fraction = float(np.mean(np.any(pixels[:, :, :3] < 235, axis=2)))
    if fraction < 0.005:
        raise RuntimeError(f"Blank figure: {name}")
    fig.savefig(IMAGES / f"{name}.svg", bbox_inches="tight")
    fig.savefig(IMAGES / f"{name}.png", dpi=320, bbox_inches="tight")
    fig.savefig(
        PREVIEWS / f"{name}.jpg",
        dpi=150,
        bbox_inches="tight",
        pil_kwargs={"quality": 92},
    )
    manifest.append(
        {
            "name": name,
            "files": [f"{name}.svg", f"{name}.png"],
            "svg_sha256": sha256(IMAGES / f"{name}.svg"),
            "preview": str(PREVIEWS / f"{name}.jpg"),
            "sources": sources,
            "scope_and_limit": scope,
            "text_bounds": "PASS",
            "nonblank_fraction": fraction,
            "font_glyphs": "PASS",
            "dpi": 320,
            "plot_code_sha256": sha256(Path(__file__).resolve()),
            "visual_review": "PENDING_JPEG_INSPECTION",
        }
    )
    plt.close(fig)


def plot_inputs(
    env: np.ndarray,
    radius: np.ndarray,
    meta: dict,
    summary: dict,
    manifest: list,
) -> None:
    """Distinguish observed ambient conditions, assumed extension and observed radius."""
    fig, axes = plt.subplots(3, 1, figsize=(8.8, 8.9), layout="constrained")
    end = meta["end_s"] / 3600
    units = ("°C", "kg/kg")
    mode = meta.get("environment_mode", "last")
    grid = summary.get("ambient_grid")
    path = summary.get("ambient_path")
    for j, (label, color) in enumerate(
        (("环境温度 / °C", "#C44E52"), ("环境水分浓度 / (kg/kg)", COLORS[0]))
    ):
        ax = axes[j]
        ax.axvspan(0, 4, color="#8A949C", alpha=0.09)
        ax.plot(
            env[:, 0] / 3600,
            env[:, j + 1],
            color=color,
            lw=1.4,
            label="附件 1 实测插值",
        )
        tail = meta["environment_tail"][j]
        if mode == "fluct" and grid is not None and grid.size:
            ax.plot(
                grid / 3600,
                path[j],
                color=color,
                lw=0.8,
                alpha=0.9,
                label=f"AR(1) 随机波动（种子 {meta.get('environment_seed')}）",
            )
            ax.axhline(tail, color="#50565C", ls="--", lw=1.0, label="末 30 min 均值")
            ax.annotate(
                f"均值 {tail:.6g} {units[j]}",
                (0.52 * end, tail),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.8,
                color="#50565C",
            )
        else:
            ax.plot([4, end], [tail] * 2, "--", color=color, label="4 h 后末值假设")
            ax.annotate(
                f"延拓 {tail:.6g} {units[j]}",
                (0.52 * end, tail),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.8,
                color=color,
            )
        ax.axvline(4, color="#72777C", lw=0.8, ls=":")
        ax.text(
            0.30,
            0.28,
            f"实测 {env[0, j + 1]:.4g} $\\to$ {env[-1, j + 1]:.4g} {units[j]}",
            transform=ax.transAxes,
            va="top",
            fontsize=8.6,
            color="#30363B",
        )
        ax.text(
            0.06,
            0.06,
            "灰色带：实测窗口 0–4 h",
            transform=ax.transAxes,
            fontsize=8.4,
            color="#505960",
        )
        ax.set_xlim(0, end)
        finish_axes(ax, "时间 / h", label)
        ax.legend(loc="upper right", fontsize=8.5)
    r6 = 100 * float(np.interp(6 * 3600, radius[:, 0], radius[:, 1]))
    r72 = 100 * radius[-1, 1]
    rend = 100 * float(np.interp(meta["end_s"], radius[:, 0], radius[:, 1]))
    axes[2].plot(
        radius[:, 0] / 3600,
        100 * radius[:, 1],
        color=COLORS[1],
        lw=1.3,
        marker="o",
        markersize=2.2,
        markevery=4,
        label="附件 2 半径",
    )
    axes[2].axvline(6, color="#8A949C", ls=":", lw=0.8)
    axes[2].axvline(end, color="#50565C", ls="--", lw=1, label="主方案结束时刻")
    axes[2].annotate(
        f"6 h：{r6:.4f} cm",
        (6, r6),
        xytext=(7, 14),
        textcoords="offset points",
        fontsize=8.6,
        color="#505960",
    )
    axes[2].annotate(
        f"72 h：{r72:.4f} cm",
        (72, r72),
        xytext=(-7, 16),
        textcoords="offset points",
        ha="right",
        fontsize=8.6,
        color="#505960",
    )
    axes[2].annotate(
        f"结束 {end:.4f} h\nR={rend:.4f} cm",
        (end, rend),
        xytext=(-8, 10),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=8.6,
        color="#50565C",
    )
    axes[2].text(
        0.42,
        0.86,
        f"附件 1：{len(env)} 条 · 60 s；附件 2：{len(radius)} 条 · 1800 s",
        transform=axes[2].transAxes,
        fontsize=8.4,
        color="#505960",
    )
    axes[2].set_xlim(0, 72)
    finish_axes(axes[2], "时间 / h", "实际半径 / cm")
    axes[2].legend(loc="upper right", fontsize=8.5)
    save_figure(
        fig,
        "01_inputs_and_radius",
        ["data/ambient_observed.csv", "data/radius_observed_SI.csv"],
        "环境在 4 h 后为假设；半径数据仅覆盖 72 h，内部收缩方式不是外半径数据唯一识别的。",
        manifest,
    )


def plot_history(summary: dict, meta: dict, manifest: list) -> None:
    """Show all-node maximum, center, surface and dry-solid-weighted mean moisture."""
    h = summary["history"]
    crit = meta["event"]["critical_s"] / 3600
    endh = meta["end_s"] / 3600
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.7), layout="constrained")
    series = [
        (4, "中心", COLORS[0], "-"),
        (5, "表面", COLORS[3], "-"),
        (8, "干物质加权平均", COLORS[1], "-"),
        (6, "全域最大", "#32383D", ":"),
    ]
    for ax in axes:
        for column, label, color, style in series:
            ax.plot(
                h[:, 0] / 3600,
                h[:, column],
                style,
                color=color,
                linewidth=1.15 if column != 6 else 1.8,
                label=label,
            )
        ax.axhline(0.15, color="#686D72", ls="--", lw=0.9, label="达标阈值 0.15")
        ax.axvline(crit, color=COLORS[4], ls="--", lw=1)
        ax.axvline(endh, color="#343A40", ls=":", lw=0.8)
        finish_axes(ax, "时间 / h", "干基含水率 / (kg/kg)")
    axes[0].set_xlim(0, endh)
    axes[0].legend(loc="upper right", fontsize=8.5)
    axes[0].annotate(
        f"阈值 0.15",
        (0.2, 0.15),
        xytext=(4, 5),
        textcoords="offset points",
        fontsize=8.5,
        color="#565C62",
    )
    axes[1].set_xlim(max(0, endh - 12), endh + 0.35)
    axes[1].set_ylim(0.045, 0.245)
    axes[1].set_title("末期全域达标（放大）")
    axes[1].axvspan(crit, endh, color="#DCE3E8", alpha=0.45, zorder=0)
    axes[1].text(
        0.035,
        0.96,
        f"临界 {crit:.8f} h\n"
        f"严格结束 {endh:.8f} h\n"
        f"裕度 {meta['event']['margin_kg_kg']:.3e} kg/kg",
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=8.7,
        color="#30363B",
    )
    axes[1].scatter([endh], [h[-1, 6]], s=28, color="#32383D", zorder=6)
    axes[1].annotate(
        f"最大/中心 {h[-1, 6]:.4f}",
        (endh, h[-1, 6]),
        xytext=(-8, 8),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=8.4,
        color="#32383D",
    )
    axes[1].scatter([endh], [h[-1, 5]], s=28, color=COLORS[3], zorder=6)
    axes[1].annotate(
        f"表面 {h[-1, 5]:.4f}",
        (endh, h[-1, 5]),
        xytext=(-8, 9),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=8.4,
        color=COLORS[3],
    )
    save_figure(
        fig,
        "02_moisture_history",
        ["main/summary.npz", "main/metadata.json"],
        "中心与最大值可重合；严格判据使用未舍入全部节点；事件夹逼精度不等于累计数值精度。",
        manifest,
    )


def plot_profiles(
    summary: dict, meta: dict, radius: np.ndarray, manifest: list
) -> None:
    """Draw each radial profile only as far as its own actual moving surface."""
    times, states, x = summary["check_times"], summary["check_states"], summary["x"]
    selected = np.flatnonzero(
        (times >= 21600) | np.isclose(times, meta["end_s"], atol=1e-8, rtol=0)
    )
    fig, ax = plt.subplots(figsize=(8.6, 5.4), layout="constrained")
    colors = plt.colormaps["viridis"](np.linspace(0.05, 0.9, len(selected)))
    for color, index in zip(colors, selected):
        t = times[index]
        R = float(np.interp(t, radius[:, 0], radius[:, 1]))
        label = f"{t/3600:g} h" if index != selected[-1] else f"结束 {t/3600:.4f} h"
        ax.plot(100 * R * x, states[index, 1], color=color, lw=1.6, label=label)
        ax.scatter([100 * R], [states[index, 1, -1]], s=16, color=color, zorder=4)
        if index in (selected[0], selected[-1]):
            ax.annotate(
                f"{states[index, 1, -1]:.4f}",
                (100 * R, states[index, 1, -1]),
                xytext=(6, -3),
                textcoords="offset points",
                fontsize=8.4,
                color=color,
            )
    ax.axhline(0.15, color="#565C62", ls="--", lw=0.9)
    ax.annotate(
        "达标阈值 0.15 kg/kg",
        (0.15, 0.15),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=8.5,
        color="#565C62",
        bbox=dict(facecolor="white", alpha=0.82, edgecolor="none", pad=1.6),
    )
    ax.text(
        0.02,
        0.05,
        f"{len(selected)} 个剖面（6 h 间隔加严格结束）；曲线止于各自真实表面；"
        f"结束半径 {100 * float(np.interp(meta['end_s'], radius[:, 0], radius[:, 1])):.4f} cm",
        transform=ax.transAxes,
        fontsize=8.4,
        color="#505960",
        bbox=dict(facecolor="white", alpha=0.82, edgecolor="none", pad=1.8),
    )
    finish_axes(ax, "距药材中心的实际距离 / cm", "干基含水率 / (kg/kg)")
    ax.set_xlim(left=0)
    ax.legend(ncol=2, loc="upper right", fontsize=8.5)
    save_figure(
        fig,
        "03_actual_radius_profiles",
        ["main/summary.npz", "data/radius_observed_SI.csv"],
        "曲线止于各自真实表面；每 6 h 加严格结束时刻；横坐标不是归一化半径。",
        manifest,
    )


def plot_heatmap(
    report: dict, meta: dict, summary: dict, radius: np.ndarray, manifest: list
) -> None:
    """Reconstruct moisture at fixed physical radii and mask the exterior explicitly."""
    case = OUT / report["main"]
    distances = np.linspace(0, 2, 401)
    times, values, boundaries = [], [], []
    for chunk in meta["chunks"]:
        with np.load(case / chunk, allow_pickle=False) as data:
            for t, state in zip(data["times"], data["states"]):
                if 0 < t < 60:
                    continue
                R = 100 * float(np.interp(t, radius[:, 0], radius[:, 1]))
                row = np.full(len(distances), np.nan)
                inside = distances <= R
                row[inside] = np.interp(distances[inside] / R, summary["x"], state[1])
                times.append(t / 3600)
                values.append(row)
                boundaries.append(R)
    values = np.asarray(values)
    if np.isfinite(
        values[np.asarray(distances)[None, :] > np.asarray(boundaries)[:, None]]
    ).any():
        raise AssertionError("Exterior region is not masked")
    fig, ax = plt.subplots(figsize=(9.2, 5.1), layout="constrained")
    ax.set_facecolor("#ECEFF0")
    mesh = ax.pcolormesh(
        times,
        distances,
        np.ma.masked_invalid(values.T),
        shading="nearest",
        cmap="viridis",
        norm=Normalize(vmin=0, vmax=2.55),
        rasterized=True,
    )
    ax.plot(times, boundaries, color="#C44E52", lw=1.6, label="实测收缩边界")
    contour = ax.contour(
        times,
        distances,
        np.ma.masked_invalid(values.T),
        levels=[0.15],
        colors=["#FFFFFF"],
        linestyles="--",
        linewidths=1.6,
        corner_mask=False,
        zorder=4,
    )
    outline = [
        patheffects.Stroke(linewidth=2.9, foreground="#29343D"),
        patheffects.Normal(),
    ]
    contour.set_path_effects(outline)
    segments = [segment for segment in contour.allsegs[0] if len(segment) > 1]
    if not segments:
        raise RuntimeError("No 0.15 kg/kg contour in the saved moisture field")
    for segment in segments:
        outer_radius = np.interp(segment[:, 0], times, boundaries)
        if not np.isfinite(segment).all() or np.any(
            segment[:, 1] > outer_radius + 1e-9
        ):
            raise AssertionError("Threshold contour extends outside the material")
    fig.colorbar(mesh, ax=ax, label="干基含水率 / (kg/kg)", pad=0.025)
    ax.set(xlim=(0, meta["end_s"] / 3600), ylim=(0, 2.0))
    finish_axes(ax, "时间 / h", "距药材中心的实际距离 / cm")
    crit = meta["event"]["critical_s"] / 3600
    ax.axvline(crit, color="white", ls=":", lw=1.3, zorder=5, path_effects=outline)
    ax.annotate(
        f"临界 {crit:.4f} h",
        (crit, 1.52),
        xytext=(-5, 0),
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=8.8,
        color="white",
        path_effects=outline,
        zorder=6,
    )
    ax.text(
        0.02,
        0.06,
        f"{len(times)} 个保存时刻；灰色为域外；白色虚线为 0.15 kg/kg 等值线",
        transform=ax.transAxes,
        fontsize=8.4,
        color="#F2F5F7",
        bbox=dict(facecolor="#29343D", alpha=0.55, edgecolor="none", pad=2.5),
    )
    boundary_handle = ax.lines[0]
    threshold_handle = Line2D(
        [], [], color="white", ls="--", lw=1.6, path_effects=outline,
        label="含水率 0.15 kg/kg 等值线",
    )
    critical_handle = Line2D(
        [], [], color="white", ls=":", lw=1.3, path_effects=outline,
        label=f"临界 {crit:.4f} h",
    )
    ax.legend(
        handles=[boundary_handle, threshold_handle, critical_handle],
        loc="upper right",
        fontsize=8.5,
    )
    save_figure(
        fig,
        "04_moisture_space_time",
        [report["main"] + "/fields_*.npz", "data/radius_observed_SI.csv"],
        "一维径向时空图；灰色为药材外部。白色虚线为 0.15 kg/kg 等值线，"
        "按已保存场插值展示，不替代未舍入全场事件定位；未求解二维轴向场。",
        manifest,
    )
    manifest[-1]["threshold_contour"] = {
        "level_kg_kg": 0.15,
        "segments": len(segments),
        "vertices": sum(len(segment) for segment in segments),
        "outside_material": False,
    }


def plot_temperature(summary: dict, meta: dict, manifest: list) -> None:
    """Show local versus ambient temperature and diffusion dependence on saved local states."""
    h = summary["history"]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6), layout="constrained")
    for column, label, color, style in [
        (2, "药材中心", COLORS[0], "-"),
        (3, "药材表面", COLORS[3], "-"),
        (17, "对应烘房环境", "#444A50", "--"),
    ]:
        axes[0].plot(
            h[:, 0] / 3600, h[:, column], style, color=color, lw=1.3, label=label
        )
    axes[0].set_xlim(0, 6)
    axes[0].text(
        0.97,
        0.06,
        f"环境平台 {h[-1, 17]:.4f} °C\n表面峰值 {h[:, 3].max():.3f} °C",
        transform=axes[0].transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="#505960",
    )
    finish_axes(axes[0], "时间 / h", "温度 / °C", "初期热传递（0–6 h）")
    axes[0].legend(fontsize=8.5)
    diffusion = {}
    for T_column, C_column, label, color, marker in [
        (2, 4, "中心局部扩散系数", COLORS[0], "o"),
        (3, 5, "表面局部扩散系数", COLORS[3], "s"),
    ]:
        D = (
            4.2e-4
            * np.exp(-0.30 / h[:, C_column])
            * np.exp(-3850 / (h[:, T_column] + 273.15))
        )
        diffusion[label] = D
        axes[1].semilogy(h[:, 0] / 3600, D, color=color, lw=1.3, label=label)
        axes[1].scatter(
            [h[0, 0] / 3600, h[-1, 0] / 3600],
            [D[0], D[-1]],
            s=22,
            color=color,
            marker=marker,
            zorder=5,
        )
        axes[1].annotate(
            f"{D[0]:.2e}",
            (h[0, 0] / 3600, D[0]),
            xytext=(5, 7),
            textcoords="offset points",
            fontsize=8.2,
            color=color,
        )
        axes[1].annotate(
            f"{D[-1]:.2e}",
            (h[-1, 0] / 3600, D[-1]),
            xytext=(-5, 7),
            textcoords="offset points",
            ha="right",
            fontsize=8.2,
            color=color,
        )
    axes[1].set_xlim(0, meta["end_s"] / 3600)
    center_d = diffusion["中心局部扩散系数"]
    surface_d = diffusion["表面局部扩散系数"]
    axes[1].text(
        0.03,
        0.06,
        f"末/首扩散系数：中心 {center_d[-1] / center_d[0]:.4f}；"
        f"表面 {surface_d[-1] / surface_d[0]:.4f}",
        transform=axes[1].transAxes,
        fontsize=8.5,
        color="#505960",
    )
    finish_axes(axes[1], "时间 / h", "局部扩散系数 / (m²/s)", "局部温湿状态决定扩散")
    axes[1].legend(fontsize=8.5, loc="upper right")
    save_figure(
        fig,
        "05_temperature_and_diffusion",
        ["main/summary.npz"],
        "扩散系数按保存的局部温度与干基含水率重构，绝对温度进入指数；未重新积分。",
        manifest,
    )


def convergence_levels(items: list, report: dict) -> tuple:
    """Read each actual refinement level and validate its adjacent event differences."""
    if not items:
        raise ValueError("No executed refinement comparisons")
    relatives = [items[0]["coarse"]]
    for item in items:
        if item["coarse"] != relatives[-1]:
            raise ValueError("Refinement comparisons do not form a contiguous sequence")
        relatives.append(item["fine"])
    metadata = []
    for relative in relatives:
        path = OUT / relative / "metadata.json"
        if sha256(path) != report["case_metadata_sha256"][relative]:
            raise RuntimeError(f"Convergence metadata altered: {relative}")
        value = read_json(path)
        if value["event"] is None:
            raise ValueError("Refinement level has no recorded threshold crossing")
        metadata.append(value)
    critical = np.array([value["event"]["critical_s"] for value in metadata])
    if not np.isfinite(critical).all():
        raise ValueError("Nonfinite convergence event times")
    if not np.allclose(
        np.diff(critical), [item["event_delta_s"] for item in items],
        rtol=0, atol=1e-8,
    ):
        raise ValueError("Event metadata and recorded refinement differences disagree")
    return relatives, metadata, critical


def plot_convergence(report: dict, meta: dict, manifest: list) -> None:
    """Show executed levels, chosen resolution and adjacent acceptance as line charts."""
    fig = plt.figure(figsize=(11.8, 10.0), layout="constrained")
    grid = fig.add_gridspec(4, 2, height_ratios=(2.45, 2.45, 1.75, 0.62))
    table_rows, evidence, sources = [], {}, ["review/verification.json"]
    orders = []
    targets = report["targets"]["thresholds"]
    for column, (key, label, color) in enumerate(
        (("space", "空间", COLORS[0]), ("time", "时间", COLORS[1]))
    ):
        items = report[key]
        relatives, metadata, critical = convergence_levels(items, report)
        configs = [value["identity"]["config"] for value in metadata]
        variable = "n" if key == "space" else "factor"
        fixed = "factor" if key == "space" else "n"
        if len({config[fixed] for config in configs}) != 1:
            raise ValueError("Refinement plot mixes spatial and temporal changes")
        levels = [config[variable] for config in configs]
        chosen = meta["identity"]["config"][variable]
        chosen_index = levels.index(chosen)
        offsets_ms = 1000 * np.abs(critical - critical[-1])
        xpos = np.arange(len(levels))
        tick_labels = [f"{level:g}" for level in levels]
        chosen_label = "主方案网格" if key == "space" else "主方案时间倍率"

        ax = fig.add_subplot(grid[0, column])
        ax.plot(
            xpos, offsets_ms, "-o", color=color, lw=1.9, ms=6.5, zorder=4,
            label="相邻级时长差 $|\\Delta t_*|$",
        )
        ax.scatter(
            [chosen_index], [offsets_ms[chosen_index]], s=155,
            facecolors="none", edgecolors="#B4433D", linewidths=1.8,
            zorder=5, label=chosen_label,
        )
        for x, value in zip(xpos, offsets_ms):
            ax.annotate(
                f"{value:.3f}", (x, value), xytext=(0, -11),
                textcoords="offset points", ha="center", va="top", fontsize=9,
                bbox=dict(facecolor="white", alpha=0.72, edgecolor="none", pad=1.2),
            )
        span = max(float(offsets_ms.max()), 1e-3)
        ax.set_ylim(-0.12 * span, 1.35 * span)
        ax.set_xlim(-0.45, len(levels) - 0.55)
        ax.set_xticks(xpos, tick_labels)
        ax.axhline(0, color="#7C858C", lw=0.7)
        finish_axes(
            ax,
            "径向区间数 N（由粗到细）" if key == "space" else "时间步倍率（由粗到细）",
            "相对最细解的时长差 / ms",
        )
        ax.grid(False, axis="x")
        setting = (
            f"时间倍率 {configs[0]['factor']:g}"
            if key == "space" else f"N={configs[0]['n']}"
        )
        ax.set_title(
            f"({'ab'[column]}) {label}加密时长差（折线）· {setting}", loc="left", pad=10
        )
        ax.text(
            0.025, 0.95, f"参考时长 {critical[-1] / 3600:.4f} h",
            transform=ax.transAxes, va="top", fontsize=9, color="#505960",
        )
        ax.legend(loc="upper right", frameon=False, fontsize=8.5)

        differences = [item["metrics"]["full"]["C"] for item in items]
        order = (
            float(np.log2(differences[-2] / differences[-1]))
            if len(items) >= 2 else None
        )
        orders.append(f"{label} {order:.2f}" if order is not None else f"{label}未估计")

        ax2 = fig.add_subplot(grid[1, column])
        xc = np.arange(len(items))
        field_c = [item["metrics"]["full"]["C"] for item in items]
        field_t = [item["metrics"]["full"]["T"] for item in items]
        ax2.semilogy(
            xc, field_c, "-o", color=COLORS[0], lw=1.9, ms=6.5, label="全场 $\\Delta C$"
        )
        ax2.semilogy(
            xc, field_t, "-s", color=COLORS[3], lw=1.9, ms=6.5, label="全场 $\\Delta T$"
        )
        for x, value in zip(xc, field_c):
            ax2.annotate(
                f"{value:.2e}", (x, value), xytext=(0, -14),
                textcoords="offset points", ha="center", va="top",
                fontsize=8.3, color=COLORS[0],
                bbox=dict(facecolor="white", alpha=0.72, edgecolor="none", pad=1.2),
            )
        for x, value in zip(xc, field_t):
            ax2.annotate(
                f"{value:.2e}", (x, value), xytext=(0, -14),
                textcoords="offset points", ha="center", va="top",
                fontsize=8.3, color=COLORS[3],
                bbox=dict(facecolor="white", alpha=0.72, edgecolor="none", pad=1.2),
            )
        all_values = field_c + field_t
        ax2.set_ylim(min(all_values) * 0.15, max(all_values) * 2.1)
        ax2.text(
            0.03,
            0.96,
            f"目标：$\\Delta C$ ≤ {targets['field_C_kg_kg']:g}，"
            f"$\\Delta T$ ≤ {targets['field_T_C']:g}（均高于显示范围）",
            transform=ax2.transAxes,
            va="top",
            fontsize=8.2,
            color="#505960",
        )
        pair_labels = [
            f"{pair[0]:g} $\\to$ {pair[1]:g}"
            for pair in (item["N" if key == "space" else "factors"] for item in items)
        ]
        ax2.set_xticks(xc, pair_labels)
        ax2.set_xlim(-0.4, len(items) - 0.6)
        ax2.grid(False, axis="x")
        finish_axes(ax2, "相邻级别", "全场最大差 / (kg/kg)、°C")
        order_text = (
            f"含水率观察阶 {order:.2f}" if order is not None else "含水率观察阶未估计"
        )
        ax2.set_title(
            f"({'cd'[column]}) {label}加密全场差（对数折线）· {order_text}",
            loc="left", pad=10,
        )
        ax2.legend(loc="upper right", fontsize=8.2, ncol=2)

        for item in items:
            pair = item["N" if key == "space" else "factors"]
            table_rows.append(
                [
                    label, f"{pair[0]:g} → {pair[1]:g}",
                    f"{item['metrics']['full']['C']:.3e}",
                    f"{item['metrics']['full']['T']:.3e}",
                    f"{abs(item['event_delta_s']):.6f}",
                    "通过" if item["status"] == "PASS" else "未通过",
                ]
            )
        sources.extend(relative + "/metadata.json" for relative in relatives)
        evidence[key] = {
            "levels": levels, "critical_s": critical.tolist(),
            "reference_case": relatives[-1], "absolute_offset_ms": offsets_ms.tolist(),
            "chosen_level": chosen, "field_C_observed_order": order,
        }

    table_ax = fig.add_subplot(grid[2, :])
    table_ax.axis("off")
    table_ax.set_title("(e) 相邻加密级的全场差与验收", loc="left", fontsize=11, pad=8)
    table = table_ax.table(
        cellText=table_rows,
        colLabels=["方向", "相邻级别", "全场 ΔC / (kg/kg)", "全场 ΔT / °C", "|Δt*| / s", "验收"],
        colWidths=[0.09, 0.25, 0.20, 0.18, 0.16, 0.12],
        cellLoc="center", bbox=[0, 0, 1, 0.94],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#D5DCE0")
        cell.set_linewidth(0.45)
        if row == 0:
            cell.set_facecolor("#EAF0F3")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("#F8FAFB" if row % 2 else "white")
            if table_rows[row - 1][-1] == "未通过" and col in (2, 5):
                cell.set_text_props(color="#B4433D", weight="bold")
    note_ax = fig.add_subplot(grid[3, :])
    note_ax.axis("off")
    note_ax.text(
        0, 0.95,
        f"验收：ΔC ≤ {targets['field_C_kg_kg']:g} kg/kg；"
        f"ΔT ≤ {targets['field_T_C']:g} °C；|Δt*| ≤ {targets['event_change_s']:g} s。"
        f"随机波动情景 ΔT ≤ {targets['fluct_field_T_C']:g} °C。"
        f"含水率观察阶：{'，'.join(orders)}。",
        transform=note_ax.transAxes, va="top", fontsize=8.7, color="#505960",
    )
    note_ax.text(
        0, 0.22,
        "上排为时长差折线，中排为全场差对数折线；最细级作为参考，自差为 0。"
        "上述差值均不是连续解误差界或工艺精度保证。",
        transform=note_ax.transAxes, va="top", fontsize=8.7, color="#505960",
    )
    save_figure(
        fig, "06_numerical_convergence", list(dict.fromkeys(sources)),
        "两组均展示三级实际计算结果，相对本组最细解的时长差按存档事件直接计算。"
        "红圈标记主方案选用的网格或时间倍率；表格保留粗级未通过及全部相邻场差。"
        "参考解自差为零不代表真实误差为零，观察阶不是严格精度证明。",
        manifest,
    )
    manifest[-1]["convergence_levels"] = evidence


def plot_scenarios(report: dict, meta: dict, manifest: list) -> None:
    """Compare validated geometry/property scenarios and the declared ambient sensitivity."""
    cases = {}
    for case in report["scenarios"]:
        if case["status"] != "PASS":
            raise RuntimeError("Unverified scenario cannot be plotted as accepted")
        cases[case["name"]] = read_json(OUT / case["path"] / "metadata.json")
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), layout="constrained")
    main_hours = meta["event"]["critical_s"] / 3600
    names = ["appendix3_fixed", "appendix3_shrink", "appendix4_fixed", "main"]
    labels = [LABELS.get(name, "附录 4 · 实测收缩") for name in names]
    values = [cases[name] if name != "main" else meta for name in names]
    for j, value in enumerate(values):
        dry = value["event"] is not None
        hours = value["event"]["critical_s"] / 3600 if dry else value["end_s"] / 3600
        axes[0].barh(
            j,
            hours,
            color=COLORS[j],
            height=0.56,
            hatch=None if dry else "///",
            alpha=0.85,
            edgecolor="#3D4652",
            linewidth=0.6,
        )
        label = f"{hours:.4f} h" if dry else f"{hours:g} h 内未达标"
        axes[0].text(hours + 0.8, j, label, va="center", fontsize=8.8)
    axes[0].axvline(main_hours, color=COLORS[0], ls=":", lw=1.1)
    axes[0].set_yticks(range(4), labels, fontsize=9)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 92)
    finish_axes(
        axes[0],
        "临界时长或数据覆盖 / h",
        "",
        "相同初值与数值精度下的物性与几何\n点线：主方案临界时长；斜纹：覆盖内未达标",
    )
    sensitivity_series = (
        (meta, "随机波动 AR(1)（种子 0，主方案）", "随机波动", COLORS[1]),
        (cases["appendix4_last"], "末次实测值延拓", "末值", COLORS[0]),
        (cases["appendix4_mean30"], "末 30 min 算术均值延拓", "末段均值", COLORS[4]),
    )
    for j, (value, label, short, color) in enumerate(sensitivity_series):
        dry = value["event"] is not None
        hours = value["event"]["critical_s"] / 3600 if dry else 72
        axes[1].barh(
            j,
            hours,
            color=color,
            height=0.45,
            alpha=0.85,
            edgecolor="#3D4652",
            linewidth=0.6,
        )
        if dry:
            text = f"{hours:.4f} h（{(hours - main_hours) * 60:+.2f} min）"
        else:
            text = "72 h 内未达标"
        axes[1].text(hours + 0.4, j, text, va="center", fontsize=8.8)
    axes[1].axvline(main_hours, color=COLORS[0], ls=":", lw=1.1)
    axes[1].set_yticks(
        range(len(sensitivity_series)), [s[2] for s in sensitivity_series], fontsize=9
    )
    axes[1].invert_yaxis()
    axes[1].set_xlim(
        0,
        max(
            (72 if value["event"] is None else value["event"]["critical_s"] / 3600)
            for value, _, _, _ in sensitivity_series
        )
        * 1.38,
    )
    sensitivity_title = "附录 4 实测收缩的环境敏感性"
    sensitivity = cases["appendix4_mean30"]
    if sensitivity["event"] is not None and meta["event"] is not None:
        last_min = (
            cases["appendix4_last"]["event"]["critical_s"] - meta["event"]["critical_s"]
        ) / 60
        mean_min = (
            sensitivity["event"]["critical_s"] - meta["event"]["critical_s"]
        ) / 60
        sensitivity_title += (
            f"\n主方案随机波动；末值变化 {last_min:+.2f} min；均值变化 {mean_min:+.2f} min"
        )
    stats = meta["fluctuation_stats"]
    phi = meta["environment_ar1_phi"]
    sensitivity_title += (
        f"\n波动参数：$\\sigma_T$={stats['T']['sigma']:.3g} °C，"
        f"$\\sigma_C$={stats['C']['sigma']:.3g} kg/kg，"
        f"$\\varphi_T$={phi['T']:.3f}，$\\varphi_C$={phi['C']:.3f}"
    )
    finish_axes(axes[1], "临界时长 / h", "", sensitivity_title)
    save_figure(
        fig,
        "07_scenarios_and_sensitivity",
        ["review/verification.json", "cases/*/metadata.json"],
        "4 h 后确定性延拓与随机波动情景并列；未达标情景仅报告覆盖内状态，不推断无限烘干时长；效应不普遍可加。",
        manifest,
    )


def plot_ambient_fluctuation(
    report: dict, meta: dict, env: np.ndarray, summary: dict, manifest: list
) -> None:
    """Compare the realized AR(1) ambient tail with the deterministic extensions."""
    scenarios = {case["name"]: case for case in report["scenarios"]}
    if "appendix4_last" not in scenarios or "appendix4_mean30" not in scenarios:
        raise RuntimeError("Ambient sensitivity scenarios missing from verification")
    last_case = OUT / scenarios["appendix4_last"]["path"]
    mean_case = OUT / scenarios["appendix4_mean30"]["path"]
    last_meta = read_json(last_case / "metadata.json")
    mean_meta = read_json(mean_case / "metadata.json")
    with np.load(last_case / "summary.npz", allow_pickle=False) as data:
        last_history = data["history"]
    with np.load(mean_case / "summary.npz", allow_pickle=False) as data:
        mean_history = data["history"]
    main_history = summary["history"]
    grid = summary["ambient_grid"]
    path = summary["ambient_path"]
    if path.shape != (2, grid.size):
        raise ValueError("Saved fluctuation path shape mismatch")
    stats = meta["fluctuation_stats"]
    phi = meta["environment_ar1_phi"]
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.9), layout="constrained")
    end = 72.0
    for j, (label, color, unit, key) in enumerate(
        (
            ("环境温度 / °C", "#C44E52", "°C", "T"),
            ("环境水分浓度 / (kg/kg)", COLORS[0], "kg/kg", "C"),
        )
    ):
        ax = axes[0, j]
        mean_value = meta["environment_tail"][j]
        sigma = stats[key]["sigma"] * meta["environment_sigma_scale"]
        ax.axvspan(0, 4, color="#8A949C", alpha=0.09)
        ax.fill_between(
            grid / 3600,
            mean_value - 2 * sigma,
            mean_value + 2 * sigma,
            color=color,
            alpha=0.14,
            label="均值 $\\pm 2\\sigma$",
        )
        ax.plot(
            env[:, 0] / 3600, env[:, j + 1], color=color, lw=1.5, label="附件 1 实测"
        )
        ax.plot(
            grid / 3600,
            path[j],
            color=color,
            lw=0.9,
            alpha=0.9,
            label=f"随机波动 AR(1)（种子 {meta.get('environment_seed')}，主方案）",
        )
        ax.axhline(
            last_meta["environment_tail"][j],
            color="#50565C",
            ls="--",
            lw=1.0,
            label="末次实测值延拓",
        )
        ax.axhline(
            mean_meta["environment_tail"][j],
            color=COLORS[4],
            ls=":",
            lw=1.1,
            label="末 30 min 均值",
        )
        ax.axvline(4, color="#72777C", lw=0.8, ls=":")
        ax.set_xlim(0, end)
        span_values = np.r_[
            env[:, j + 1],
            path[j],
            last_meta["environment_tail"][j],
            mean_meta["environment_tail"][j],
        ]
        span = float(span_values.max() - span_values.min())
        ax.set_ylim(span_values.min() - 0.08 * span, span_values.max() + 0.20 * span)
        finish_axes(ax, "时间 / h", label)
        ax.legend(loc="lower right", fontsize=7.8)
        ax.text(
            0.025,
            0.96,
            f"均值 {mean_value:.6g} {unit}\n$\\sigma$={sigma:.3g}，$\\varphi$={phi[key]:.4f}",
            transform=ax.transAxes,
            va="top",
            fontsize=8.6,
            color="#505960",
        )
    configs = [value["identity"]["config"] for value in (meta, last_meta, mean_meta)]
    grid_values = {(config["n"], config["factor"]) for config in configs}
    if len(grid_values) == 1:
        setup = (
            "主方案与两对照均为同网格、同时间步\n"
            f"（N={configs[0]['n']}，时间倍率 {configs[0]['factor']:g}）\n"
            "差异仅来自 4 h 后环境延拓"
        )
    else:
        setup = "\n".join(
            f"{name}：N={config['n']}，倍率 {config['factor']:g}"
            for name, config in zip(("波动", "末值", "均值"), configs)
        )
    series = (
        (
            f"随机波动（主方案，N={configs[0]['n']}，f={configs[0]['factor']:g}）",
            main_history,
            COLORS[1],
        ),
        (
            f"末值延拓（N={configs[1]['n']}，f={configs[1]['factor']:g}）",
            last_history,
            COLORS[0],
        ),
        (
            f"末段均值（N={configs[2]['n']}，f={configs[2]['factor']:g}）",
            mean_history,
            COLORS[4],
        ),
    )
    ax = axes[1, 0]
    for name, history, color in series:
        ax.plot(history[:, 0] / 3600, history[:, 6], "-", color=color, lw=1.4, label=name)
    ax.axhline(0.15, color="#686D72", ls="--", lw=0.9, label="阈值 0.15")
    for value, color in (
        (meta, COLORS[1]),
        (last_meta, COLORS[0]),
        (mean_meta, COLORS[4]),
    ):
        ax.axvline(value["event"]["critical_s"] / 3600, color=color, ls=":", lw=1.0)
    ax.set_xlim(0, max(v["end_s"] for v in (meta, last_meta, mean_meta)) / 3600)
    ax.text(
        0.20,
        0.50,
        setup,
        transform=ax.transAxes,
        fontsize=8.2,
        color="#30363B",
        va="center",
        bbox=dict(facecolor="white", alpha=0.80, edgecolor="#C9D1D8", pad=3),
    )
    finish_axes(ax, "时间 / h", "全域最大含水率 / (kg/kg)", "三种环境延拓的全域最大值")
    ax.legend(loc="upper right", fontsize=8.0)
    ax.text(
        0.03,
        0.06,
        "三条曲线在前 45 h 几乎重合；差异集中在事件前约 2 h",
        transform=ax.transAxes,
        fontsize=8.4,
        color="#505960",
    )
    ax = axes[1, 1]
    focus = meta["event"]["critical_s"] / 3600
    for name, history, color in series:
        ax.plot(history[:, 0] / 3600, history[:, 6], "-", color=color, lw=1.6, label=name)
    ax.axhline(0.15, color="#686D72", ls="--", lw=0.9)
    ax.set_xlim(focus - 1.6, focus + 1.2)
    ax.set_ylim(0.148, 0.158)
    for value, color in (
        (meta, COLORS[1]),
        (last_meta, COLORS[0]),
        (mean_meta, COLORS[4]),
    ):
        ax.axvline(value["event"]["critical_s"] / 3600, color=color, ls=":", lw=1.0)
        ax.plot([value["event"]["critical_s"] / 3600], [0.15], "o", color=color, ms=6, zorder=6)
    ax.text(
        0.035,
        0.96,
        "\n".join(
            f"{name}：{value['event']['critical_s'] / 3600:.4f} h"
            for name, value in (
                ("波动", meta),
                ("末值", last_meta),
                ("均值", mean_meta),
            )
        ),
        transform=ax.transAxes,
        va="top",
        fontsize=8.4,
        color="#30363B",
        bbox=dict(facecolor="white", alpha=0.78, edgecolor="#C9D1D8", pad=3),
    )
    finish_axes(
        ax,
        "时间 / h",
        "全域最大含水率 / (kg/kg)",
        "临界时刻邻域放大（与正式求解同 N、同时间步）",
    )
    ax.legend(loc="lower left", fontsize=8.0)
    save_figure(
        fig,
        "08_ambient_fluctuation",
        [
            "main/summary.npz",
            "main/metadata.json",
            scenarios["appendix4_last"]["path"] + "/summary.npz",
            scenarios["appendix4_last"]["path"] + "/metadata.json",
            scenarios["appendix4_mean30"]["path"] + "/summary.npz",
            scenarios["appendix4_mean30"]["path"] + "/metadata.json",
            "data/ambient_observed.csv",
        ],
        "波动由观测末段去趋势残差估计标准差与滞后一阶自相关，种子固定；"
        "随机情景热场时间加密阈值放宽到 1e-3 °C，湿场仍 5e-5 kg/kg。",
        manifest,
    )


def workbook_previews(manifest: list) -> None:
    """Render saved workbook samples with its actual number formats for visual inspection."""
    book = openpyxl.load_workbook(OUT / "result4.xlsx", read_only=False, data_only=True)
    sheet = book["Sheet1"]
    row_ids = [1, *range(2, 8), 361, sheet.max_row - 1, sheet.max_row]
    row_ids = sorted(set(row for row in row_ids if row <= sheet.max_row))
    for number, columns in enumerate(
        (
            list(range(1, min(12, sheet.max_column + 1))),
            [1, *range(12, sheet.max_column + 1)],
        ),
        1,
    ):
        text = []
        for row in row_ids:
            row_text = []
            for col in columns:
                cell = sheet.cell(row, col)
                value = cell.value
                if value is None:
                    formatted = ""
                elif row == 1:
                    formatted = str(value).replace(
                        "时间\\到药材中心的距离", "时间/s\\距离/cm"
                    )
                elif col == 1:
                    formatted = f"{value:.8f}".rstrip("0").rstrip(".")
                else:
                    formatted = f"{value:.4f}"
                row_text.append(formatted)
            text.append(row_text)
        fig, ax = plt.subplots(figsize=(12, 3.9), layout="constrained")
        ax.axis("off")
        widths = [0.19] + [(1 - 0.19) / (len(columns) - 1)] * (len(columns) - 1)
        table = ax.table(cellText=text, cellLoc="right", colWidths=widths, loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.7)
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("#DBE0E3")
            cell.set_linewidth(0.35)
            if row == 0:
                cell.set_facecolor("#E8EDF0")
        ax.set_title(
            f"result4.xlsx · 选定行与第 {number} 组实际距离列", loc="left", fontsize=11
        )
        fig.canvas.draw()
        fig.savefig(
            PREVIEWS / f"workbook_part{number}.jpg",
            dpi=150,
            bbox_inches="tight",
            pil_kwargs={"quality": 92},
        )
        plt.close(fig)
    book.close()
    manifest.append(
        {
            "name": "workbook_preview",
            "sources": ["result4.xlsx"],
            "preview": [
                str(PREVIEWS / "workbook_part1.jpg"),
                str(PREVIEWS / "workbook_part2.jpg"),
            ],
            "scope_and_limit": "保存后工作簿选定行的数值与四位小数格式重构预览；数值逐格校验见 delivery_checks.json。",
        }
    )


def main() -> None:
    """Generate all or selected figures while preserving the numerical delivery."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-verified",
        action="store_true",
        help="Require verified data (also enforced by default)",
    )
    parser.add_argument(
        "--figures", nargs="+", choices=FIGURE_NAMES,
        help="Regenerate selected groups and preserve other verified figure entries",
    )
    args = parser.parse_args()
    selected = set(args.figures) if args.figures else None
    font = configure_style()
    report, meta, summary, env, radius, before = load_verified(selected)
    manifest_path = IMAGES / "figure_manifest.json"
    retained = []
    if selected is not None and manifest_path.exists():
        previous = read_json(manifest_path)
        if (
            previous["solve_code_sha256"] != meta["identity"]["solve_code_sha256"]
            or previous["verification_sha256"]
            != sha256(OUT / "review/verification.json")
        ):
            raise RuntimeError("Existing figure batch is stale; regenerate all figures")
        for entry in previous["figures"]:
            if entry["name"] in selected:
                continue
            if "svg_sha256" in entry:
                if sha256(IMAGES / f"{entry['name']}.svg") != entry["svg_sha256"]:
                    raise RuntimeError(f"Retained figure altered: {entry['name']}")
            entry.setdefault("plot_code_sha256", previous["plot_code_sha256"])
            entry.setdefault("visual_review", previous["visual_review"])
            retained.append(entry)
    IMAGES.mkdir(parents=True, exist_ok=True)
    PREVIEWS.mkdir(parents=True, exist_ok=True)
    figures = []
    jobs = (
        (FIGURE_NAMES[0], plot_inputs, (env, radius, meta, summary)),
        (FIGURE_NAMES[1], plot_history, (summary, meta)),
        (FIGURE_NAMES[2], plot_profiles, (summary, meta, radius)),
        (FIGURE_NAMES[3], plot_heatmap, (report, meta, summary, radius)),
        (FIGURE_NAMES[4], plot_temperature, (summary, meta)),
        (FIGURE_NAMES[5], plot_convergence, (report, meta)),
        (FIGURE_NAMES[6], plot_scenarios, (report, meta)),
        (FIGURE_NAMES[7], plot_ambient_fluctuation, (report, meta, env, summary)),
    )
    for name, function, arguments in jobs:
        if selected is None or name in selected:
            function(*arguments, figures)
    regenerated = [entry["name"] for entry in figures]
    if selected is None:
        workbook_previews(figures)
    figures = sorted([*retained, *figures], key=lambda entry: entry["name"])
    for path, digest in before.items():
        if sha256(Path(path)) != digest:
            raise RuntimeError("Numerical data changed during plotting")
    manifest = {
        "status": "PASS",
        "automated_visual_checks": "PASS",
        "visual_review": "PENDING_JPEG_INSPECTION",
        "font": font,
        "plot_dependencies": {
            "matplotlib": matplotlib.__version__,
            "numpy": np.__version__,
            "openpyxl": openpyxl.__version__,
        },
        "plot_code_sha256": sha256(Path(__file__).resolve()),
        "plot_code_scope": "Latest invocation; each figure retains its generating source",
        "regenerated_figures": regenerated,
        "solve_code_sha256": meta["identity"]["solve_code_sha256"],
        "verification_sha256": sha256(OUT / "review/verification.json"),
        "main_case": report["main"],
        "numerical_preservation": "PASS",
        "validated_source_files": len(before),
        "PNG_outputs_read": False,
        "figures": figures,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "scientific_figure_groups": sum(
                    "svg_sha256" in entry for entry in figures
                ),
                "regenerated_figures": regenerated,
                "font": font,
                "preview_directory": str(PREVIEWS),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
