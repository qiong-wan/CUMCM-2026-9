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
)
LABELS = {
    "appendix3_fixed": "附录 3 · 固定半径",
    "appendix3_shrink": "附录 3 · 实测收缩",
    "appendix4_fixed": "附录 4 · 固定半径",
    "appendix4_mean30": "附录 4 · 末段均值",
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
    if "07_scenarios_and_sensitivity" in requested:
        case_paths.extend(case["path"] for case in report["scenarios"])
    for relative in case_paths:
        path = OUT / relative
        meta = read_json(path / "metadata.json")
        expected_meta = report["case_metadata_sha256"][relative]
        if sha256(path / "metadata.json") != expected_meta:
            raise RuntimeError("Case metadata altered")
        numerical_hashes[str(path / "metadata.json")] = expected_meta
        for filename, digest in meta["files"].items():
            if selected is not None and not (
                relative == report["main"]
                and ("04_moisture_space_time" in requested or filename == "summary.npz")
            ):
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
    env: np.ndarray, radius: np.ndarray, meta: dict, manifest: list
) -> None:
    """Distinguish observed ambient conditions, assumed extension and observed radius."""
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 8.3), layout="constrained")
    end = meta["end_s"] / 3600
    for j, (label, color) in enumerate(
        (("环境温度 / °C", "#C44E52"), ("环境水分浓度 / (kg/kg)", COLORS[0]))
    ):
        ax = axes[j]
        ax.plot(
            env[:, 0] / 3600,
            env[:, j + 1],
            color=color,
            lw=1.3,
            label="附件 1 实测插值",
        )
        ax.plot(
            [4, end],
            [meta["environment_tail"][j]] * 2,
            "--",
            color=color,
            label="4 h 后末值假设",
        )
        ax.axvline(4, color="#72777C", lw=0.8, ls=":")
        ax.set_xlim(0, end)
        finish_axes(ax, "时间 / h", label)
        ax.legend(loc="lower right")
    axes[2].plot(
        radius[:, 0] / 3600,
        100 * radius[:, 1],
        color=COLORS[1],
        lw=1.2,
        marker="o",
        markersize=2,
        markevery=4,
        label="附件 2 半径",
    )
    axes[2].axvline(end, color="#50565C", ls="--", lw=1, label="主方案结束时刻")
    axes[2].set_xlim(0, 72)
    finish_axes(axes[2], "时间 / h", "实际半径 / cm")
    axes[2].legend()
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
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), layout="constrained")
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
        ax.axvline(meta["event"]["critical_s"] / 3600, color=COLORS[4], ls="--", lw=1)
        ax.axvline(meta["end_s"] / 3600, color="#343A40", ls=":", lw=0.8)
        finish_axes(ax, "时间 / h", "干基含水率 / (kg/kg)")
    axes[0].set_xlim(0, meta["end_s"] / 3600)
    axes[0].legend(loc="upper right")
    axes[1].set_xlim(max(0, meta["end_s"] / 3600 - 12), meta["end_s"] / 3600 + 0.3)
    axes[1].set_ylim(0.045, 0.24)
    axes[1].set_title("末期全域达标")
    axes[1].text(
        0.04,
        0.96,
        f"临界 {meta['event']['critical_s']/3600:.8f} h\n"
        f"严格结束 {meta['end_s']/3600:.8f} h",
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=9,
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
    fig, ax = plt.subplots(figsize=(8.2, 5.1), layout="constrained")
    colors = plt.colormaps["viridis"](np.linspace(0.05, 0.9, len(selected)))
    for color, index in zip(colors, selected):
        t = times[index]
        R = float(np.interp(t, radius[:, 0], radius[:, 1]))
        label = f"{t/3600:g} h" if index != selected[-1] else f"结束 {t/3600:.4f} h"
        ax.plot(100 * R * x, states[index, 1], color=color, lw=1.6, label=label)
        ax.scatter([100 * R], [states[index, 1, -1]], s=15, color=color, zorder=4)
    ax.axhline(0.15, color="#565C62", ls="--", lw=0.9)
    finish_axes(ax, "距药材中心的实际距离 / cm", "干基含水率 / (kg/kg)")
    ax.set_xlim(left=0)
    ax.legend(ncol=2, loc="upper right")
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
    boundary_handle = ax.lines[0]
    threshold_handle = Line2D(
        [], [], color="white", ls="--", lw=1.6, path_effects=outline,
        label="含水率 0.15 kg/kg 等值线",
    )
    ax.legend(handles=[boundary_handle, threshold_handle], loc="upper right")
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
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3), layout="constrained")
    for column, label, color, style in [
        (2, "药材中心", COLORS[0], "-"),
        (3, "药材表面", COLORS[3], "-"),
        (17, "对应烘房环境", "#444A50", "--"),
    ]:
        axes[0].plot(
            h[:, 0] / 3600, h[:, column], style, color=color, lw=1.3, label=label
        )
    axes[0].set_xlim(0, 6)
    finish_axes(axes[0], "时间 / h", "温度 / °C", "初期热传递")
    axes[0].legend()
    for T_column, C_column, label, color in [
        (2, 4, "中心局部扩散系数", COLORS[0]),
        (3, 5, "表面局部扩散系数", COLORS[3]),
    ]:
        D = (
            4.2e-4
            * np.exp(-0.30 / h[:, C_column])
            * np.exp(-3850 / (h[:, T_column] + 273.15))
        )
        axes[1].semilogy(h[:, 0] / 3600, D, color=color, lw=1.3, label=label)
    axes[1].set_xlim(0, meta["end_s"] / 3600)
    finish_axes(axes[1], "时间 / h", "局部扩散系数 / (m²/s)", "局部温湿状态决定扩散")
    axes[1].legend()
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
    """Show executed levels, the chosen resolution, and adjacent field acceptance."""
    fig = plt.figure(figsize=(11.8, 7.0), layout="constrained")
    grid = fig.add_gridspec(3, 2, height_ratios=(3.2, 1.65, 0.55))
    table_rows, evidence, sources = [], {}, ["review/verification.json"]
    orders = []
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
        ax = fig.add_subplot(grid[0, column])
        ax.bar(xpos, offsets_ms, width=0.32, color=color, alpha=0.72, zorder=3)
        ax.plot(xpos, offsets_ms, "o", color=color, ms=5, zorder=4)
        ax.scatter(
            [chosen_index], [offsets_ms[chosen_index]], s=125,
            facecolors="none", edgecolors="#B4433D", linewidths=1.6,
            zorder=5, label="主方案网格" if key == "space" else "主方案时间倍率",
        )
        for x, value in zip(xpos, offsets_ms):
            ax.annotate(
                f"{value:.2f}", (x, value), xytext=(0, 8),
                textcoords="offset points", ha="center", va="bottom", fontsize=10,
            )
        span = max(float(offsets_ms.max()), 0.01)
        ax.set_ylim(-0.12 * span, 1.55 * span)
        ax.set_xlim(-0.5, len(levels) - 0.5)
        ax.set_xticks(xpos, [f"{level:g}" for level in levels])
        ax.axhline(0, color="#7C858C", lw=0.7)
        finish_axes(
            ax,
            "径向区间数 N（由粗到细）" if key == "space" else "时间步倍率（由粗到细）",
            "相对本组最细解的时长绝对差 / ms",
        )
        ax.grid(False, axis="x")
        setting = (
            f"时间倍率 {configs[0]['factor']:g}"
            if key == "space" else f"N={configs[0]['n']}"
        )
        ax.set_title(f"({'ab'[column]}) {label}加密 · {setting}", loc="left", pad=10)
        ax.text(
            0.025, 0.95, f"参考时长 {critical[-1] / 3600:.4f} h",
            transform=ax.transAxes, va="top", fontsize=9, color="#505960",
        )
        ax.legend(loc="upper right", frameon=False, fontsize=9)
        differences = [item["metrics"]["full"]["C"] for item in items]
        order = (
            float(np.log2(differences[-2] / differences[-1]))
            if len(items) >= 2 else None
        )
        orders.append(f"{label} {order:.2f}" if order is not None else f"{label}未估计")
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

    table_ax = fig.add_subplot(grid[1, :])
    table_ax.axis("off")
    table_ax.set_title("(c) 相邻加密级的全场差与验收", loc="left", fontsize=11, pad=8)
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
    targets = report["targets"]["thresholds"]
    note_ax = fig.add_subplot(grid[2, :])
    note_ax.axis("off")
    note_ax.text(
        0, 0.92,
        f"验收：ΔC ≤ {targets['field_C_kg_kg']:g} kg/kg；"
        f"ΔT ≤ {targets['field_T_C']:g} °C；|Δt*| ≤ {targets['event_change_s']:g} s。"
        f"含水率观察阶：{'，'.join(orders)}。",
        transform=note_ax.transAxes, va="top", fontsize=8.7, color="#505960",
    )
    note_ax.text(
        0, 0.26, "最细级作为参考，自差为 0；上述差值均不是连续解误差界或工艺精度保证。",
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
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), layout="constrained")
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
        )
        label = f"{hours:.4f} h" if dry else f"{hours:g} h 内未达标"
        axes[0].text(hours + 0.8, j, label, va="center", fontsize=9)
    axes[0].set_yticks(range(4), labels)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 92)
    finish_axes(axes[0], "临界时长或数据覆盖 / h", "", "相同末值环境下的物性与几何")
    sensitivity = cases["appendix4_mean30"]
    for j, (value, label) in enumerate(
        ((meta, "末次实测值延拓"), (sensitivity, "末 30 min 算术均值延拓"))
    ):
        dry = value["event"] is not None
        hours = value["event"]["critical_s"] / 3600 if dry else 72
        axes[1].barh(j, hours, color=[COLORS[0], COLORS[4]][j], height=0.45, alpha=0.85)
        axes[1].text(
            hours + 0.4,
            j,
            f"{hours:.4f} h" if dry else "72 h 内未达标",
            va="center",
            fontsize=9,
        )
    axes[1].set_yticks([0, 1], ["末值", "末段均值"])
    axes[1].invert_yaxis()
    axes[1].set_xlim(
        0,
        max(
            (
                72
                if sensitivity["event"] is None
                else sensitivity["event"]["critical_s"] / 3600
            ),
            meta["event"]["critical_s"] / 3600,
        )
        * 1.3,
    )
    sensitivity_title = "附录 4 实测收缩的环境敏感性"
    if sensitivity["event"] is not None:
        change_min = (
            sensitivity["event"]["critical_s"] - meta["event"]["critical_s"]
        ) / 60
        sensitivity_title += f"\n均值延拓变化 {change_min:+.2f} min"
    finish_axes(axes[1], "临界时长 / h", "", sensitivity_title)
    save_figure(
        fig,
        "07_scenarios_and_sensitivity",
        ["review/verification.json", "cases/*/metadata.json"],
        "4 h 后边界为确定性假设；未达标情景仅报告覆盖内状态，不推断无限烘干时长；效应不普遍可加。",
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
        (FIGURE_NAMES[0], plot_inputs, (env, radius, meta)),
        (FIGURE_NAMES[1], plot_history, (summary, meta)),
        (FIGURE_NAMES[2], plot_profiles, (summary, meta, radius)),
        (FIGURE_NAMES[3], plot_heatmap, (report, meta, summary, radius)),
        (FIGURE_NAMES[4], plot_temperature, (summary, meta)),
        (FIGURE_NAMES[5], plot_convergence, (report, meta)),
        (FIGURE_NAMES[6], plot_scenarios, (report, meta)),
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
