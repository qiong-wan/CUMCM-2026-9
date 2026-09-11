"""Render axisymmetric Problem 2 fields at ten times in a four-by-five grid.

Run in Solution: python src/problem-2/generate_cross_sections.py
The existing result workbook is read-only. Outputs use the image directory;
visual previews are saved directly from the live canvas outside output/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from generate_figures import (
    DEFAULT_IMAGE_DIR,
    DEFAULT_PREVIEW_DIR,
    NEUTRAL_COLOR,
    RESULT_FILE,
    T_END,
    Fields,
    configure_matplotlib,
    load_results,
    plt,
    save_figure,
    sha256,
)
from matplotlib.colors import Normalize
from matplotlib.patches import Circle
from matplotlib.ticker import MaxNLocator


DEFAULT_TIMES_MIN = (10, 20, 30, 40, 50, 60, 90, 120, 150, 180)
DEFAULT_TIMES_S = tuple(minute * 60 for minute in DEFAULT_TIMES_MIN)
DISPLAY_SIZE = 601
FIGURE_NAMES = {
    3: "fig7_cross_sections",
    10: "fig8_cross_sections_10times",
}


def expand_profile(
    radii_cm: np.ndarray,
    values: np.ndarray,
    distance_cm: np.ndarray,
) -> np.ma.MaskedArray:
    """Linearly map a radial profile into the disk, masking its exterior."""
    inside = distance_cm <= radii_cm[-1]
    displayed = np.zeros(distance_cm.shape, dtype=float)
    displayed[inside] = np.interp(distance_cm[inside], radii_cm, values)
    return np.ma.array(displayed, mask=~inside)


def build_figure(
    fields: Fields, times_s: tuple[int, ...]
) -> tuple[plt.Figure, dict]:
    """Arrange each variable chronologically and verify the display mapping."""
    if len(times_s) not in FIGURE_NAMES:
        raise ValueError("截面图支持3个或10个时刻。")
    columns = 5 if len(times_s) == 10 else 3
    rows_per_variable = len(times_s) // columns
    total_rows = 2 * rows_per_variable
    row_indices = np.searchsorted(fields.times, times_s)
    if not np.array_equal(fields.times[row_indices], times_s):
        raise ValueError("所选时刻不在工作簿时间轴上，未进行时间插值。")

    radius = float(fields.radii_cm[-1])
    coordinates = np.linspace(-radius, radius, DISPLAY_SIZE)
    x_cm, y_cm = np.meshgrid(coordinates, coordinates)
    distance_cm = np.hypot(x_cm, y_cm)
    spacing = float(coordinates[1] - coordinates[0])
    extent = (
        -radius - spacing / 2,
        radius + spacing / 2,
        -radius - spacing / 2,
        radius + spacing / 2,
    )
    temperature_limits = (
        float(np.floor(fields.temperature.min())),
        float(np.ceil(fields.temperature.max())),
    )
    moisture_limits = (
        float(np.floor(fields.moisture.min() * 10) / 10),
        float(np.ceil(fields.moisture.max() * 100) / 100),
    )
    specifications = (
        (fields.temperature, "温度", "°C", "turbo", temperature_limits),
        (fields.moisture, "干基含水率", "kg/kg", "viridis", moisture_limits),
    )

    figure, axes = plt.subplots(
        total_rows, columns,
        figsize=(16.4, 12.8) if len(times_s) == 10 else (11.6, 7.2),
        layout="constrained",
    )
    figure.suptitle("问题2  圆柱横截面温度与含水率分布", fontsize=14)
    checks = []
    for variable_index, (field, variable, unit, palette, limits) in enumerate(
        specifications
    ):
        first_row = variable_index * rows_per_variable
        normalization = Normalize(vmin=limits[0], vmax=limits[1])
        color_map = plt.get_cmap(palette).copy()
        color_map.set_bad("white", alpha=0)
        for time_index, (second, index) in enumerate(
            zip(times_s, row_indices, strict=True)
        ):
            row_offset, column = divmod(time_index, columns)
            row = first_row + row_offset
            axis = axes[row, column]
            profile = field[index]
            displayed = expand_profile(
                fields.radii_cm, profile, distance_cm
            )
            midpoint = DISPLAY_SIZE // 2
            node_columns = np.rint(
                (fields.radii_cm + radius) / spacing
            ).astype(int)
            node_error = float(np.max(np.abs(
                displayed.data[midpoint, node_columns] - profile
            )))
            symmetry_error = float(max(
                np.max(np.abs(displayed.data - displayed.data[::-1, :])),
                np.max(np.abs(displayed.data - displayed.data[:, ::-1])),
                np.max(np.abs(displayed.data - displayed.data.T)),
            ))
            values = displayed.compressed()
            mask_matches_disk = bool(np.array_equal(
                displayed.mask, distance_cm > radius
            ))
            mask_symmetric = bool(
                np.array_equal(displayed.mask, displayed.mask[::-1, :])
                and np.array_equal(displayed.mask, displayed.mask[:, ::-1])
            )
            no_overshoot = bool(
                values.min() >= profile.min() - 1e-12
                and values.max() <= profile.max() + 1e-12
            )
            if (
                node_error > 1e-10
                or symmetry_error > 1e-10
                or not mask_matches_disk
                or not mask_symmetric
                or not no_overshoot
            ):
                raise RuntimeError("截面映射的节点、对称性或值域检查失败。")

            image = axis.imshow(
                displayed,
                origin="lower",
                extent=extent,
                cmap=color_map,
                norm=normalization,
                interpolation="none",
            )
            boundary = Circle(
                (0, 0), radius, fill=False,
                edgecolor=NEUTRAL_COLOR, linewidth=0.6,
            )
            axis.add_patch(boundary)
            image.set_clip_path(boundary)
            panel_letter = chr(97 + variable_index * len(times_s) + time_index)
            show_x_labels = row_offset == rows_per_variable - 1
            axis.set(
                title=f"({panel_letter}) {variable}  "
                      f"t = {second / 60:g} min",
                xlim=(-radius, radius), ylim=(-radius, radius),
                aspect="equal", xticks=(-radius, 0, radius),
                yticks=(-radius, 0, radius),
                xlabel="x / cm" if show_x_labels else "",
                ylabel="y / cm" if column == 0 else "",
            )
            axis.tick_params(
                length=3, width=0.6,
                labelbottom=show_x_labels, labelleft=column == 0,
            )
            for spine in axis.spines.values():
                spine.set_linewidth(0.6)
                spine.set_color(NEUTRAL_COLOR)
            checks.append({
                "variable": variable,
                "time_s": int(second),
                "time_min": second / 60,
                "figure_row": row + 1,
                "figure_column": column + 1,
                "workbook_row": (
                    None if second == 0 and fields.initial_state_added
                    else int(second) + (
                        1 if fields.initial_state_added else 2
                    )
                ),
                "source_profile": profile.tolist(),
                "center_value": float(profile[0]),
                "surface_value": float(profile[-1]),
                "source_min": float(profile.min()),
                "source_max": float(profile.max()),
                "node_max_abs_error": node_error,
                "symmetry_max_abs_error": symmetry_error,
                "exterior_mask_correct": mask_matches_disk,
                "mask_symmetric": mask_symmetric,
                "no_interpolation_overshoot": no_overshoot,
            })
        colorbar = figure.colorbar(
            image,
            ax=axes[first_row:first_row + rows_per_variable, :].ravel().tolist(),
            fraction=0.018 if rows_per_variable == 2 else 0.028,
            pad=0.02,
            aspect=40 if rows_per_variable == 2 else 25,
        )
        colorbar.set_label(f"{variable} / ({unit})")
        ticks = MaxNLocator(nbins=5).tick_values(*limits)
        ticks = ticks[(ticks > limits[0]) & (ticks < limits[1])]
        colorbar.set_ticks(np.concatenate(([limits[0]], ticks, [limits[1]])))
        colorbar.outline.set_linewidth(0.6)

    return figure, {
        "times_s": list(times_s),
        "times_min": [second / 60 for second in times_s],
        "layout_rows": total_rows,
        "layout_columns": columns,
        "rows_per_variable": rows_per_variable,
        "radius_cm": radius,
        "source_radial_nodes_cm": fields.radii_cm.tolist(),
        "display_grid_shape": [DISPLAY_SIZE, DISPLAY_SIZE],
        "temperature_color_limits_C": list(temperature_limits),
        "moisture_color_limits_kg_kg": list(moisture_limits),
        "color_limits_basis": "all 0--3 h fields including initial state",
        "display_method": "axisymmetric radial piecewise-linear mapping",
        "panels": checks,
    }


def write_notes(output_dir: Path, manifest: dict) -> None:
    """Record the numbered display derivation, source mapping, and limits."""
    times = "、".join(f"{t / 60:g}" for t in manifest["times_s"])
    rows_per_variable = manifest["rows_per_variable"]
    figure_name = manifest["figure"]["name"]
    time_arguments = " ".join(str(t) for t in manifest["times_s"])
    notes = rf"""# 问题2圆柱截面图说明

## 图题与布局

建议图题：圆柱药材在{times} min时的横截面温度与干基含水率分布。
采用{manifest['layout_rows']}行、{manifest['layout_columns']}列布局，
前{rows_per_variable}行为温度，后{rows_per_variable}行为干基含水率；每种变量
按从左到右、从上到下的顺序排列时刻，上下两组对应位置采用相同时刻。
横纵坐标均为cm，药材半径为2 cm。同一变量的全部子图共用色标，
便于直接比较不同时刻的绝对数值。
温度单位为°C，干基含水率单位为kg/kg，与工作簿“水分浓度”表对应。

## 数据与审计

唯一数值输入为 `output/problem-2/result2.xlsx` 的“温度”和“水分浓度”表。
读取时核对字段类型、缺失、有限值、物理定义域，以及逐秒时间轴和21个径向节点。
绘制所选正时刻的实际数据行，不进行时间插值。节点位置为0至2 cm、间隔0.1 cm。
若工作簿从1 s开始，复用读取函数在内存添加题给0 s均匀初值28°C、2.55 kg/kg，
用于确定整个0至3 h的统一色标；如显式选择0 s，则绘制该题给初值，并将清单中的
源行号记为null。原始工作簿保持不变，输入SHA-256和各子图
源行号、完整径向数据与检查结果见 `{manifest['manifest_filename']}`。

## 从径向场到圆形截面

本图沿用第二问的一维轴对称假设：温度与含水率只随半径和时间变化，
忽略周向、轴向变化与端面效应，药材不收缩。现有求解结果通过坐标映射展开，
没有重新求解二维方程，也没有引入新物理参数或优化目标。

以横截面中心为原点，由直角三角形的距离关系得：

$$r^2=x^2+y^2.\tag{{1}}$$

由于径向距离非负，对公式（1）两侧取非负平方根得：

$$r(x,y)=\sqrt{{x^2+y^2}},\quad 0\le r\le R.\tag{{2}}$$

对于相邻输出节点之间的距离，定义无量纲插值权重：

$$\alpha=\frac{{r-r_i}}{{r_{{i+1}}-r_i}},
\quad r_i\le r\le r_{{i+1}},\quad 0\le\alpha\le1.\tag{{3}}$$

由公式（3），对变量u的相邻节点值作线性组合：

$$\widetilde u(r,t)=(1-\alpha)u(r_i,t)
+\alpha u(r_{{i+1}},t),\quad u\in\{{T,C\}}.\tag{{4}}$$

将公式（2）代入公式（3），再代入公式（4），得到截面显示值：

$$U(x,y,t)=\widetilde u\!\left(\sqrt{{x^2+y^2}},t\right),
\quad x^2+y^2\le R^2.\tag{{5}}$$

公式（5）在601×601显示网格上计算，圆外区域掩膜为空白，并在绘制时沿圆周裁切。
由公式（3）、（4）可得，显示值是相邻节点值的凸组合，因而不会产生超出两端值的
新极值。该步骤仅改善截面显示的连续性；显示像素数不代表求解精度，数值信息仍
来自21个实际输出节点。未作高阶平滑，也未改变原数据或伪造周向差异。

## 色标与验证指标

温度和含水率分别使用turbo与viridis色图。色标范围由完整0至3 h结果及初值确定，
温度下上限分别向下、向上取整，含水率下限向下取一位小数、上限向上取两位小数，
所以没有截断实际值。每种变量的全部子图复用同一归一化对象。

节点一致性检查在正x轴的21个原始节点比较显示值和工作簿值，其指标为：

$$E_{{\mathrm{{node}},u}}=
\max_i|U(r_i,0,t)-u(r_i,t)|.\tag{{6}}$$

根据公式（2）、（5）的旋转对称性，检查横纵翻转与转置的最大差：

$$E_{{\mathrm{{sym}},u}}=\max_{{x,y}}\max\left\{{
|U(x,y,t)-U(-x,y,t)|,
|U(x,y,t)-U(x,-y,t)|,
|U(x,y,t)-U(y,x,t)|\right\}}.\tag{{7}}$$

公式（6）、（7）的单位与所检验变量一致，数值越小表示映射越一致；
代码使用1e-10作为浮点显示检查容差。另核对圆外掩膜、掩膜对称性和插值值域。
这些指标检查的是绘图映射，不是偏微分方程的数值收敛或物理误差。

## 论文解读边界

可以比较表面与中心的升温差、失水差，以及它们随时间的变化；共享色标保证比较
口径一致。不得仅凭颜色判断等温程度或高精度梯度，具体差值应引用原表。
圆形分布是轴对称假设下的截面展示，不能据此声称验证了周向均匀性、端面效应
或蒸发潜热模型。细窄边界层的显示受工作簿0.1 cm输出间隔限制。

## 文件与复现

- `{figure_name}.png`：{manifest['dpi']} dpi论文位图。
- `{figure_name}.svg`：截面图像嵌入，文字与轴线保留矢量路径。
- 临时JPG预览由同一画布直接生成，没有读回output目录中的PNG。

本图复现命令：

`python src/problem-2/generate_cross_sections.py --times-s {time_arguments}`

默认选取10、20、30、40、50、60、90、120、150、180 min，生成4行5列图。
`--times-s`支持3个或10个严格递增时刻（单位s，限0至10800 s），传入3个时刻
时生成2行3列图。可用 `--dpi` 调整导出分辨率。
"""
    (output_dir / manifest["notes_filename"]).write_text(
        notes, encoding="utf-8"
    )


def main() -> None:
    """Generate the requested comparison from audited, immutable results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--times-s", type=int, nargs="+", default=DEFAULT_TIMES_S,
        metavar="SECOND",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument(
        "--preview-dir", type=Path, default=DEFAULT_PREVIEW_DIR
    )
    args = parser.parse_args()
    times_s = tuple(args.times_s)
    if not 100 <= args.dpi <= 600:
        parser.error("--dpi must be between 100 and 600")
    if len(times_s) not in FIGURE_NAMES:
        parser.error("--times-s needs 3 or 10 times")
    if (
        times_s[0] < 0 or times_s[-1] > T_END
        or any(first >= second for first, second in zip(times_s, times_s[1:]))
    ):
        parser.error("--times-s needs increasing times in 0--10800 s")
    figure_name = FIGURE_NAMES[len(times_s)]
    manifest_filename = (
        "cross_sections_10times_manifest.json" if len(times_s) == 10
        else "cross_sections_manifest.json"
    )
    notes_filename = (
        "十时刻截面图说明.md" if len(times_s) == 10 else "截面图说明.md"
    )

    fingerprint = sha256(RESULT_FILE)
    fields = load_results()
    font = configure_matplotlib()
    figure, metadata = build_figure(fields, times_s)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.preview_dir.mkdir(parents=True, exist_ok=True)
    asset = save_figure(
        figure, figure_name, args.output_dir, args.preview_dir, args.dpi
    )
    if asset["text_outside_original_canvas"]:
        raise RuntimeError("图中文字超出画布，请检查布局。")
    if sha256(RESULT_FILE) != fingerprint:
        raise RuntimeError("绘图期间结果文件发生变化，请重新生成。")
    manifest = {
        "source_file": str(RESULT_FILE),
        "source_sha256": fingerprint,
        "source_files_unchanged": True,
        "source_sheets": ["温度", "水分浓度"],
        "font": font,
        "dpi": args.dpi,
        "figure": asset,
        "manifest_filename": manifest_filename,
        "notes_filename": notes_filename,
        "preview_file": str(
            (args.preview_dir / f"{figure_name}.jpg").resolve()
        ),
        "output_pngs_read_back": False,
        **metadata,
    }
    write_notes(args.output_dir, manifest)
    (args.output_dir / manifest_filename).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Generated {figure_name}.png and .svg", flush=True)
    print("PASS: source unchanged; nodes, symmetry, mask and range.")


if __name__ == "__main__":
    main()
