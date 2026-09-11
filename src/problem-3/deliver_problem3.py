"""Generate problem-3 tables, publication figures and a data-backed report."""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform

os.environ["MPLCONFIGDIR"] = str(Path(__file__).resolve().parents[2] / "output/problem-3/.mplconfig")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import openpyxl

from solve_problem3 import OUT, ROOT, R, Environment, audit, dump_json
from verify_problem3 import load_case


def save_figure(fig, name):
    """Save an editable scientific SVG and a separate authorized QA raster."""
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    fig.savefig(ROOT / "src" / "problem-3" / f"{name}_qa.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def figures(data, mean_data, report, values):
    """Plot actual solved states, environmental assumptions and convergence evidence."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none"})
    t = data["times"] / 3600
    fig, ax = plt.subplots(1, 2, figsize=(11.4, 4.1), layout="constrained")
    ax[0].plot(t, data["outputs"][:, 1, 0], label="Center / full-field maximum", color="#176c9b")
    ax[0].plot(data["history"][:, 0] / 3600, data["history"][:, 3], label="Volume-weighted mean", color="#aa662e")
    ax[0].plot(t, data["outputs"][:, 1, -1], label="Surface", color="#247c56")
    ax[0].axhline(.15, color="#9f3546", linestyle="--", label="Threshold 0.15")
    ax[0].set(xlabel="Time from start (h)", ylabel="Dry-basis moisture (kg/kg)", title="Moisture trajectories")
    ax[0].legend(fontsize=8)
    ax[1].plot(t, data["outputs"][:, 1, 0], color="#176c9b", label="Hold last observation")
    ax[1].plot(mean_data["times"] / 3600, mean_data["outputs"][:, 1, 0], color="#a45178", label="Hold last 30-min mean")
    ax[1].axhline(.15, color="#9f3546", linestyle="--")
    ax[1].set(xlim=(55.5, 58), ylim=(.148, .153), xlabel="Time from start (h)",
              ylabel="Maximum moisture (kg/kg)", title="Endpoint and boundary sensitivity")
    ax[1].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=.2)
    save_figure(fig, "drying_history")

    fig, ax = plt.subplots(1, 2, figsize=(11.4, 4.1), layout="constrained")
    choose = [21600, 43200, 64800, 86400, 129600, 172800, data["full_times"][-1]]
    for t_value, color in zip(choose, ["#176c9b", "#aa662e", "#247c56", "#a45178", "#536d35", "#444444", "#c23b34"]):
        idx = int(np.argmin(abs(data["full_times"] - t_value)))
        C = data["full_states"][idx, 1]
        label = f"{t_value / 3600:.2f} h" if t_value == choose[-1] else f"{t_value / 3600:.0f} h"
        for a in ax:
            a.plot(data["r"] * 100, C, color=color, label=label)
    ax[0].set(xlabel="Radius (cm)", ylabel="Moisture (kg/kg)", title="Full radial fields", xlim=(0, 2))
    ax[1].set(xlabel="Radius (cm)", ylabel="Moisture (kg/kg)", title="Resolved surface layer", xlim=(1.97, 2), ylim=(.045, .21))
    ax[0].legend(ncol=2, fontsize=8)
    for a in ax:
        a.grid(alpha=.2)
        a.axhline(.15, color="#888888", linestyle="--", linewidth=.8)
    save_figure(fig, "radial_profiles")

    fig, ax = plt.subplots(1, 2, figsize=(11.4, 4.1), layout="constrained")
    selected = t <= 8
    ax[0].plot(t[selected], data["outputs"][selected, 0, 0], label="Material center", color="#176c9b")
    ax[0].plot(t[selected], data["outputs"][selected, 0, -1], label="Material surface", color="#247c56")
    boundary_t = np.r_[values[:, 0], 8 * 3600]
    ax[0].plot(boundary_t / 3600, np.r_[values[:, 1], values[-1, 1]], label="Air, last-value tail", color="#aa662e", linewidth=1)
    ax[0].set(xlabel="Time (h)", ylabel="Temperature (degC)", title="Temperature is solved throughout")
    ax[0].legend(fontsize=8)
    ax[1].plot(values[:, 0] / 3600, values[:, 2], color="#444444", label="Observed air moisture")
    for mode, color in [("last", "#176c9b"), ("mean30", "#a45178")]:
        env = Environment(values, mode)
        ax[1].plot([4, 8], [env.tail[1], env.tail[1]], color=color, label=f"Tail: {mode}")
    ax[1].set(xlabel="Time (h)", ylabel="Air moisture (kg/kg)", title="Observed interval remains unchanged")
    ax[1].legend(fontsize=8)
    for a in ax:
        a.axvline(4, color="#888888", linestyle=":")
        a.grid(alpha=.2)
    save_figure(fig, "temperature_and_environment")

    spatial, temporal = report["space_convergence"], report["time_convergence"]
    fig, ax = plt.subplots(1, 2, figsize=(11.4, 4.1), layout="constrained")
    ax[0].loglog([v["fine_N"] for v in spatial], [abs(v["event_difference_s"]) for v in spatial], "o-", color="#176c9b")
    ax[0].set(xlabel="Fine mesh intervals N", ylabel="Change in critical time (s)", title="Spatial refinement, fixed time strategy")
    ax[1].loglog([v["factors"][1] for v in temporal], [v["all_saved_fields"]["C_max_kg_kg"] for v in temporal], "o-", color="#a45178")
    ax[1].set(xlabel="Fine time-step multiplier", ylabel="Full-field moisture difference (kg/kg)", title="Time refinement, fixed N=12800")
    for a in ax:
        a.grid(which="both", alpha=.2)
    save_figure(fig, "convergence")


def main():
    """Export only from the selected case after necessary numerical checks pass."""
    report = json.loads((OUT / "verification.json").read_text(encoding="utf-8"))
    if report["status"] != "PASS":
        raise RuntimeError("Numerical verification is not PASS")
    data, meta = load_case("production")
    mean_data, mean_meta = load_case("mean12800")
    values, input_report = audit()
    times, output = data["times"], data["outputs"]
    end, event = meta["event"]["end_s"], meta["event"]
    expected = np.r_[np.arange(60, np.floor(end / 60) * 60 + 1, 60), end]
    if not np.array_equal(times[1:], expected):
        raise RuntimeError("Missing or duplicate regular/end sample")
    header = [input_report["template"]["rows"][0][0]] + [j / 10 for j in range(21)]
    payload = {"header": header, "rows": np.column_stack([times[1:], output[1:, 1]]).tolist()}
    dump_json(OUT / "workbook_payload.json", payload)
    np.savetxt(OUT / "moisture_60s_full_precision.csv", np.column_stack([times[1:], output[1:, 1]]),
               delimiter=",", comments="", fmt="%.17g",
               header="time_s," + ",".join(f"r_{j / 10:.1f}_cm" for j in range(21)))
    table_idx = [i for i, t in enumerate(times) if t > 0 and t < end and t % 21600 == 0]
    table_idx.append(len(times) - 1)
    table = np.column_stack([times[table_idx] / 3600, times[table_idx],
                             output[table_idx, 1][:, [0, 5, 10, 15, 20]]])
    np.savetxt(OUT / "table5_moisture.csv", table, delimiter=",", comments="", fmt="%.17g",
               header="time_h,time_s,r_0_cm,r_0.5_cm,r_1_cm,r_1.5_cm,r_2_cm")
    lines = ["# 表 5 药材烘干过程的水分浓度", "", "单位：时间 h，距离 cm，含水率 kg/kg。", "",
             "| 时间/h | 0 cm | 0.5 cm | 1 cm | 1.5 cm | 2 cm |", "|---:|---:|---:|---:|---:|---:|"]
    for row in table:
        label = f"{row[0]:.0f}" if row[1] != end else f"烘干结束 {row[0]:.8f}"
        lines.append("| " + label + " | " + " | ".join(f"{v:.4f}" for v in row[2:]) + " |")
    lines += ["", f"结束行取 {end:.12f} s 的原始场。中心未舍入值为 {output[-1, 1, 0]:.15f}，严格小于 0.15；显示四位小数后为 0.1500。",
              "CSV 保留完整精度，表格显示格式不等于每一末位均已被证明准确。"]
    table_text = "\n".join(lines) + "\n"
    (OUT / "table5.md").write_text(table_text, encoding="utf-8")
    boundary_times = np.r_[values[:, 0], np.arange(14460, np.ceil(mean_meta["event"]["end_s"] / 60) * 60 + 1, 60)]
    boundary_data = np.array([[t, *Environment(values, "last")(t),
                              *Environment(values, "mean30")(t)] for t in boundary_times])
    np.savetxt(OUT / "derived_boundaries.csv", boundary_data, delimiter=",", comments="", fmt="%.17g",
               header="time_s,T_last_degC,C_last_kg_kg,T_mean30_degC,C_mean30_kg_kg")
    figures(data, mean_data, report, values)
    space, temporal, error = report["space_convergence"], report["time_convergence"], report["empirical_error"]
    echeck, sensitivity = report["event"], report["boundary_sensitivity"]
    main_steps = meta["steps"]
    result_lines = [
        "# A 题第三问计算结果与验证记录", "",
        f"在 14400 s 之后温度保持 50.165°C、环境水分浓度保持 0.04986 kg/kg 的假设下，烘干临界时间估计为 **{event['estimate_s'] / 3600:.8f} h**（{event['estimate_s']:.12f} s）。经全域未舍入场确认严格达标的交付结束时刻为 **{end / 3600:.8f} h**（{end:.12f} s）。时间均从烘干开始计，未拼接前两问状态。",
        "", "## 1. 输入与主方案", "",
        "附件 1 的 241 条观测均为有限数值，无缺失、重复时间或重复整条记录，时间严格升序、间隔均为 60 s。覆盖 0 至 14400 s，末值已重复核实。温度范围 28 至 50.246°C，环境水分浓度范围 0.01963 至 0.05025 kg/kg。原始观测没有修改，派生边界只在观测区间之外作情景延拓。",
        "",
        f"主解使用 N={meta['N']} 个径向区间、{meta['N'] + 1} 个节点，网格宽 {R / meta['N'] * 1e6:.4f} μm，SDIRK2 二阶隐式积分，时间倍率 {meta['factor']}。名义步长上限在观测区间为 1 s，后段为 15 s；启动阶段更小，实际最小/最大步长为 {meta['step_range_s'][0]:.8g}/{meta['step_range_s'][1]:.8g} s。60 s 仅是交付间隔。共接受 {main_steps} 步，拒绝 {meta['rejected_steps']} 步，阶段最大迭代次数 {meta['max_iterations']}。",
        "",
        "物性、初边值、完整推导、离散矩阵和每项指标定义见 `src/problem-3/模型与算法说明.md` 的连续编号公式（1）至（65）。内部不舍入，温度持续求解至结束。",
        "", "## 2. 全域事件与严格小于条件", "",
        "以下依据模型说明公式（30）、公式（31）、公式（51）至公式（53）。",
        "", "| 项目 | 实算值 |", "|---|---:|",
        f"| 未达标夹逼端点 / s | {event['lower_s']:.12f} |",
        f"| 该端点最大含水率减阈值 / (kg/kg) | {event['g_lower']:.12e} |",
        f"| 已达标夹逼端点 / s | {event['upper_s']:.12f} |",
        f"| 该端点最大含水率减阈值 / (kg/kg) | {event['g_upper']:.12e} |",
        f"| 夹逼宽度 / s | {event['width_s']:.12f} |",
        f"| 严格结束时刻全域最大含水率 / (kg/kg) | {echeck['strict_end_max_C']:.15f} |",
        f"| 严格结束时刻阈值裕度 / (kg/kg) | {echeck['strict_end_margin']:.12e} |",
        f"| 结束最大值位置 / cm | {100 * echeck['max_location_m']:.8f} |",
        "",
        f"每个接受时刻均检查全部 {meta['N'] + 1} 个节点；全程最大值超出中心值的最大差为 {meta['max_C_above_center']:.3e} kg/kg，径向含水率反序的最大幅度为 {meta['radial_inversion_max']:.3e} kg/kg。由这些实测检查，中心是最后达标位置。全域分段线性重构无节点间超调，空间误差另由加密验证。没有用 21 个交付点替代全域检查。",
        "",
        f"把根定位容差从 0.001 s 改为 0.00001 s，临界估计变化 {echeck['root_tolerance_change_s']:.9f} s；同一事件前完整状态出发，把局部步拆为两个半步，变化 {echeck['local_step_change_s']:.9f} s。后者记为零时，仅表示两个结果落入同一定位小区间，不表示严格零误差。夹逼宽度只度量事件定位误差，不涵盖空间、累计时间或物理假设误差。",
        "", "## 3. 表 5", "", "\n".join(lines[4:]),
        "", "## 4. 空间收敛", "",
        "固定时间倍率为 1，比较同一物理时刻与嵌套节点，指标按公式（61）。全场列覆盖保存的启动截面及每 6 h 截面；交付列覆盖共同的全部 60 s 时刻和 21 个交付点。不同结束时刻的场不直接相减。",
        "", "| N 粗→细 | 全场 C 最大差/(kg/kg) | 交付点 C 最大差/(kg/kg) | 临界时长变化/s |", "|---|---:|---:|---:|"]
    for row in space:
        result_lines.append(f"| {row['coarse_N']}→{row['fine_N']} | {row['all_saved_fields']['C_max_kg_kg']:.8e} | {row['delivery']['C_max_kg_kg']:.8e} | {row['event_difference_s']:.8f} |")
    result_lines += ["",
        f"最细两级全场最大差 {space[-1]['all_saved_fields']['C_max_kg_kg']:.8e} kg/kg，出现在 {space[-1]['all_saved_fields']['C_max_time_s'] / 3600:g} h、半径 {space[-1]['all_saved_fields']['C_max_radius_m'] * 100:.8f} cm，说明表面薄层比固定交付点更敏感。启动 0.01 至 60 s 全场最大差 {space[-1]['startup_to_60s']['C_max_kg_kg']:.8e} kg/kg。共同临近终点 205800 s 的全场差为 {report['near_end_common_time_fields']['space6400_to_12800_full_C_max']:.8e} kg/kg。",
        "",
        f"最近三级临界时间的实测空间阶约为 {error['space_observed_order']:.4f}。按公式（63）估计 N=12800 剩余空间时长误差约 {error['space_fine_residual_time_estimate_s']:.4f} s，这是渐近估计而非严格上界，未把外推结果替代主解。",
        "", "## 5. 独立时间加密", "",
        "固定 N=12800，时间倍率依次为 1、0.5、0.25。主交付采用倍率 0.5，倍率 0.25 用于验证；两个场始终共同推进。",
        "", "| 时间倍率 | 全场 C 最大差/(kg/kg) | 全场 T 最大差/°C | 临界时长变化/s |", "|---|---:|---:|---:|"]
    for row in temporal:
        result_lines.append(f"| {row['factors'][0]}→{row['factors'][1]} | {row['all_saved_fields']['C_max_kg_kg']:.8e} | {row['all_saved_fields']['T_max_degC']:.8e} | {row['event_difference_s']:.8f} |")
    result_lines += ["",
        f"时间细化覆盖启动、6 h 截面及末期；主解与更细时间解在 205800 s 的全场含水率最大差为 {report['near_end_common_time_fields']['time_factorhalf_to_quarter_full_C_max']:.8e} kg/kg。完整分时段误差和差异位置见 verification.json。空间收敛与时间收敛分别开展，短时独立 BDF 对照也没有替代这两项。",
        "",
        f"按公式（62），事件附近最大含水率的下降斜率约为 {echeck['slope_kg_kg_per_s']:.9e} (kg/kg)/s；5×10⁻⁵ kg/kg 的最大值误差可放大为约 {echeck['time_amplification_s_per_5e_5']:.2f} s 的时间误差。因此毫秒级根定位不能解释为总时长有毫秒精度。结合最细空间/时间差，取约 **{error['conservative_numeric_allowance_s']:.0f} s** 的经验数值误差尺度（取两种最细时长差之和的两倍，加定位及严格终点裕度后向上取整），不是严格误差界或真实工艺安全余量。四位小数是格式，不能保证每个末位舍入结果都相同。",
        "", "## 6. 平衡、边界与物理趋势", "",
        "平衡定义按公式（55）、公式（56）、公式（58），均使用完整计算网格和实际隐式阶段权重。水分左侧取全域首末积分差，右侧独立累计表面通量；热方程累计各阶段变热容储存与表面热流。",
        "", "| 检查 | 实算值 |", "|---|---:|",
        f"| 水分独立平衡相对缺陷 | {meta['mass_relative_defect']:.8e} |",
        f"| 变热容热平衡相对缺陷 | {meta['heat_relative_defect']:.8e} |",
        f"| 最大尺度化迭代变化 | {meta['max_scaled_change']:.8e} |",
        f"| 最大原非线性方程尺度化缺陷 | {meta['max_scaled_residual']:.8e} |",
        f"| 热储存累计与边界累计乘 πL 后/J | {meta['cumulative_heat_storage_scaled'] * np.pi * .25:.9f} / {meta['cumulative_heat_boundary_scaled'] * np.pi * .25:.9f} |",
        f"| 水分状态积分变化（缩放量） | {meta['mass_change_scaled']:.12e} |",
        f"| 水分独立表面累计（缩放量） | {meta['cumulative_moisture_boundary_scaled']:.12e} |",
        "",
        "在诊断副本中把独立累计表面水分通量置零后，平衡相对缺陷为 1，验证能发现该故障。均匀平衡、关闭边界的共享通量抵消、变系数通量相消、非法状态拒绝、强制迭代失败报错均已执行。",
        "",
        f"独立 SciPy BDF 短时对照范围是 N=100、0 至 60 s，其与 SDIRK2 的末场温度/含水率最大差分别为 {report['synthetic_tests']['independent_BDF_field_error'][0]:.8e} °C、{report['synthetic_tests']['independent_BDF_field_error'][1]:.8e} kg/kg。它独立检查时间推进，但共享空间通量算子，不能当作独立空间方法。",
        "", "独立单侧梯度检查按公式（59）、公式（60），下表是 60 s 交付时刻及结束时刻的最大绝对值。", "",
        "| N | 热 Robin/(W/m²) | 湿 Robin/[(kg/kg)·m/s] | 中心温度导数/(°C/m) | 中心含水率导数/[(kg/kg)/m] |",
        "|---:|---:|---:|---:|---:|"]
    for row in report["boundary_convergence"] + [report["production_boundaries"]]:
        label = row["case"].replace("space", "")
        result_lines.append("| " + label + " | " + " | ".join(f"{x:.7e}" for x in row["delivery_abs_max_RT_RC_symT_symC"]) + " |")
    result_lines += ["",
        "初始水分角点不相容，t=0 的梯度残差不作验收。0.01、0.1、1、10、60 s 的额外重构值已记录在 JSON；早期薄层重构残差随网格加密下降，交付时刻的验收阈值为热 Robin 10⁻⁴ W/m²、湿 Robin 10⁻⁹ (kg/kg)·m/s、两个中心梯度绝对值各 10⁻⁶（各自单位）。原离散方程残差与重构误差分别报告。",
        "",
        f"全程温度范围 {meta['temperature_range'][0]:.9f} 至 {meta['temperature_range'][1]:.9f}°C；含水率范围 {meta['moisture_range'][0]:.12f} 至 {meta['moisture_range'][1]:.12f} kg/kg；局部 D 范围 {meta['diffusivity_range'][0]:.8e} 至 {meta['diffusivity_range'][1]:.8e} m²/s。所有状态有限、为正。主解最大单步温度下降幅度为 {-meta['temperature_step_decrease_min']:.8f}°C，反映环境正常波动，未强行要求温度单调。全域含水率未出现超出舍入噪声的径向反序或时间增加。不到 72 h 即已找到事件，本次无需延长，但程序保留未达标自动延长及记录逻辑。",
        "", "## 7. 后续环境敏感性", "",
        f"使用 12600 至 14400 s（含两个端点，31 点）算术均值：温度 {sensitivity['tail'][0]:.12f}°C，环境水分浓度 {sensitivity['tail'][1]:.15f} kg/kg。仅 t>14400 s 使用这些常数。主方案与均值方案在观测区间的全部保存交付值最大差为 {sensitivity['observed_interval_difference']:.1f}。两个情景都重新从均匀初值积分。",
        "",
        f"同样 N=12800、时间倍率 0.5 下，均值情景临界时间为 **{sensitivity['event_s'] / 3600:.8f} h**（{sensitivity['event_s']:.12f} s），比末值情景增加 **{sensitivity['difference_h']:.8f} h**，即 {sensitivity['difference_s']:.6f} s（{sensitivity['difference_s'] / 60:.4f} min）。此变化明显大于本次经验数值误差尺度。该对照同时改变了温度与环境水分浓度，没有将变化分别归因于其中某一个因素。",
        "", "## 8. 与第二问前 3 h 的衔接", "",
        "同一附录 3、同一初值、同一观测边界意味着连续模型相同。本问在前 3 h 每 60 s 的 21 点与第二问已有工作簿的只读快照比较。原审查报告针对后向 Euler 0.25 s；计算期间其他工作已更新第二问源码为启动 0.0025 s、后段 0.03125 s 的渐增步长，N 仍为 3200。已有工作簿也发生了外部更新，因此不推定其与某一版源码严格同批次，更不把它作为精确真值或续算状态。",
        "",
        f"对照最大温度差为 {report['previous_problem_comparison']['T']['max_abs_difference']:.8e}°C，最大含水率差为 {report['previous_problem_comparison']['C']['max_abs_difference']:.8e} kg/kg。差异包含时间算法、空间分辨率以及旧表每格最多 5×10⁻⁵ 的舍入不确定性；不能单凭此差判定本问错误，也不能用两法同时间步一致来替代时间收敛。对照样本已保存在 problem2_comparison_snapshot.npz，对照工作簿快照 SHA-256 为 `{report['previous_problem_comparison']['source_workbook_sha256']}`。",
        "", "## 9. 工作簿与一致性", "",
        f"result3.xlsx 仅含 Sheet1。A1 保持原模板文本，B 至 V 为 0、0.1、…、2.0 cm；A 列包含 60 至 {int(times[-2])} s 的全部 {len(times)-2} 个规则时刻，并追加 {end:.12f} s 的实际结束行。因此共 {len(times)} 行（含表头）、22 列，含水率数据格 {21*(len(times)-1)} 个。最后一行不是 60 s 整数倍，是明确的例外。结果格均存数值并设置 0.0000，存储值没有预先舍入。",
        "",
        "表 5、CSV、工作簿数据和结束时间均来自 production.npz 的同一状态序列。数值验收通过后才生成导出载荷；导出后由 verify_problem3.py --workbook 重读全部格值、格式、表头、时间、距离及表 5 对应点。最终工作簿验证状态见 workbook_verification.json，未生成该文件时不能声称 Excel 已通过检查。",
        "", "## 10. 验证状态及限制", "",
        "| 必要检查 | 状态 |", "|---|---|"]
    result_lines += [f"| {name} | {status} |" for name, status in report["checks"].items()]
    result_lines += ["", "| 未执行验证 | 状态 |", "|---|---|"]
    result_lines += [f"| {name} | {status} |" for name, status in report["skipped"].items()]
    result_lines += ["",
        f"共审计 {report['protected_files']['checked']} 个保护文件。关键模型输入（题目、原始附件、公共函数及第一问源码）哈希检查为 {report['protected_files']['critical_model_inputs_status']}。完整保护文件哈希比较为 **{report['protected_files']['status']}**：运行期间观察到其他工作的第二问文件更新，本问未写入这些文件，也没有覆盖首次哈希基线。数值验收 PASS 不覆盖这项外部文件不一致。初次阻止导出的记录另存 verification_initial_scope_failure.json。按项目要求没有读取 output 下的 PNG；图像 QA 副本保存在本问 src 目录。公式编号检查为 {report['formula_numbering']}，连续编号 1 至 65。",
        "", "观察到变更的外部文件：" + "、".join(report['protected_files']['changed']) + "。",
        "",
        "模型仍采用有效传质势，未计相变潜热，质量与焓闭合不完整，忽略端面、收缩及材料不均匀性。没有计算确定的潜热缺口倍数，没有证明严格温度上界，也没有把端面面积比例当作模型误差。本次环境敏感性只是一个有明确窗口的对照，不是对未知未来环境的概率置信区间。",
        "", "## 11. 图表", "",
        "![含水率轨迹与终点敏感性](drying_history.svg)", "",
        "![径向全场与表面薄层](radial_profiles.svg)", "",
        "![温度与环境](temperature_and_environment.svg)", "",
        "![空间与时间收敛](convergence.svg)", ""]
    (OUT / "结果与验证.md").write_text("\n".join(result_lines), encoding="utf-8")
    dump_json(OUT / "run_environment.json", {"Python": platform.python_version(),
              "NumPy": np.__version__, "SciPy": scipy.__version__,
              "openpyxl_readonly": openpyxl.__version__, "Matplotlib": matplotlib.__version__,
              "platform": platform.platform()})
    print(f"Prepared {len(payload['rows'])} workbook rows, {len(table)} Table-5 rows and 4 SVG figures.")


if __name__ == "__main__":
    main()
