# 第三问复现与交付

全部计算从均匀初值 28°C、2.55 kg/kg 重新开始，附录 3 局部物性、固定半径 0.02 m、长度 0.25 m。主方案在 14400 s 后同时固定末次温度和环境水分浓度，均值情景仅替换观测区间之后的边界；另有一个独立的平台期随机波动集合（AR(1)）用于量化环境波动对临界时间的影响。详细推导见[模型与算法说明](模型与算法说明.md)（连续编号公式 1–74），计算结果和全部证据见 `output/problem-3/delivery/结果与验证.md`。

## 运行环境

Python 3.11+，NumPy、SciPy、openpyxl（只读输入与复核）、Matplotlib；具体执行版本记录在 `output/problem-3/audit/run_environment.json`。工作簿由 Node.js 的 `@oai/artifact-tool` 从原模板导入、扩展、导出；`node_modules` 联接仅位于第三问目录，不修改依赖目录。

## 运行

在 `D:/CUMCM/Solution` 运行完整流程（单文件入口，推荐）：

```powershell
python -B -X utf8 src/problem-3/solve_problem3.py --all
```

可拆开执行（不要运行前两问入口）：

```powershell
python -B -X utf8 src/problem-3/solve_problem3.py --audit
python -B -X utf8 src/problem-3/solve_problem3.py --case production --n 12800 --factor 0.5
python -B -X utf8 src/problem-3/solve_problem3.py --verify
python -B -X utf8 src/problem-3/solve_problem3.py --deliver
python -B -X utf8 src/problem-3/generate_figures.py
node src/problem-3/build_workbook.mjs
python -B -X utf8 src/problem-3/solve_problem3.py --workbook
```

试运行加速（`--all` 的开关，按依赖级联）：

```powershell
# 只求解主方案并绘图，跳过收敛/敏感性/验证/交付/工作簿
python -B -X utf8 src/problem-3/solve_problem3.py --all --no-convergence --no-sensitivity
# 粗网格快速冒烟（不用于交付）：N=400、倍率 1
python -B -X utf8 src/problem-3/solve_problem3.py --all --no-convergence --no-sensitivity --prod-n 400 --prod-factor 1
# 已有多案例文件时，只重跑验证/交付/绘图/工作簿
python -B -X utf8 src/problem-3/solve_problem3.py --all --resume
# 以随机平台（fluct）运行完整流程；同一随机种子用于全部收敛案例
python -B -X utf8 src/problem-3/solve_problem3.py --all --prod-mode fluct --prod-seed 0
```

| 开关 | 作用 |
|---|---|
| `--no-convergence` | 跳过空间/时间加密案例（最耗时），并级联关闭验证、交付、工作簿 |
| `--no-sensitivity` | 跳过均值边界案例 |
| `--no-verify` / `--no-deliver` / `--no-figures` / `--no-workbook` | 逐级关闭验证、交付、绘图、工作簿 |
| `--prod-n` / `--prod-factor` | 试运行主解的网格数与时间倍率（默认 12800 / 0.5） |
| `--prod-mode` / `--prod-seed` | 主解族平台模型（`last`/`mean30`/`fluct`）与随机种子 |
| `--resume` | 复用已存在的命名案例，只重跑后续阶段 |

所有求解阶段都输出进度条（`t/72h`、步数、耗时、ETA）；重定向到文件时按行输出。

注意：粗网格试运行的物理阈值（如边界重构）通常不达标，这是预期的；试运行结果不能作为交付。

平台期随机波动集合：

```powershell
python -B -X utf8 src/problem-3/ensemble_problem3.py --n 400 --factor 1 --members 8
python -B -X utf8 src/problem-3/ensemble_problem3.py --n 12800 --factor 0.5 --members 16 --tau 1800
```

单案例随机边界（用于检查模型本身）：

```powershell
python -B -X utf8 src/problem-3/solve_problem3.py --case fluct_demo --n 800 --factor 1 --mode fluct --seed 3
```

## 文件职责

| 文件 | 职责 |
|---|---|
| `solve_problem3.py` | **单一入口**：只读审计、局部物性、圆柱有限体积、SDIRK2/Picard、独立累计通量与严格终点；环境尾段支持 `last`/`mean30`/`fluct`；内置验证（`--verify`）、交付（`--deliver`）、工作簿重读（`--workbook`）与串联（`--all`）；不含 Matplotlib |
| `generate_figures.py` | **独立绘图**：从已保存案例与 `verification.json` 生成 4 幅主图（SVG+PNG）到 `output/problem-3/figures`；无验证数据时仍绘 3 幅（`--require-full` 强制完整）；若存在随机集合数据，再为每个种子生成一幅 `temperature_and_environment_seed{N}`（空气温度/水分，`--no-stochastic` 可关闭） |
| `ensemble_problem3.py` | **随机集合**：平台期 AR(1) 波动的 Monte Carlo，统计临界时间分布并写入 `output/problem-3/stochastic` |
| `build_workbook.mjs` | 从 `delivery/workbook_payload.json` 导出 `delivery/result3.xlsx` 并保留 Sheet1/A1、21 个距离列与四位小数显示 |
| `模型与算法说明.md` | 连续编号公式 1–74、完整模型、推导、参数、假设、随机平台模型与指标 |

## 输出目录结构

`output/problem-3/` 按用途分类，脚本读写同一布局：

```
audit/        input_audit.json, observations_readonly_copy.csv,
              protected_hashes_before/after.json, run_environment.json
cases/        {name}.npz, {name}.json, {name}_event_seed.npz（space*/production/
              time12800quarter/mean12800 及随机单案例）
validation/   verification.json, workbook_verification.json,
              near_end_common_fields.npz, problem2_comparison_snapshot.npz
delivery/     result3.xlsx, workbook_payload.json, table5.md, table5_moisture.csv,
              moisture_60s_full_precision.csv, derived_boundaries.csv,
              结果与验证.md, artifact_inspection.txt
figures/      drying_history / radial_profiles / temperature_and_environment /
              convergence 的 .svg 与 .png
stochastic/   data/, ensemble/, figures/, reports/（平台期随机模型）
```

## 主要交付文件

- `delivery/result3.xlsx`：唯一提交工作簿，仅水分浓度 Sheet1；单位秒、cm、kg/kg；末尾含非规则实际结束行。
- `delivery/table5.md`、`delivery/table5_moisture.csv`：每 6 h 加结束行，后者保留完整精度。
- `delivery/结果与验证.md`、`validation/verification.json`：结论、误差、边界敏感性与实际执行状态。
- `validation/workbook_verification.json`：全部 Excel 格值/格式/时间/表头和表 5 一致性复核。
- `cases/production.npz`、`cases/production_event_seed.npz`：主解未舍入状态与事件前完整状态。
- `delivery/derived_boundaries.csv`、`audit/input_audit.json`：环境派生结果和原始输入审计。
- `figures/*.svg`、`figures/*.png`：科学图表；其中 `temperature_and_environment_seed{N}` 为各随机种子的平台期空气温度/水分随时间曲线（观测段 0–4 h 加 AR(1) 波动段）。
- `stochastic/`：平台期随机波动的参数、逐实现结果、统计、图与报告。

原始题目、附件、模板、前两问、公共模块不写回。`production` 对应正式主解，`space12800` 是相同空间网格但时间步较粗的研究案例，两者不能混用。

## NPZ 数组约定

`times` 含初值、每 60 s 及实际结束时刻；`outputs` 形状为时间数×2×21，第二维依次为摄氏温度、干基含水率。导出时去除初值行。`full_times/full_states` 保存启动、论文时刻、每 6 h 和结束的 N+1 节点全场。`final_state` 为同一结束全场；`r` 单位 m。

`history` 各列依次为：时间 s；全域最大含水率；最大位置 m；体积加权平均含水率；全域含水率积分首末差；累计变热容储热；累计边界热输入；累计边界水分输入；热 Robin 重构残差；湿 Robin 重构残差；中心温度梯度；中心含水率梯度。几何积分除以公共因子 πL，真实热量需乘回 πL；未指定一致干物质参考密度时不把有效含水率积分换算为确定水质量。

数值 PASS 表示执行了所列检查并达到给定数值目标，不表示经验模型已通过真实药材实验验证；未执行项明确保留 SKIPPED。
