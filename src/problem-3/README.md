# 第三问复现与交付

目录导航：先看[文件分类与用途](文件分类与用途.md)，模型与代码问题见[代码审查报告](问题3_代码审查报告_GPT.md)；只查看计算成果可从[输出目录索引](../../output/problem-3/README.md)进入。

全部计算从均匀初值 28°C、2.55 kg/kg 重新开始，附录 3 局部物性、固定半径 0.02 m、长度 0.25 m。主方案在 14400 s 后同时固定末次温度和环境水分浓度，均值情景仅替换观测区间之后的边界。详细推导见[模型与算法说明](模型与算法说明.md)，计算结果和全部证据见 `../../output/problem-3/结果与验证.md`。

当前新增的合并入口及随机模型仍存在审查报告列出的未解决问题，下列原流程命令不代表新版本已完成整套验收。本次仅统一图中文字和输出路径；审查报告保留原审查时的证据，不表示其中其余问题已经修复。

## 运行环境

Python 3.11+，NumPy、SciPy、openpyxl（只读输入与复核）、Matplotlib；具体执行版本记录在 `output/problem-3/data/run_environment.json`。工作簿由 Node.js 的 `@oai/artifact-tool` 从原模板导入、扩展、导出；本机已经建立仅位于第三问目录中的 `node_modules` 联接，指向 Codex 提供的运行库，不修改依赖目录。

在 `D:/CUMCM/Solution` 运行完整流程：

```powershell
python -B -X utf8 src/problem-3/run_all.py
```

当前 Codex 附带的 Python 可显式调用：

```powershell
& 'C:/Users/lenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -B -X utf8 src/problem-3/run_all.py
```

流程会重算全部必要空间序列、时间序列和均值环境情景，依次执行验证、表 5/图表准备、XLSX 导出及全量重读核查。任何必要验证失败均返回非零状态。不是 72 h 到时即结束，事件之前未达标会自动继续延长。`--resume` 仅适用于已确认代码与输入不变的中断续跑，明确复用已有命名案例；默认不复用。

可拆开执行（不要运行前两问入口）：

```powershell
python -B -X utf8 src/problem-3/solve_problem3.py --audit
python -B -X utf8 src/problem-3/solve_problem3.py --case production --n 12800 --factor 0.5
python -B -X utf8 src/problem-3/verify_problem3.py
python -B -X utf8 src/problem-3/deliver_problem3.py
node src/problem-3/build_workbook.mjs
python -B -X utf8 src/problem-3/verify_problem3.py --workbook
```

其中验证入口要求其余必要对照已经完成，不能只跑单个 production 后跳过收敛验证。若单个阶段在当前 Python 环境缺包，使用上述附带环境。工作簿工具支持通过 `NODE_EXE` 和 `ARTIFACT_NODE_MODULES` 指定已有安装位置，不自动联网安装依赖。

## 文件职责

| 文件 | 职责 |
|---|---|
| `solve_problem3.py` | 只读审计、局部物性、圆柱有限体积、SDIRK2/Picard、独立累计通量与严格终点 |
| `verify_problem3.py` | 空间、时间、事件、边界、短时独立 BDF、故障反例、旧结果衔接、保护文件哈希、XLSX 全量核查 |
| `deliver_problem3.py` | 从通过验收的 production 解生成表 5、未舍入载荷、4 幅科学图表、计算结果报告 |
| `build_workbook.mjs` | 保留 Sheet1/A1，扩展为 21 个距离列，保持数值并设置四位小数显示 |
| `run_all.py` | 串联上述流程，失败立即退出 |
| `generate_figures.py` | 独立读取保存案例，重新生成中文图像 |
| `ensemble_problem3.py` | 随机环境试算及中文样本图，保留现有计算逻辑 |
| `plot_style.py` | 统一中文字体、负号及 SVG 字形嵌入 |
| `review/verify_presentation.py` | 核心代码与数值文件哈希、绘图数据和中文字形回归检查 |

本目录按前两问约定保存代码、模型推导、审查文档、运行说明、分类索引及必要配置。[视觉审查记录](review/visual_verification.md)与审查脚本放在 `review/`。计算结果、原始测试记录和图像放在 `output/problem-3/`；科学图统一位于其 `image/` 子目录，工作簿预览及直接由 Figure 生成的中文 JPEG 核查图位于 `image/previews/`。

## 主要交付文件

- `output/problem-3/result3.xlsx`：唯一的提交工作簿，仅水分浓度 Sheet1；单位秒、cm、kg/kg；末尾包含明确的非规则实际结束行。
- `output/problem-3/table5.md`、`table5_moisture.csv`：每 6 h 加结束行，后者保留完整精度。
- `output/problem-3/结果与验证.md`、`output/problem-3/review/verification.json`：结论、误差、边界敏感性与实际执行状态。
- `output/problem-3/review/workbook_verification.json`：全部 Excel 格值/格式/时间/表头和表 5 一致性复核。
- `output/problem-3/cases/production.npz`：同一主解的未舍入状态、输出数组、完整检查截面和累计诊断。
- `output/problem-3/cases/production_event_seed.npz`：事件前未舍入完整状态，支持局部重算。
- `output/problem-3/data/derived_boundaries.csv`、`output/problem-3/data/input_audit.json`：环境派生结果和原始输入审计。
- `output/problem-3/image/` 中的 `drying_history`、`radial_profiles`、`temperature_and_environment`、`convergence`（SVG/PNG）：科学图表。

`output/problem-3/cases/` 下其余必要的 `space*.npz/json`、`time12800quarter*.npz/json`、`mean12800*.npz/json` 为独立运行证据。`output/problem-3/image/previews/` 保存此前视觉核查使用的图像，后续导出也写入该目录；没有重新读取 output 下的 PNG。已清理不参与正式流程的 `time6400half`、`mean6400` 早期案例、题目临时截图和冗余诊断缓存。原始题目、附件、模板、前两问、公共模块不写回。`production` 对应正式主解，`space12800` 是相同空间网格但时间步较粗的研究案例，两者不能混用。

## 仅重新绘图与整理验证

```powershell
python -B -X utf8 src/problem-3/generate_figures.py --require-full --no-stochastic
python -B -X utf8 src/problem-3/review/verify_presentation.py
python -B -X utf8 src/problem-3/review/verify_layout.py
```

独立绘图从 `cases/`、`data/`、`review/` 读取已有解和验证记录，不重算烘干过程。省略 `--no-stochastic` 时沿用原有规则，仅在实际随机数据存在时追加每个种子的图。中文字体依次查找 Microsoft YaHei、SimHei、Noto Sans CJK SC、Source Han Sans SC；缺少可用字体时明确报错。SVG 嵌入字形，换机查看不依赖本机字体安装。

本次检查基线位于 `output/problem-3/review/presentation_before.json`；`--capture` 仅用于建立新的整理前基线，已有基线会拒绝覆盖。具体执行范围见 [中文图与目录整理验证](review/中文图与目录整理验证.md)。

## NPZ 数组约定

`times` 含初值、每 60 s 及实际结束时刻；`outputs` 形状为时间数×2×21，第二维依次为摄氏温度、干基含水率。导出时去除初值行。`full_times/full_states` 保存启动、论文时刻、每 6 h 和结束的 N+1 节点全场。`final_state` 为同一结束全场；`r` 单位 m。

`history` 各列依次为：时间 s；全域最大含水率；最大位置 m；体积加权平均含水率；全域含水率积分首末差；累计变热容储热；累计边界热输入；累计边界水分输入；热 Robin 重构残差；湿 Robin 重构残差；中心温度梯度；中心含水率梯度。几何积分除以公共因子 πL，真实热量需乘回 πL；未指定一致干物质参考密度时不把有效含水率积分换算为确定水质量。

数值 PASS 表示执行了所列检查并达到给定数值目标，不表示经验模型已通过真实药材实验验证；未执行项明确保留 SKIPPED。
