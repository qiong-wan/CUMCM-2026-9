# 第三问输出目录

参照问题 1、2，结果表留在根目录，图像归 image/，检查证据归 review/；完整数值状态与派生数据分别归 cases/、data/。

| 位置 | 内容 |
|---|---|
| [result3.xlsx](result3.xlsx) | 正式水分浓度工作簿，仅 Sheet1 |
| [table5.md](table5.md)、[table5_moisture.csv](table5_moisture.csv) | 表 5 阅读版与完整精度数据 |
| [moisture_60s_full_precision.csv](moisture_60s_full_precision.csv) | 21 个距离列的完整精度交付数据，末尾保留实际结束时刻 |
| [结果与验证.md](结果与验证.md) | 已有计算结论、环境假设、误差与验证范围 |
| `image/` | 中文科学图 SVG/PNG；previews/ 放工作簿预览和科学图 JPEG |
| `cases/` | production、空间/时间加密、均值环境等 9 个案例的 27 个数值文件 |
| `data/` | 审计结果、观测副本、派生边界、工作簿载荷与原运行环境 |
| `review/` | 数值验证、代码审查原始测试、哈希、整理检查和工作簿诊断 |
| `stochastic/` | 随机试算的数据和报告（实际运行后生成）；其图像归 image/stochastic/ |

本次只调整图中文字、输出位置和读取路径，现有数值结果保持不变。原审查问题没有统一修复，完整求解与随机集合没有重跑。

代码、模型推导与审查说明均在 src：

- [完整文件分类与用途](../../src/problem-3/文件分类与用途.md)
- [运行及重绘命令](../../src/problem-3/README.md)
- [模型与算法说明](../../src/problem-3/模型与算法说明.md)
- [代码审查报告](../../src/problem-3/问题3_代码审查报告_GPT.md)
- [本次整理验证说明](../../src/problem-3/review/中文图与目录整理验证.md)

直接查看四幅中文科学图：[干燥历程](image/drying_history.svg)、[径向分布](image/radial_profiles.svg)、[温度与环境](image/temperature_and_environment.svg)、[收敛](image/convergence.svg)。
