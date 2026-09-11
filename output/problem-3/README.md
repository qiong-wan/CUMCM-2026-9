# 第三问输出目录索引

全部文件的具体职责、案例参数和生成关系见[文件分类与用途](文件分类与用途.md)，复现命令见[源目录 README](../../src/problem-3/README.md)。成果文档、图像及计算证据统一保存在本目录。

| 类别 | 文件 | 用途 |
|---|---|---|
| 正式结论 | [结果与验证.md](结果与验证.md) | 时长、全域达标证据、数值误差、环境敏感性和模型限制 |
| 模型说明 | [模型与算法说明.md](模型与算法说明.md) | 连续编号公式、完整推导、参数、假设及求解算法 |
| 题目工作簿 | [result3.xlsx](result3.xlsx) | 单 Sheet1 水分浓度表，末尾含实际结束时刻 |
| 表 5 | [table5.md](table5.md)、[table5_moisture.csv](table5_moisture.csv) | 可阅读表格及完整精度数据 |
| 完整采样数据 | `moisture_60s_full_precision.csv` | 工作簿对应的完整精度水分数据，末行允许非 60 s 整数倍 |
| 正式图表 | `drying_history.svg`、`radial_profiles.svg`、`temperature_and_environment.svg`、`convergence.svg` | 过程、径向分布、温湿边界和收敛证据 |
| 正式主解 | `production.npz`、`production.json`、`production_event_seed.npz` | 未舍入数值场、求解摘要和末期事件局部重算起点 |
| 必需对照计算 | `space400` 至 `space12800` 六组、`time12800quarter`、`mean12800` 对应的三类文件 | 空间加密、时间加密和边界敏感性；具体设置见完整分类表 |
| 验证记录 | `verification.json`、`workbook_verification.json`、`visual_verification.md` | 数值、逐格一致性和视觉检查 |
| 审计与复现证据 | `input_audit.json`、`protected_hashes_*.json`、`verification_initial_scope_failure.json`、`run_environment.json`、两个对照场 NPZ | 输入与保护检查、失败历史、环境和比较快照；详见完整分类表 |
| 派生与导出数据 | `observations_readonly_copy.csv`、`derived_boundaries.csv`、`workbook_payload.json` | 观测副本、假设边界、工作簿生成载荷 |
| 预览与简要诊断 | `previews/`、`artifact_inspection.txt` | 已归档的 7 张图表和工作簿核查预览、简要结构检查 |

已删除两组不参与正式流程的早期试算、题目临时截图、冗长 NDJSON 导出诊断及旧字体缓存；保留全部必要案例和审计记录。详细清理范围见文件分类文档。

正式主解为 `production`。`space12800` 使用较粗内部时间步，不可混入表 5 或工作簿。原始输入保留在项目 `data/`，本目录的观测 CSV 是派生副本。

验证状态必须按范围阅读：数值验收 PASS 不抹去完整保护文件检查中记录的外部第二问文件变化，也不代表 SKIPPED 项已经执行。详细说明保留在结果报告和验证 JSON 中。
