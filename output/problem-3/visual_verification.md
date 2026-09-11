# 图表与工作簿视觉核查

2026-09-11，原视觉核查实际查看了以下预览，当时位于 `src/problem-3/`。按用户后续目录整理要求，7 张预览已统一迁移至 `output/problem-3/previews/`，生成路径也已更新。下表保留原核查结果；此次文件整理未重新进行视觉核查（SKIPPED），未读取 output 下的 PNG。

| 预览 | 实际检查 | 结果 |
|---|---|---|
| `template_preview.png` | 原模板 Sheet1、A1、60/120/180 s 示意行与省略号 | PASS |
| `workbook_preview.png` | 最终 A1:V8，完整 21 个距离列、四位小数、行列对齐、表头文字 | PASS |
| `workbook_end_preview.png` | 最后 5 行，3430 个规则时刻之后的实际结束时间小数行，无文字截断 | PASS |
| `drying_history_qa.png` | 中心/平均/表面曲线、阈值、两个边界情景的不同终点、轴标题和图例 | PASS |
| `radial_profiles_qa.png` | 6 h 等径向截面、结束场与表面层局部放大，标签无重叠 | PASS |
| `temperature_and_environment_qa.png` | 材料温度与环境波动、观测终点及派生边界，标签无截断 | PASS |
| `convergence_qa.png` | 空间与时间加密分开展示，轴单位、刻度和标题完整 | PASS |

科学图的正式交付为同名 SVG，PNG 是绘制同一 Figure 时生成的核查副本。工作簿最终版本已在统一扩展行对齐后再次渲染和查看。数值真实性、严格阈值和逐格一致性由 `verification.json` 与 `workbook_verification.json` 单独验证，视觉检查不替代数值检查。
