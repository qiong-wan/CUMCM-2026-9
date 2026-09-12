# 第四问派生数据说明

原始文件仅从 `Solution/data` 和 `Solution/problem.pdf` 只读读取。当前目录不存放修正后的原始数据，不包含删行、插补、平滑或拟合结果。

| 文件 | 字段与生成方法 |
|---|---|
| ambient_observed.csv | time_s、temperature_C、moisture_kg_kg；附件 1 全部原始数值的独立 CSV 副本 |
| radius_observed_SI.csv | time_s、radius_m；附件 2 半径由 cm 乘 0.01 转 m，时间与观测顺序不变 |
| radius_history.csv | time_s、radius_m、radius_cm；正式交付各时刻的分段线性实测半径，末行为同一严格结束时刻 |

原始输入审计和 SHA-256 见 `../review/input_audit.json`。主案例与全部加密、对照数据位于 `../cases`。每个案例目录的名称含完整输入、求解脚本、配置、依赖与数据结构的指纹摘要。

每个 `fields_*.npz` 包含升序 `times`，单位 s，及 `states`，形状为保存时刻数、2、径向节点数；第二维依次是局部温度 °C、干基含水率 kg/kg，采用未舍入 float64。网格 `x` 为 0 至 1 的随体坐标，实际距离为当时半径乘 x，不能直接标为 cm。

`summary.npz` 保存 `x`、圆柱权重 `weights`、全部保存时刻的 `history`、启动和每 6 h 加末时刻的 `check_times/check_states`、`final_state`，以及事件前完整未舍入 `event_seed/event_t0/event_width`。未发生达标事件时，以 metadata 中事件为空为准，不能把事件占位初值误认为真实事件。

`history` 的列名与单位在 metadata 的 `history_columns` 中完整保存：时间、半径、中心/表面双场、全域最大水分及其 x 位置、干物质加权平均、经验显热储存累计 J、边界热累计 J、独立水分通量累计 kg/kg、水分平衡差、表面 Robin 残差、轴线梯度以及对应环境。

全域最大与径向趋势在每个内部接受步检查；完整双场仅在 60 s、启动诊断及事件终点保存，避免保存全部内部小步造成不必要的数据体积。每个分块文件的 SHA-256 与参数均在 metadata 中，可逐块复核。逐格 Excel 与表 6 校验使用这些未舍入场，不由显示值反算。

案例完成不等于该案例通过全部精度验证。只有 `../review/verification.json` 与 `../delivery_manifest.json` 同时匹配的正式批次可作为已验收交付。失败试算与粗网格 FAIL 均保留原证据。
