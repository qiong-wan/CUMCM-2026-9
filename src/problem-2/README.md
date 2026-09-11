# 问题 2：药材整个烘干过程（预热平衡 + 恒温干燥）

本目录实现问题 2 的耦合传热传质模型：半径 $R=2$ cm、长 $L=25$ cm 的圆柱药材在
热风烘干过程中的温度场 $T(r,t)$ 与干基含水率 $C(r,t)$。

## 依赖

- Python 3.11+
- `numpy`, `scipy`, `openpyxl`

## 运行

```bash
# 生产运行 + 网格/时间收敛 + 独立差分交叉验证（生成全部交付物）
python src/problem-2/solve_problem2.py

# 额外运行完整 2~3 天烘干过程并写入验证报告
python src/problem-2/solve_problem2.py --full

# 只生成结果文件、跳过较慢的验证（约 1 分钟）
python src/problem-2/solve_problem2.py --skip-verify
```

## 输出

| 文件 | 内容 |
|---|---|
| `output/problem-2/result2.xlsx` | 工作表 `温度`、`水分浓度`，A1 为 `时间\到药材中心的距离`，A 列为 1~10800 s，首行 0~2.0 cm（步长 0.1 cm），共 10801 行 × 22 列 |
| `output/problem-2/table3_temperature.csv` | 表 3：3 h 内每 0.5 h、半径 0/0.5/1/1.5/2 cm 的温度 |
| `output/problem-2/table4_moisture.csv` | 表 4：同上位置的水分浓度 |
| `output/problem-2/tables3_4.md` | 表 3、表 4 的 Markdown 版本 |
| `output/problem-2/verification.txt` | 网格/时间收敛、全局能量与水分守恒审计、独立差分交叉验证、局限性说明 |

## 模型与算法要点

- 控制方程：径向轴对称非稳态导热 + 非线性水分扩散，物性采用附录 3 的经验公式
  $\rho=650+128C$、$c_p=1450+2736C/(C+1)$、$k=0.21+0.38C/(C+1)$、
  $D=2.4\times10^{-3}e^{-0.45/C}e^{-3850/T}$（$T$ 为 K）。
- 边界：$r=0$ 对称；$r=R$ 为对流换热与对流传质（Robin）边界，$h=25$ W/(m²·K)、
  $h_m=8\times10^{-7}$ m/s（取自附录 2，附录 3 未给出）。
- 数值：节点中心有限体积、后向 Euler、Picard 迭代求解耦合非线性；生产网格
  $N=3200$（$\Delta r=0.00625$ mm），每个输出秒 4 个隐式子步。全场上限
  $\max|\Delta C|=4.88\times10^{-5}$（四位小数交付目标 $<5\times10^{-5}$）。
- 阶段切换：`附件1` 覆盖 0~14400 s，视为预热平衡阶段；其后环境温度/水分浓度冻结为
  其最后一个观测值（恒温干燥阶段）。报告的前 3 h 完全位于预热阶段内。
- 不收缩：问题 2 保持 $R=2$ cm；`附件2` 仅用于问题 4。
- 不含汽化潜热：题面未给出潜热或热湿耦合系数，故能量方程不含蒸发项；由此产生的能量
  不一致在 `verification.txt` 中定量给出。

详见 `问题2_模型与算法说明.md`。
