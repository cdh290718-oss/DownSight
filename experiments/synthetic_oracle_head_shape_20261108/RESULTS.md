# 结果、对照与证据

[返回实验入口](README.md)。所有数值来自源实验，未因本轮整理重新拟合。精确值和逐关节结果见 [evidence/results.json](evidence/results.json)，完整原始数值证据见[源 metrics](../../runs/2026-11-08_synthetic-head-robustness/evidence/metrics.json)。

## 主结果与同初始化对照

单位 mm。身体为 25 点绝对平均误差；腕/肘分别对双侧求平均；手指为 40 点平均。中性体型网格采用共同零姿态、各自骨盆居中，不做尺度/旋转对齐。

| 条件 | 身体 | 骨盆 | 腕 | 肘 | 手指 | 中性体型网格 |
|---|---:|---:|---:|---:|---:|---:|
| legacy_oracle_left | 57.99 | 62.90 | 6.88 | 42.00 | 30.96 | 28.06 |
| oracle_left | 35.51 | 42.14 | 6.14 | 22.68 | 20.97 | 5.68 |
| legacy_none_left | 64.75 | 52.59 | 6.47 | 30.18 | 30.47 | 14.66 |
| none_left | 44.85 | 45.28 | 6.00 | 23.19 | 21.41 | 5.35 |
| oracle_right | 39.71 | 43.69 | 5.71 | 22.55 | 22.57 | 13.48 |
| none_right | 44.62 | 47.92 | 5.42 | 21.05 | 22.90 | 6.72 |

`legacy_oracle_left` 与主组使用相同自动初始化、头部锚点、步数和原姿态范围；它保留旧二维目标、全部手点、自由十维 beta。两者差异是三维加权、手点集合和体型子空间这一整套方法，不能当作单变量消融。

`oracle_right` 是从右目网络预测重新初始化同一个双目拟合，不是另一个样本，也不是只用右图。主组仍固定为预先指定的 left，不按真值误差选优。

## 较早结果只作外部参考

| 条件 | 身体 mm | 中性体型网格 mm |
|---|---:|---:|
| B_previous | 69.48 | 38.30 |
| E_previous | 47.39 | 20.26 |
| D_oracle_shape | 36.93 | 0.00 |

B 固定网络体型，E 自由十维体型并有均值先验，D 固定真值体型；三者使用此前初始化与目标。主组身体误差低于表中 D，不能解释为自动体型比真值体型更准确。主组双肘仍比 D 差，具体见主组证据及源报告。

## 头部扰动敏感性

每种平移范数有四个预定测试，均加 3° 朝向扰动。下表范围是有限测试最小/最大值，不是统计区间。

| 平移范数 | 身体均值 [最小, 最大] mm | 腕均值 [最小, 最大] mm |
|---|---|---|
| 10 mm | 37.58 [36.91, 38.15] | 6.17 [6.13, 6.20] |
| 20 mm | 40.06 [38.71, 41.32] | 6.23 [6.12, 6.37] |

全部八次结果（含体型、手指和头部误差）保留在 [evidence/results.json](evidence/results.json) 的 `comparison_rows` 与[原报告](../../runs/2026-11-08_synthetic-head-robustness/report.md)。扰动向量见[原始预设](../../runs/2026-11-08_synthetic-head-robustness/evidence/perturbations.json)。

## 验证范围

源实验已完成几何导数、协方差正定、初始化重建、梯度有限性、参数重载、投影一致性、独立损失重算、同初值、子空间、输入隔离和两目共享坐标检查；见[方法检查](../../runs/2026-11-08_synthetic-head-robustness/evidence/method_tests.json)、[后验验证](../../runs/2026-11-08_synthetic-head-robustness/evidence/verification.json)、[执行源文件核对](../../runs/2026-11-08_synthetic-head-robustness/evidence/executed_source_check.json)。这些检查支持实现与记录一致，不能证明全局最优、真实标定可靠或跨人体泛化。
