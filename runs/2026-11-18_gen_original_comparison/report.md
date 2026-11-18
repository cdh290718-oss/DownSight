# Gen 原模型与三种微调模型

在此前固定的 Gen 20 对、40 张图上，补跑原始 Sapiens2-1B 的全二维 D 拟合。原始二维预测复用 [此前推理](../2026-11-17_gen_sapiens_compare/evidence/summary.json)，图像与前轮一致；同一 Head 位置和朝向、双目外参、画内参考掩码、均值体型、损失和 20000 步预算与三组微调对照一致。拟合成功 20/20 对，收敛 0/20；见 [3d_results.json](evidence/3d_results.json) 与 [summary.json](evidence/summary.json)。

| 模型 | 对 DFM 2D 平均 px | 距参考 ≤20 px | 对 DFM 3D 非 PA mm | 对 DFM 3D PA mm |
|---|---:|---:|---:|---:|
| 原模型 | 37.20 | 121 / 424 | 126.99 | 93.58 |
| RW500 | **34.57** | **139 / 424** | 118.81 | **86.42** |
| RW5000 | 35.01 | 90 / 424 | 115.18 | 86.85 |
| RW20000 | 36.67 | 87 / 424 | **114.11** | 88.57 |

微调三组证据见 [前轮 comparison.json](../2026-11-18_gen_three_models/evidence/comparison.json)。DFM 身体和头部为模型参考，不是人工真值。右肩参考均画外，鼻点无参考；二维只统计参考画内点但画内不意味着可见。三维评价 12 点，PA 是每对做相似变换；只有同一人一条记录，不能声称完整 Gen benchmark。20 对中所有模型均有可行结果但未通过收敛标准。

## Head 朝向来源

20 个时刻中，10 个读取此前网络的 canonical 原始参数，10 个对 cam3 图像重新运行同一 EgoSMPLX 网络获得原始参数；由网络的 global_orient_network 和 body_pose 沿 SMPL-X 根→脊柱→颈→头旋转链计算头部全局朝向。记录见 [前轮 preparation.json](../2026-11-18_gen_three_models/evidence/preparation.json) 和 [prepare_fit.py](../2026-11-18_gen_three_models/code/prepare_fit.py)。它是模型估计不是标注真值，四组共享该朝向。Head 位置取 DFM，双目外参使用 DFM 身体参考配准，DFM 画内掩码也用于拟合，存在参考信息参与，不能解读为独立自动评估。

执行基线 f348a47，工作树含本轮新增及既有未跟踪内容，原始数据、权重和完整坐标留在本地；代码、数值证据及路径脱敏后发布。

## 推送前自检

工作区、暂存、待推增量和远端已核对，增量不含图片、数据集、视频、权重、NPY/NPZ、完整输出、凭据、.env、完整 provenance 和 patch；单文件小于 50 MB，开发机路径已脱敏。见 [prepush.json](evidence/prepush.json)，历史未审计。
