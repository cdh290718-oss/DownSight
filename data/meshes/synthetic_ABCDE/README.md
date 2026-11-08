# 合成样本 A / B / C / D / E 网格

这里是同一合成姿态的六个实验结果，可用 MeshLab 的 **File → Import Mesh** 打开。所有文件都使用 **cam3（左目）CV 相机坐标系、米制单位**：X 向右、Y 向下、Z 向前。没有居中、缩放、地面校正或删除镜头后方顶点，便于直接叠加比较。

| OBJ | 内容 |
|---|---|
| [A_cam3.obj](A_cam3.obj) | A：只使用 cam3 观测的独立拟合 |
| [A_cam4.obj](A_cam4.obj) | A：只使用 cam4 观测的独立拟合；**坐标仍是 cam3 系** |
| [B.obj](B.obj) | B：两目共享人体，固定网络预测体型 |
| [C.obj](C.obj) | C：相对 B 放宽姿态硬范围 |
| [D.obj](D.obj) | D：相对 B 固定为真值体型，仅作诊断参考 |
| [E.obj](E.obj) | E：带先验优化体型，使用预先指定的主权重 **0.001** |

全部采用已报告的主初始化（cam3 网络预测），不是按真值误差挑选的结果。B/C/D/E 各自只有一个共享人体，所以无需分别导出两目版本。E 不是强先验 0.01 对照。

这些是 SMPL-X 身体网格，包含模型自带的手部；**没有进行 WiLoR MANO 手替换和接缝融合**。每份 OBJ 含 10475 个顶点、20908 个三角面，无贴图和材质文件。

- [A–D 实验报告](../../../runs/2026-11-08_synthetic-stereo-abcd/report.md) · [A–D 投影](../../../runs/2026-11-08_synthetic-stereo-abcd/projection.md)
- [E 实验报告](../../../runs/2026-11-08_synthetic-shape-prior/report.md) · [E 投影](../../../runs/2026-11-08_synthetic-shape-prior/projection.md)
- [导出报告](../../../runs/2026-11-08_publish-synthetic-abcde/report.md) · [文件摘要及来源](../../../runs/2026-11-08_publish-synthetic-abcde/evidence/meshes.json)
