# UnrealEgo-RW 测试集：10 对 × 4 模型 OBJ

**[下载全部 40 个 OBJ（ZIP，11.59 MiB）](https://github.com/cdh290718-oss/DownSight/releases/download/unrealego-rw-test10-four-models-obj-20261119/UnrealEgoRW_test10pairs_original_rw500_rw5000_rw20000_OBJ.zip)**

从上一轮冻结 100 对测试样本中固定抽取 10 对：S1 4 对，S9/S11 各 3 对，覆盖 10 条不同序列。按固定 SHA-256 排序取样，不依据误差、外观或收敛状态挑选。四个模型使用相同样本。每个样本文件夹有 `original`、`rw500`、`rw5000`、`rw20000` 四份网格；一对双目共同拟合一个人体，因此总数为 40 个 OBJ。

文件名格式：`编号_人物_序列_frame帧号_原评测pair编号__模型__leftcam_m.obj`。包内有 `index.csv`、`manifest.json`、`README_zh.md`、相机坐标变换和 `SHA256SUMS.txt`。

| 文件夹编号 | 人物 | 原评测样本 | 序列 | 帧号 |
|---|---|---|---|---:|
| 01 | S1 | pair007 | seq_1-2-29 | 0 |
| 02 | S1 | pair022 | seq_3-0-58 | 0 |
| 03 | S1 | pair026 | seq_3-1-18 | 0 |
| 04 | S1 | pair033 | seq_3-1-72 | 0 |
| 05 | S9 | pair038 | seq_1-0-31 | 0 |
| 06 | S9 | pair040 | seq_1-0-5 | 0 |
| 07 | S9 | pair047 | seq_1-1-0 | 220 |
| 08 | S11 | pair073 | seq_1-0-21 | 20 |
| 09 | S11 | pair084 | seq_1-1-11 | 0 |
| 10 | S11 | pair094 | seq_1-1-18 | 0 |

单位米，保持原左相机 CV 坐标（X 向右、Y 向下、Z 向前），未做 PA 对齐、居中、缩放或翻转。同一帧四模型可直接叠加；不同帧不是共同世界坐标。相机文件对应已有 1280×720 适配画布，原生 872×872 的缩放/偏移见包内说明。

直接从上一轮已评价参数重建，没有重新检测或优化；保留固定手姿态，**没有 MANO 手部拼接**。OBJ 含变形顶点与三角面，没有贴图、骨骼绑定或可编辑姿态系数。原实验的标注辅助和收敛限制仍适用，每份状态见索引。

## 验证与执行

执行基线 `34ff79ba1ec199db8a48803935a7f15e15810aa5`；工作树包含本轮新增代码及既有修改，既有文件改动保留。核对源参数与输入 SHA-256，按原评价的 SMPL-X 层及根旋转轴心变换重建。与原评价关节坐标最大差异 1.1920929e-07 m；OBJ 回读顶点最大差异 5.00000041e-10 m，拓扑、索引范围、40 个文件及 ZIP 全部校验通过。证据：[抽样](evidence/selection.json)、[导出](evidence/exports.json)、[附件自检](evidence/attachment_audit.json)、[执行](evidence/execution.json)。没有更改原 100 对的指标。

ZIP SHA-256：`a03646fa05b23a25eebdf584c21f83fc13a53d6d3313045bf665628bb28e90eb`。

## 发布自检

用户要求命名打包并沿用 GitHub 下载交付。拟合 OBJ 只作为独立 Release 附件，Git 仅保存代码、报告和精简脱敏证据；不包含权重、数据集、原图、视频、NPY/NPZ/OBJ/ZIP、完整原始输出、完整 provenance/patch、凭据或 .env。每文件小于 50 MB，机器路径替换为占位符。工作区、暂存、待推提交和远端增量审计见 [prepush.json](evidence/prepush.json)，不将本轮检查作为历史审计。

[原测试报告](../2026-11-18_rw20000_frozen200/report.md) · [黄金 D 方法和指标](../2026-11-18_rw_frozen200_golden_d/report.md)
