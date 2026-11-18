# 真实样本001：四模型 OBJ 与自动点二维 RMSE

**[下载四模型 OBJ 包](https://github.com/cdh290718-oss/DownSight/releases/download/real001-obj-rmse-20261118/real001_t029p0_four_models_OBJ_cam3_cam4.zip)**

包内 cam3_cv 是四模型的主要网格；cam4_cv 是相同四个人体变换到另一相机后的坐标版本，共 8 个 OBJ，不是 8 次拟合。文件名包含样本、29秒、检测器和坐标系。单位米，CV 坐标 +X 向右、+Y 向下、+Z 向前；不做中心化、缩放或额外翻转。

OBJ 保存已变形的顶点与三角面，不保存 SMPL-X 形状/姿态参数或骨骼绑定。需要修改姿态时请使用[上一轮 NPZ/JSON 参数包](https://github.com/cdh290718-oss/DownSight/releases/tag/real001-goldend2d-four-models-20261118)。网格继承上一轮固定手型，没有新增 MANO 手部拼接。

## RMSE 定义与范围

使用原图像素坐标下的径向二维 RMSE：

`RMSE = sqrt(sum((u_mesh_joint-u_auto)^2 + (v_mesh_joint-v_auto)^2) / N)`

范围是左右肩、肘、腕、髋、膝、踝共12个身体关节，不含手指，不能衡量手掌翻转或 MANO 拼接质量。这里投影的是从同一套 SMPL-X 参数还原的身体关节，不是寻找网格最近顶点。双目合并汇总两目的平方距离再开根号，不直接平均两目的 RMSE。若用 x/y 分量作为 2N 个样本，结果会再除以 sqrt(2)；evidence/rmse.json 同时保留该定义，正文统一用径向 RMSE。

**每个模型对照它自身的自动检测点，因此不是共同独立真值，不能用此表评判检测器或真实三维准确率。**

## 全部自动点：每目12点，合并24点

包含之前未进入共同拟合掩码的 cam3 左肩、cam4 右肩；这些自动预测可在画面外，仍参与本表，不按分数或误差删除。

| 检测器 | cam3 RMSE px | cam4 RMSE px | 双目合并 RMSE px |
|---|---:|---:|---:|
| 原模型 | 16.36 | 25.57 | 21.46 |
| 500 张 | 9.94 | 22.44 | 17.35 |
| 5000 张 | 10.20 | 23.30 | 17.99 |
| 20000 张 | 11.55 | 28.81 | 21.95 |

## 共同拟合掩码：每目11点，合并22点

该表仅评估上一轮实际使用的同一组观测掩码；不是重新挑选小误差点。

| 检测器 | cam3 RMSE px | cam4 RMSE px | 双目合并 RMSE px |
|---|---:|---:|---:|
| 原模型 | 15.81 | 15.05 | 15.44 |
| 500 张 | 8.52 | 7.74 | 8.14 |
| 5000 张 | 9.81 | 9.25 | 9.53 |
| 20000 张 | 11.41 | 10.71 | 11.06 |

两种口径下，500 张模型的拟合结果与自身自动点距离最小。这是本次单个真实双目样本上的自一致性，不代表其检测最准；不同模型的目标点本来就不同。

## 导出与复现

执行基线 `60f2c92c282d595cfadc7a5a424a158ae1c4e10c`，上轮 review 待审核。本轮代码运行时未提交，原有工作树改动保留。读取上一轮四份发布参数，逐个核对其 SHA-256，再用相同 SMPL-X 层重建。相机、Head、体型、手型及拟合状态均未修改，也未重跑检测或优化。上一轮四组均未通过末段稳定性诊断，本次导出不会改变该限制。

OBJ 使用一基三角面索引，保留完整表面，无贴图。每个文件写出后重新读取，检查顶点数、面数、索引范围、拓扑一致及顶点精度；相机标定与变换、文件 SHA-256 放在 ZIP 中。未包含训练权重、SMPL-X 模型权重、原图、完整原始流水线结果。

逐点二维残差：[point_residuals.json](evidence/point_residuals.json)；完整 RMSE 与分母：[rmse.json](evidence/rmse.json)；OBJ 校验与哈希：[exports.json](evidence/exports.json)；运行记录：[execution.json](evidence/execution.json)。原始坐标和完整本地路径保留本机。

## 发布自检

用户明确请求 OBJ 打包并沿用 GitHub 下载交付。拟合网格作为独立 Release 附件，Git 只保存代码、报告、精简证据；没有 NPY/NPZ/OBJ/ZIP、权重、数据集、视频、凭据、.env 或完整 provenance/patch 入库。单文件小于 50 MB。附件内容/哈希自检见 evidence/attachment_audit.json，工作区、暂存、待推提交和远端检查见 evidence/prepush.json；历史不在本轮审计范围。
