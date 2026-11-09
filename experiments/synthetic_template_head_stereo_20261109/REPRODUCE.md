# 来源、资产与复现条件

[返回入口](README.md)。本目录是已完成D组的归档入口，不是自带权重和输入的一键运行包。本次整理没有重新进行检测、三角化、拟合或评测。

## 固定版本和配置

- 源发布提交：`485fc71ca37d3c68a74bf99338fe247c197bf6eb`。
- 实际拟合基线HEAD：`6d5994401b33450ea3343d20f44f2cc54e9ac5d6`；拟合代码当时为新增工作树文件，源run已记录执行哈希并核对发布版。
- 主组：[case.json](case.json)，即 `case=D, shape_seed=mean, pose_seed=template`。其中 `seed=left` 是历史输出标签/公共cam3坐标约定，D不会据此读取左目网络预测。
- 主预算20000步，按本组训练目标选中第19960步；3400步以及其他预先声明预算保留在[结果证据](evidence/results.json)。没有按GT挑选检查点。
- [settings.json](settings.json) 是源配置的逐字节快照。源拟合器读取自己的run目录配置，修改这里的快照不会改变已归档方法。
- [manifest.json](manifest.json) 对实现、证据和图片给出固定提交链接、大小与SHA-256，对外部资源给出逻辑标识、哈希与访问要求；本目录没有另一份拟合实现。

## 权威代码入口

| 源文件 | 职责 |
|---|---|
| [fit.py](../../runs/2026-11-09_initialization-ablation/fit.py) | 共同目标、允许初值读取范围、优化、检查点和参数保存 |
| [initialization.py](../../runs/2026-11-09_initialization-ablation/initialization.py) | 共同体型子空间与Head对齐初始根位姿 |
| [prepare.py](../../runs/2026-11-09_initialization-ablation/prepare.py) | 为完整四组准备公共观测和分离的网络因素 |
| [run.py](../../runs/2026-11-09_initialization-ablation/run.py) | 四组运行、环境设置和退出状态；不是只跑D的入口 |
| [evaluate.py](../../runs/2026-11-09_initialization-ablation/evaluate.py)、[evaluate_shape.py](../../runs/2026-11-09_initialization-ablation/evaluate_shape.py)、[evaluate_mesh.py](../../runs/2026-11-09_initialization-ablation/evaluate_mesh.py) | 独立的关节、体型、网格与骨骼朝向评测 |
| [verify.py](../../runs/2026-11-09_initialization-ablation/verify.py) | 因素隔离、允许读取、相同子空间、固定参数、损失重算和数值一致性 |
| [stages.py](../../runs/2026-11-09_initialization-ablation/stages.py)、[evaluate_all.py](../../runs/2026-11-09_initialization-ablation/evaluate_all.py) | 预定预算及整批评测依赖 |

历史D拟合命令是：

```bash
"$BODY_PYTHON" runs/2026-11-09_initialization-ablation/fit.py --case D
```

这不是让你覆盖现有输出重跑。拟合器会拒绝已有的 `D_left` 目录，准备脚本也会拒绝已有输出根目录。实际重跑应在仓库内建立新的run/output位置，保留原结果；在复制的入口中只适配路径并记录修改及源哈希，不改已有源run。`common.py`、批量 `run.py` 和 `evaluate_all.py` 各有输出位置，不能只改其中一个就假定整批入口已适配。

需具备已有后端Python环境，设置 `PYTHONPATH=<REPO>`、`PYTHONDONTWRITEBYTECODE=1`；临时文件和缓存应指向新仓库输出目录，CPU/线程环境参考源 `run.py`。没有默认安装或下载步骤。

## D单独拟合与四组准备的区别

D拟合只需要公共 `inputs.json`、`triangulated.json`、`anchors/oracle.json`、私有 `model_paths.json` 和可访问的SMPL-X后端/模型资产。**不需要 `seeds/shape.json`、`seeds/pose.json` 或EgoSmplx raw预测文件。** 允许读取清单见[结果证据](evidence/results.json)中的 `D_verification`。

源 `prepare.py` 为四组设计，会提取A/B/C所需的网络因素，因此准备整批数据时仍读取原raw预测。若仅复现D，应由资产持有人提供经过哈希核对的上述公共输入到新的输出位置，不必运行四组网络因素提取；机器本地后端路径须按实际环境配置并记录新哈希。

源批量评测和验证依赖A/B/C/D及旧配置参考；只跑D之后不能原样宣称整批验证通过。应明确适配D专用评测依赖，或准备完整四组输出后使用原批量入口。本次归档没有开发或验证新的D专用runner。

## 外部资源

| 资源 | 获取与访问要求 |
|---|---|
| 代码、设置、精确数值、派生投影图 | 本仓库固定源run；manifest记录SHA-256 |
| 合成数据 | [DownSightSmplxdata](https://github.com/cdh290718-oss/DownSightSmplxdata)，提交 `192671271a74b99f41b9eb033d8f88b1dac77ada`，样本000001；用于生成相机/Head锚点及拟合后的评测 |
| D公共准备输入 | `<WORKSPACE>/DownSight/outputs/initialization_ablation_20261109/`，由资产持有人提供；未提供公开下载端点，manifest记录逻辑路径和哈希 |
| 后端和模型权重 | 沿用[既有资源说明](../synthetic_oracle_head_shape_20261108/REPRODUCE.md)与[原资源清单](../../runs/2026-11-08_synthetic-original/evidence/resources.json)，需对应模型授权和访问条件；本目录不重新分发 |
| D初始/各预算/最终NPZ | 同一输出根的 `D_left/`；坐标cam3 CV、米，大小及SHA-256见manifest；没有上传这些完整原始输出 |

外部后端不是Git工作树，原run获取Git版本返回128；只记录了拟合后的源码摘要，不能将其当作拟合前后的不可变性证明。权重哈希沿用既有资源标识，源run未重新逐一计算全部权重哈希；这次归档同样不冒充完整环境审计。

不能访问外部资产的审核者只能审核公开证据，不能声称独立复现。源run的非致命依赖/性能警告、数值检查及未通过的收敛标志均保留；四组运行完成不等于所有组达到严格稳定性门槛，也不等于独立审核完成。此次归档核对记录见[归档报告](../../runs/2026-11-09_archive-template-initialization/report.md)。
