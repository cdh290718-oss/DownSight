# 代码、资源和复现范围

[返回实验入口](README.md)。本目录是已完成实验的整理入口，保存精确设置、精选数值证据和带哈希的源文件索引；不是自带数据/权重的一键运行包。本轮归档没有重跑推理或优化。

## 固定版本和入口

- 源实验发布提交：`c4e2b9de76573139fa66cbb953bf973d2af88c77`。
- 当时实际拟合基线 HEAD：`9012bbf856570e85da60261190ead1092cce9997`，拟合代码当时为该 run 的新增工作树文件，其执行哈希已记录并与发布版核对。
- 主配置：`case=oracle, seed=left, method=improved, anchor=oracle`；预先指定的主结果，不能事后按真值选优。
- [manifest.json](manifest.json) 给出固定提交链接和 SHA-256；[settings.json](settings.json) 是实际配置的逐字节快照。修改本目录配置不会自动改变源拟合器，因为源拟合器从自己的 run 目录读取设置。

| 入口 | 职责 |
|---|---|
| [prepare.py](../../runs/2026-11-08_synthetic-head-robustness/prepare.py) | 从旧自动观测生成不含头部的公共输入、三角化目标和分离的锚点文件；预先列出全部条件 |
| [initialization.py](../../runs/2026-11-08_synthetic-head-robustness/initialization.py) | 体型信息子空间与无人体真值的双目刚体初始化 |
| [fit.py](../../runs/2026-11-08_synthetic-head-robustness/fit.py) | 联合优化、输入访问保护、检查点选择、参数重建和保存 |
| [run.py](../../runs/2026-11-08_synthetic-head-robustness/run.py) | 批量运行已预先定义的条件，保留退出状态和源代码哈希 |
| [evaluate.py](../../runs/2026-11-08_synthetic-head-robustness/evaluate.py)、[evaluate_shape.py](../../runs/2026-11-08_synthetic-head-robustness/evaluate_shape.py) | 拟合结束后读取人体真值，评测关节、头部、中性体型并渲染 |
| [test_method.py](../../runs/2026-11-08_synthetic-head-robustness/test_method.py)、[verify.py](../../runs/2026-11-08_synthetic-head-robustness/verify.py) | 几何、输入隔离、同初值、损失重算、限幅、共享坐标与保存重建检查 |

实际主组命令记录于[执行证据](../../runs/2026-11-08_synthetic-head-robustness/evidence/execution.json)，等价于在仓库根目录运行：

```bash
"$BODY_PYTHON" runs/2026-11-08_synthetic-head-robustness/fit.py --case oracle --seed left
```

此命令仅展示历史执行入口，不能在现有输出目录原样覆盖重跑。拟合器对已存在的 case 输出会报错；准备脚本也不是完整的可重入工具。实际复现前必须在新的仓库 run/output 目录准备入口并明确记录路径适配；保留原设置和源哈希，任何方法变动都应作为新实验。需要已有后端 Python 环境、`PYTHONPATH=<REPO>`、`PYTHONDONTWRITEBYTECODE=1`，临时文件和缓存路径指向新的仓库输出目录。源 `run.py` 展示了完整环境设置；本实验使用 CPU。

准备、数值检查、拟合、评测、验证按源 run 的依赖顺序执行。批量 runner 会跑所有条件；若只复现主组，源批量评测/验证也需要匹配其预期条件集合，不能因缺少其他输出而报告整批验证通过。

## 哪些资源在 GitHub，哪些不在

| 资源 | 获取与核对 |
|---|---|
| 设置、实现、数值证据、投影图 | 本仓库源 run；固定提交路径与 SHA-256 见 manifest |
| 合成数据 | [DownSightSmplxdata](https://github.com/cdh290718-oss/DownSightSmplxdata)，固定提交 `192671271a74b99f41b9eb033d8f88b1dac77ada`、样本 `000001`；不能默认 HEAD 等于该版本 |
| 网络 raw 初始化、自动观测、局部配置、分离的锚点 | 开发工作区 `outputs/` 或由资产所有者提供的外部存储；逻辑路径、大小和 SHA-256 见 manifest 的 external_inputs 与[准备证据](../../runs/2026-11-08_synthetic-head-robustness/evidence/preparation.json)。未提供公开下载端点 |
| EgoSMPLX 后端、模型权重与人体模型 | 依赖已有部署及各模型提供者授权；访问要求、资源 SHA-256 见[原资源清单](../../runs/2026-11-08_synthetic-original/evidence/resources.json)和[原输入清单](../../runs/2026-11-08_synthetic-original/evidence/inputs.json)。本目录不安装或再分发它们 |
| `oracle_left/initial.npz`、`oracle_left/final.npz` | 位于 `<WORKSPACE>/DownSight/outputs/synthetic_head_robustness_20261108/` 下，哈希见 manifest 的 external_outputs；坐标为 cam3 CV、米；未上传原始二进制输出 |

原报告与完整执行记录应同时保留。[原始失败与修正](../../runs/2026-11-08_synthetic-head-robustness/evidence/development_failures.json)包括开发测试漏传手/脸参数、发布前 Markdown 空行检查失败；后续检查通过。14 次正式拟合全部完成，不代表整个历史没有失败。非致命依赖/性能警告见[精简日志](../../runs/2026-11-08_synthetic-head-robustness/evidence/fit_log_summary.json)。

**不能访问外部输入、后端与模型的审核者，只能审核公开代码和证据，不能声称独立复现了实验。** 只有重新运行并保存新的命令、退出状态、输入/代码哈希和评测结果，才构成新的复现记录。此次归档的链接、数值、哈希核对记录见[归档运行报告](../../runs/2026-11-08_archive-oracle-shape-baseline/report.md)。
