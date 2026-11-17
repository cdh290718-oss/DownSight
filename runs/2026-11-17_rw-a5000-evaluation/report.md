# RW5000 训练完成与检测器／全二维黄金 D 测试

新 RW5000 模型已正常完成训练；本轮相同样本比较显示，二维误差和受控三维诊断均优于旧 RW500 最佳模型。测试集只恢复到冻结候选中的一部分，三维仍使用既有诊断样本，结论不外推为完整官方 benchmark。

## 执行身份与训练完成

执行基线为 `de26a4764c549d96d4832ca4aa005bea2854e466`。执行时工作树包含本轮新目录和既有无关改动；没有修改外部后端或旧档案。Sapiens2 源码提交及本轮实际代码 SHA-256 见 [source.json](evidence/source.json)。`fit.py`、`geometry.py` 沿用上一轮实现；只调整公共输出目录和评价命令标签。

RW5000 第 1 轮完成 1250 次更新，退出码为 0，结束时间为 2026-11-17 13:11:01 UTC（北京时间 21:11:01），监督进程耗时 2984.04 秒，约 49.73 分钟。检查点与训练状态见 [checkpoints.json](evidence/checkpoints.json)。新权重 SHA-256：`3879e3a8879128955d371844ba43e96ed643f1cb088fe7ce7df60aec6cc5bf66`；本地资源为 `<REPO>/outputs/rw_a5000_training_20261117T122117126742Z/results/adapter_01.safetensors`。这是覆盖参数文件，须先严格加载对应官方权重，再覆盖；加载方式见 [evaluate_2d.py](code/evaluate_2d.py)。访问权重需要开发机资产权限，仓库不发布权重。

旧 RW500 与新 RW5000 都取第 1 轮，但分别经历 125 和 1250 次参数更新；旧模型在旧验证集选优，新模型使用扩展验证集。本轮是实际检查点效果比较，不能将全部收益因果归因于数据量。依据见 [training_comparison_scope.json](evidence/training_comparison_scope.json)；训练方案和冻结数据选择见[训练报告](../2026-11-17_rw-a5000-training/report.md)。

## 二维测试

相同官方全图推理、坐标还原及有效点掩码，评价身体对应关节的原图像素欧氏距离。验证集的官方／新模型复用冻结训练预测，并重新校验 ID、掩码及重算指标；旧模型验证集和三组模型的其余样本均重新推理。图像哈希与训练序列隔离检查包含在执行脚本中。指标以有效关节点加权，数值越低越好。

| 集合 | 官方权重 | RW500 第 1 轮 | RW5000 第 1 轮 |
| --- | ---: | ---: | ---: |
| 扩展验证集，400 张 | 43.35 px | 21.36 px | **12.53 px** |
| S6 同人物留出，100 张 | 107.09 px | 17.81 px | **13.36 px** |
| 冻结测试候选可恢复部分，46 张 | 78.31 px | 15.09 px | **10.83 px** |

可恢复测试部分的误差较旧模型降低 **28.23%**；PCK@20px 从 **78.99% → 90.94%**。逐图平均误差有 45 张改善、1 张退化；左右视图和邻近帧相关，不能按独立试验解释。所有模型与集合推理失败数均为 0。逐关节、逐人物、逐序列汇总见 [2d_summary.json](evidence/2d_summary.json)，逐图误差原始数值见 [2d_errors.json](evidence/2d_errors.json)，派生比较见 [improvements.json](evidence/improvements.json) 和 [summary.json](evidence/summary.json)。

测试原计划冻结 200 张，本机仅有不完整测试归档；扫描保留了 `ReadError: unexpected end of data`，从原冻结清单恢复 **46 张／23 对，全部来自 S9 的 12 个序列**，没有补入替代样本。该清单排除了旧 S9 十对诊断所在序列；仍缺其他测试人物，不能称作完成 200 张或代表完整跨人物测试。依据见 [test_availability.json](evidence/test_availability.json)。验证集用于选模；S6 留出与训练同人但序列隔离；S9 部分未参与本轮训练，但未穷尽审计所有历史用途，不能声称全局从未接触的盲测。

训练／测试标签元数据契约一致；人工查看了恢复样本的叠图，没有发现整图坐标偏移。该检查不独立保证遮挡关节的标注精度。证据见 [label_contract.json](evidence/label_contract.json) 和 [test_visual_check.json](evidence/test_visual_check.json)。

## 固定全二维黄金 D 的三维诊断

按当前黄金 D 定义，身体观测使用全二维重投影损失，保留 Head 和先验项。本轮没有改回三维身体观测。对旧 S9 十对诊断固定相机、Head、掩码、均值体型及优化预算，仅替换检测器的身体二维坐标及诊断分数；输入一致性见 [fit_preparation.json](evidence/fit_preparation.json)，配置见 [protocol.json](protocol.json)。身体三维真值只在拟合完成后评价时读取。

| 检测器 | 非 PA MPJPE | PA-MPJPE | 欧氏 RMSE |
| --- | ---: | ---: | ---: |
| 官方权重，本轮重新推理 | 422.49 mm | 209.20 mm | 683.70 mm |
| RW500 第 1 轮 | 111.34 mm | 86.79 mm | 140.56 mm |
| RW5000 第 1 轮 | **96.25 mm** | **73.55 mm** | **119.20 mm** |

相对 RW500，非 PA 改善 **13.55%**，PA 改善 **15.25%**。非 PA 有 9 对改善、1 对增加约 0.12 mm；PA 有 8 对改善、2 对退化。旧 RW500 逐对非 PA 与上一轮完全复现，最大差值为 0。官方权重为本轮新推理，不能与早先使用不同预测文件的官方基线混接。

并非所有关节都改善：非 PA 均值中，右肩 21.95 → 23.88 mm、右肘 28.07 → 29.31 mm、左肩 48.35 → 52.66 mm。完整逐对／逐关节结果见 [results.json](evidence/results.json)、[summary.json](evidence/summary.json)、[3d_improvements.json](evidence/3d_improvements.json)。同一诊断输入上的二维误差见 [diagnostic_2d.json](evidence/diagnostic_2d.json)。

每组运行 10 对、预算 20000 步，全部有可行输出，但**各组通过配置收敛标准的数量均为 0/10**。结果应解释为固定预算下选出的可行解，不能声称完全收敛。评价使用 15 个身体关节（不含 Head，含脚跟代理）；Head 位置、标定和掩码仍有标注辅助，体型固定均值。这是受控诊断，不是端到端部署精度或官方 benchmark。执行及收敛证据见 [fit_execution.json](evidence/fit_execution.json)、[fit_console.txt](evidence/fit_console.txt)、[results.json](evidence/results.json)。

## 复核与交接

本轮没有新增训练、安装或下载。测试恢复、检测器推理、CPU 拟合和三维评价均已结束。具体命令与退出状态分别见 [test_availability.json](evidence/test_availability.json)、[inference_execution.json](evidence/inference_execution.json)、[fit_execution.json](evidence/fit_execution.json)、[evaluation_execution.json](evidence/evaluation_execution.json)。归档截断异常保留，未静默当作完整输入。

发布前从逐点误差独立重算二维／三维汇总，核对相同图像与有效点掩码、拟合源码一致性、执行源码指纹和均值体型初始化；检查全部通过。命令与结果见 [verification.json](evidence/verification.json)，复核程序见 [verify_evidence.py](code/verify_evidence.py)。

可采用 RW5000 作为后续检测器候选；尚未自动切换部署权重。下一步若补齐测试资产，继续使用冻结清单完成多人物评价；如要论证扩大数据本身的作用，需要匹配更新次数的对照。本轮无需继续训练即可确认现有候选有提升。

发布只包含本轮代码、配置、报告、精简执行日志和派生误差证据；图像、标签、权重、完整坐标预测、拟合参数及完整 provenance 留在被忽略的本地输出目录。机器绝对路径已以 `<WORKSPACE>`／`<REPO>`／解释器占位符替换，数值、失败和退出状态保留。发布前检查工作树、暂存区及本次新增提交，排除既有无关改动，确认无数据文件、凭据、超限文件或未脱敏路径；详细清单见 [prepush.json](evidence/prepush.json)。此检查仅覆盖本次发布增量，不代表历史仓库审计。
