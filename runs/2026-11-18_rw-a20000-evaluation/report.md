# RW20000 完成与固定评价

训练和全部评价现已完成：二维均值继续改善，固定全二维黄金 D 下非 PA／PA 均值再下降约 5%，但增益小于上一档且仍有局部退化。原三维自动阶段因模块搜索路径缺失而未能启动，本轮保留失败并在新输出目录完成恢复；没有追加训练或修改评测样本。

## 训练完成与二维结果

北京时间 2026-11-18 02:26:40 完成训练，耗时约 2.99 小时，完成 5000 次参数更新、1 个 epoch，退出码 0。第 1 轮在固定验证集优于官方基线。训练状态及三个权重 SHA-256 见 [checkpoints.json](evidence/checkpoints.json)，时间转换见 [completion_time.json](evidence/completion_time.json)。新权重位于 `<REPO>/outputs/rw_a20000_training_20261117T152700444455Z/results/adapter_01.safetensors`，SHA-256 为 `57be3cd654227a7a3ecb3a7a29bdff544f7288fc63d14f41ef71226e7d971c98`；需先加载同一官方权重再覆盖参数。权重仅在开发机，访问需要相应资产权限。

训练设置与嵌套采样参见[启动报告](../2026-11-17_rw-a20000-training/report.md)。相比 RW5000，仍是同一批训练人物与序列，样本增至 20000 张、更新次数增至 5000，采样最小帧编号间隔从 10 放宽到 5。当前比较不能把全部收益因果归于不同图片数量，邻近帧也不独立。

| 二维平均误差 | 官方 | RW5000 | RW20000 | 相对 RW5000 下降 |
| --- | ---: | ---: | ---: | ---: |
| 验证 400 张 | 43.35 px | 12.53 px | 11.99 px | 4.30% |
| S6 同人物留出 100 张 | 107.09 px | 13.36 px | 11.85 px | 11.34% |
| S9 可用测试 46 张 | 78.31 px | 10.83 px | 10.45 px | 3.53% |

测试子集 PCK@20px 从 90.94% 增至 91.85%。三组模型各集合均无推理失败。原图有效关节点欧氏误差、掩码和推理流程与上一轮相同。官方／新模型验证预测来自训练保存结果并重算；其他组合重新推理，旧 RW5000 验证均值与上轮相差约 0.0000012 px，仅属数值微差，见 [validation_reproduction.json](evidence/validation_reproduction.json)。依据见 [2d_summary.json](evidence/2d_summary.json)、[2d_errors.json](evidence/2d_errors.json)、[inference_execution.json](evidence/inference_execution.json)。

均值改善不等于所有关节改善：测试中的左肩 11.48 → 12.79 px、右腕 11.10 → 11.26 px、左髋 9.95 → 10.24 px、右髋 9.79 → 11.29 px。其余测试关节均值下降；验证集左肩也略有退化。原始精度与派生计算见 [2d_improvements.json](evidence/2d_improvements.json)。

测试继续使用原冻结清单可读取的 46 张，全部来自 S9，并不是完整 200 张；该集合已经多次查看，应视为有限的回归比较，不声称新的盲测。验证用于选模，S6 留出与训练同人，三者分开解释。

## 原自动任务失败与恢复

训练和二维阶段正常退出。原三维拟合在导入 `egosmplx_pipeline.body_common` 时因缺少仓库模块路径退出，尚未开始优化。完整脱敏异常见 [original_fit_failure.txt](evidence/original_fit_failure.txt)，阶段退出与耗时见 [original_pipeline_execution.json](evidence/original_pipeline_execution.json)。这是启动脚本问题，不是检测器或优化数值失败。

恢复使用 [resume.py](code/resume.py) 显式设置 `PYTHONPATH=<REPO>`，外部后端与环境不变；输入 prepared 文件逐字节复用，改为本轮新输出目录。`fit.py`、`geometry.py` 和协议文件与原任务完全一致，保持全二维黄金 D 及相同 Head、相机、掩码、均值体型、优化预算；仅继续原已准备好的检测器对照。恢复代码及校验见 [source.json](evidence/source.json)、[fit_preparation.json](evidence/fit_preparation.json)，实际阶段退出见 [recovery_execution.json](evidence/recovery_execution.json)。原自动任务的失败记录保持原样，恢复结果单独记录。

## 三维结果

| 固定十对三维诊断 | 非 PA MPJPE | PA-MPJPE |
| --- | ---: | ---: |
| 官方权重 | 422.49 mm | 209.20 mm |
| RW5000 | 96.25 mm | 73.55 mm |
| RW20000 | **91.27 mm** | **69.71 mm** |

相对 RW5000，非 PA 下降 5.17%、PA 下降 5.22%。非 PA 9 对改善、1 对退化；body07 从 82.72 增至 95.07 mm，增加 12.36 mm。PA 有 6 对改善、4 对退化。旧 RW5000 非 PA 逐对完全复现上一轮，最大绝对差为 0。汇总与逐对数值见 [summary.json](evidence/summary.json)，派生变化见 [3d_improvements.json](evidence/3d_improvements.json)。

非 PA 关节均值中，颈、右腕、左肩、右髋退化；右腕为 40.46 → 46.90 mm，左肩为 52.66 → 53.98 mm。PA 也有若干关节退化，完整逐关节信息见 [results.json](evidence/results.json) 和 [3d_improvements.json](evidence/3d_improvements.json)。

全部 30 个实例有可行结果，但各组均为 **0/10 通过设定收敛标准**。这是相同 20000 步预算下的可行解比较，不能称完全收敛。评价仍为旧 S9 十对诊断、15 个身体关节，含脚跟代理，使用标注辅助 Head／标定／掩码与固定均值体型；不是端到端官方 benchmark。依据见 [fit_execution.json](evidence/fit_execution.json)、[fit_console.txt](evidence/fit_console.txt)、[results.json](evidence/results.json)。

恢复全部阶段退出码为 0；已从逐点误差重算二维和三维均值，核对相同样本及掩码、源码身份、体型 rank 0、完整模型和拟合运动学的一致性。复核通过，见 [verification.json](evidence/verification.json) 及 [recovery_execution.json](evidence/recovery_execution.json)。

## 解释与下一步

结果支持本轮扩量有小幅收益，但不是普遍改善：46 张二维测试图中逐图均值恰有 23 张改善、23 张退化；验证为 268 张改善、132 张退化，同人物留出为 83 张改善、17 张退化。相关双目视图和邻近帧不作为独立试验解释，见 [summary.json](evidence/summary.json)。

当前证据适合将 RW20000 保留为候选。直接继续扩量的预期收益需要重新评估；优先分析右腕、肩髋和 body07 等退化样本，再决定困难姿态采样或训练设置。当前仅一次训练且更新次数不同，不能宣称已找到数据规模上限。未自动切换部署权重或启动下一轮训练。

## 执行与发布范围

本轮基线 `4a4ead16ad79c31b30c705e5bfaf326fc65173fc` 加新增恢复目录。原训练执行的是 `6fc30623c78c8eb4db2943955c9c0c5ed83a0563` 加后来发布到 `4a4ead1` 的训练源码。旧工作区无关修改和自动产生的未发布旧目录文件不改动、不混入本轮提交。没有安装、下载或修改外部资产。

仅发布代码、配置、报告、精简执行日志及派生误差，不含图片、数据集、权重、完整预测、拟合参数或完整 provenance。机器路径以 `<WORKSPACE>`／`<REPO>` 脱敏；失败、完整数值和退出状态保留。增量发布前检查见 [prepush.json](evidence/prepush.json)，不代表仓库历史已审计。
