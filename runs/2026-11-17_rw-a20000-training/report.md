# A：RW20000 嵌套扩量训练与固定评测

用户授权继续扩大训练，并先使用现有测试子集。本轮从同一官方 Sapiens2-1B 权重开始，训练 20000 张单眼图一轮；不追加其他训练实验。当前启动状态以 [startup.json](evidence/startup.json) 为准，后台最终状态不应由启动成功推断。

正式任务于 2026-11-17 15:27 UTC（北京时间 23:27）启动，监督进程 PID `307019`。15:31 UTC 检查时已有 10 / 5000 次有效更新，loss 为 0.0007232535，峰值 allocated 显存约 41.2 GiB，训练仍在运行。短测 20 次更新已正常退出、损失与梯度有限。原权重在同一 400 张验证图上的误差为 43.345393 px，与上一轮一致；这是基线验证，不是新模型成绩。依据见 [startup.json](evidence/startup.json)、[probe.json](evidence/probe.json)、[baseline_validation.json](evidence/baseline_validation.json) 和 [startup_console.txt](evidence/startup_console.txt)。

## 数据与隔离

20000 张为 10000 对，保留原 5000 张全部样本及原标签；覆盖 9 个训练人物、401 个序列。按人物与序列均衡扩充，序列内优先选距离已选帧较远的样本。S16 在间隔规则下只有 942 对，使用全部候选，其他人物分配约 1132–1133 对。选择清单、配额和 SHA-256 见 [selection.json](evidence/selection.json)。帧编号间隔不是秒数。

原最小帧编号间隔 10 的候选不足以实现目标；初始容量断言退出后，将间隔改为 5，保留全部旧样本并按容量分配人物配额。该变更意味着新增样本相关性可能增加，不能当作新增独立人物或序列。失败与修正见 [selection_adjustment.json](evidence/selection_adjustment.json)。准备脚本首次还因本轮 `select.py` 与标准库同名而失败；已更名，只清理该失败产生的空输出目录，无图片丢弃。记录见 [preparation_retry.json](evidence/preparation_retry.json)。

图像从外部只读 SQLite 缓存读取，写入本仓库新输出目录；每张解码、记录 SHA-256 并检查有限标签，沿用原画内监督规则，不把画内等同于可见。完整统计见 [dataset.json](evidence/dataset.json)。旧 5000 张记录完全一致，原验证集 400 张清单保持字节一致；原 S6 留出、现有 S9 测试清单及图片也再次核对，训练没有与其序列或图像哈希重叠。证据见 [nested_data.json](evidence/nested_data.json)。新增样本叠图目视检查见 [visual_review.json](evidence/visual_review.json)；官方 CPU 预处理检查见 [preflight.json](evidence/preflight.json)。

现有测试仅为冻结 200 张候选中可取得的 46 张／23 对，全部来自 S9；不扩展、不替换、不下载，也不据其结果选模。它是有限的回归检查，不代表完整跨人物 benchmark。旧 S6 同人物留出和旧 S9 十对三维诊断仍单独报告。

## 训练及自动评测

训练脚本与 RW5000 完全相同，从官方权重重新训练全骨干和姿态头，保持原学习率、有效 batch 4、BF16、随机种子及监督定义。单轮计划 5000 次参数更新。按上一轮实测耗时线性估算约 3–4 小时，数据准备与后续评测另计；共享 GPU 的负载可能改变耗时，估算见 [timing_plan.json](evidence/timing_plan.json)。配置见 [config.json](config.json)，相对上一轮的配置差异和源码哈希见 [source.json](evidence/source.json)。更新次数随数据规模增加，且归一到整轮的学习率调度也随总步数展开；这不是匹配计算量的数据规模因果对照。

既有环境单卡运行，不安装、不修改外部环境、不停止他人进程。短测结果见 [probe.json](evidence/probe.json)，实际后台启动记录见 [launch.json](evidence/launch.json)。短测参数不保存，正式训练重新加载官方权重。

后台监督程序在训练正常结束后，校验流水线源码指纹，再依次执行：官方／RW5000／RW20000 的固定二维评测、全二维黄金 D 准备与拟合、身体三维评价、汇总及证据复核。验证集的官方／新模型复用本轮保存预测并重算核对；其他检测器结果重新推理。各阶段退出状态写入本地 `pipeline_status.json`，任何阶段失败即停止后续阶段；自动评价不代表自动审核或发布结论。

三维只更换检测器身体 XY 和诊断分数，沿用相同 Head、相机、掩码、均值体型、损失和优化预算，身体三维标签在拟合后评价时读取。仍有标注辅助和脚跟代理；必须报告收敛状态与逐关节退化，不能只报告均值。具体设置见 [protocol.json](protocol.json)。

本地进度读取命令：`python runs/2026-11-17_rw-a20000-training/code/status.py`。`exit.json` 表示训练子进程的真实退出状态；`pipeline_status.json` 表示包含后续评测的整个流水线状态。输出和检查点只保留本仓库忽略目录，需要开发机资产访问权限。最后权重尚未生成时，不记录或声称其哈希。

## 身份与发布

执行基线为 `6fc30623c78c8eb4db2943955c9c0c5ed83a0563` 加本轮新增文件；保留工作区既有无关修改。Sapiens 源码版本、工作树状态及实际执行文件 SHA-256 见 [source.json](evidence/source.json)。尚无独立 Claude 审核，交接见 [review.md](review.md)。

本次发布增量只含本轮代码、配置、报告及精简脱敏证据；数据、图片、权重、完整预测和本地 provenance 不发布，单文件小于 50 MB。开发机路径替换为 `<WORKSPACE>`；命令、数值、退出和异常保留。工作区、暂存内容、本次提交及远端核对见 [prepush.json](evidence/prepush.json)。该检查不覆盖既有历史。后台训练和评测后续产生的证据需要另行检查后发布，不在本次启动报告中预先宣称结果。
