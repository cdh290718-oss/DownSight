# A：5000 张 RW 全骨干微调首轮

用户已明确授权启动。执行基线 `4b9c0c37a9daff5212d5de6bec2cbc1d8e8e0a7d`；实际执行为基线加本轮新增源码与配置，工作树同时有既有无关修改与未跟踪实验，保持原状。截至 [startup.json](evidence/startup.json) 记录时，正式单卡训练已开始并有有效参数更新；后台任务继续运行，尚未完成。

## 输入与准备

训练 5000 张单眼图、2500 对、9 人；验证 400 张、200 对、4 人。候选清单与上轮冻结的 SHA-256 一致，旧 S6 留出序列继续排除。只读访问外部 SQLite 缓存，将选中的原始图片及派生监督清单写入本仓库新建忽略输出；未改外部数据或后端。全部 5400 张解码并记录图像哈希，训练／验证间无序列和图像哈希重叠；左右眼配对检查通过。各关节画内监督计数与全部统计见 [dataset.json](evidence/dataset.json)。

抽检全部 13 人、两种姿态的左右眼，共 52 张叠加图；可见肘腕大体对应肢体，未发现明显整体缩放／平移错误或系统性左右颠倒。肩髋和部分下肢被头／躯干遮挡，抽检不证明骨骼中心与表面点语义完全等价。监督沿用有限且在矩形画内的投影点规则，不是独立可见性；原始 pose JSON 仍不可访问。详见 [visual_review.json](evidence/visual_review.json)，不发布图片原件。

官方 CPU 预处理检查覆盖每个人的两眼，输入与目标形状、有效权重、鼻点屏蔽和有限值检查通过；图像哈希再次核对。证据见 [preflight.json](evidence/preflight.json) 和 [preflight_execution.json](evidence/preflight_execution.json)。

## 训练设置与执行边界

从官方 Sapiens2-1B pose 权重重新加载，训练全骨干与姿态头，保持 308 输出并监督 12 个身体点。首轮运行 1 个 epoch，有效 batch 为 4，计划 1250 次优化器更新；BF16、全图 bbox、官方预处理及 flip 验证，学习率、优化器与监督归约沿用原实验设置，精确配置见 [config.json](config.json)。鼻点不监督，其他未监督通道不保证能力保持。原模型也参与扩展验证集比较与检查点选择。

本轮适配已发布的全骨干训练脚本，替换验证掩码读取方式，增加源码／清单哈希固定、独立短测与失败记录；源文件及改动说明见 [source.json](evidence/source.json)。只用既有环境和单张 GPU，未安装或下载。输入、权重、配置与外部 Sapiens 版本按既定哈希／提交核对。短测参数不保存，正式训练再次从官方权重开始。

后台 supervisor 记录实际 child 退出码，包括模型加载前的失败；完整日志和参数只留在忽略输出。未承诺无人值守自动调参或追加轮次。

独立测试仍是冻结的 200 张标签候选，RGB 缺口未解决。本轮用户授权在训练／验证准备完成后先启动；不将扩展验证集成绩称为独立测试或下游全二维黄金 D 成绩。本轮未启动三维拟合。

## 实际启动状态

20 次优化器更新的单卡短测正常退出，损失和梯度有限；峰值 PyTorch allocated 显存约 41.2 GiB。短测仅用于资源和数值检查，不证明长时间稳定或训练质量。实际退出码、耗时和原始字节数见 [probe.json](evidence/probe.json)，启动信息见 [probe_launch.json](evidence/probe_launch.json)。

正式任务于北京时间 2026-11-17 20:21 启动，后台 supervisor PID 为 `255365`；位置见 [launch.json](evidence/launch.json)。检查时已有 10 次以上更新，完整检查时刻、当时步数和 loss 见 [startup.json](evidence/startup.json) 与 [启动日志](evidence/startup_console.txt)。此时未生成退出记录，不声称完整运行已成功。之后 supervisor 会在本机 job 下写入 `exit.json`。

原模型在扩展验证集的平均二维误差为 43.345393 px，图像评估失败数为 0；逐关节数字与样本口径见 [baseline_validation.json](evidence/baseline_validation.json)。这不是训练后的成绩，也不能与旧 S5 小验证集直接比较。正式训练后的验证和最佳轮次将保存在本机 `results/`。

实际模型全部参数均可训练，优化器参数组覆盖全部参数名称；见 [trainable.json](evidence/trainable.json)。配置里的历史 `train_last_blocks` 字段不被本轮全骨干脚本使用，不能据该字段推断冻结范围。Sapiens 源码版本及干净工作树检查见 [backend.json](evidence/backend.json)。

运行使用已有估算的一轮训练加验证约 60–90 分钟规划余量，不保证精确完成时刻；原测量与假设见 [上一轮 timing](../2026-11-17_rw-a-budget/evidence/timing.json)。短测和准备已单独完成，不算作正式训练轮次。

## 发布检查

本轮发布只含源码、配置、报告和脱敏精简证据。数据、叠加图、权重、优化器状态及完整预测均留在忽略输出目录。工作区、暂存和待推提交及远端检查结果见 [prepush.json](evidence/prepush.json)；不发布凭据、令牌、环境文件、数组、完整 provenance 或 patch。开发机路径用 `<WORKSPACE>` 等占位；只检查本轮增量，不代表历史已审计。
