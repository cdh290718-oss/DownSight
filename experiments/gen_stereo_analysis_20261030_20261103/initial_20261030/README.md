# Gen 输入与前序 A/B/C/D 实验

- [基线和 A/B 对照记录](comparison_AB.json)：基线与 A 有完整摘要；B 失败，记录包含失败原因和部分刚体阶段诊断。
- [A：调整焦距的配置与结果](experiment_A_focal407/)；[B：头部锚点及失败记录](experiment_B_head_anchor/)。
- [C：身体拟合实验](experiment_C/)；其启动日志显示完整流程 worker 失败，后续双目诊断仅使用可取得的自动观测及部分身体结果，不代表 C 端到端成功。
- [D：实验及重试](experiment_D/)；`logs/run.log.txt` 记录首次失败，`output_run2/run_state.json` 和重试日志记录另一次完成，不代表优化质量已验证。
- [历史源码变体索引](source_variants/index.json)：按公开 `269ec28d` 比较，仅收录不同或新增的 Python 模块；其他模块、输入数据、模型和运行环境未纳入，**不能把变体目录直接当作可运行源码树**。代码为历史实验快照，绝对路径等需在迁移时处理；不得覆盖主流程。

配置、JSON 和文本日志中的开发机根路径被替换成 `<WORKSPACE>`；权重、原始图像、完整输出、`.npz` / `.npy` 和 patch 不在 Git 中。与原文件的来源摘要和脱敏记录见本轮 `runs/2026-11-03_gen-stereo-archive/evidence/source_inventory.json`。
