# 9 月 30 日缓存配置重跑与相机对照

## 历史结果

在干净 DownSight 提交 `32119e25a2dc50ab2e368a1183aad61ab12cba17` 的隔离检出中，以历史 `egosmplx_source_update_20260930/fullfit_current/jobs/sample_01_cam3.json` 配置和缓存重新执行一次完整拟合。原配置未指定 `head_neck`；为当前预检仅增加与实际文件核对的权重摘要，并使用新的输出路径。没有使用 `--resume` 或重跑网络。

结果身体 RMSE **24.193575241018337 px**、手部 **11.73983920158647 px**，与 9 月 30 日记录精确相同；八个阶段参数及网格 NPZ 的字段、形状和数组逐值相同，最大绝对差 0。详见 [historical_comparison.json](evidence/historical_comparison.json)。这是一个样本及当前环境的对照，不证明全部输入或所有运行环境都等价。

与此前头部参考 0.1 的 53.0014831348791 / 12.916784053749964 px 运行相比，[缓存输入摘要相同，除输出路径和清单路径外配置仅 `head_neck` 不同](evidence/input_config_comparison.json)；运行代码提交在 pipeline 下仅完整性校验文件有差异。两次 raw 和 rigid 参数一致，body 参数和最终融合网格不同，见 [阶段比较](evidence/head01_stage_comparison.json)。这支持差异与头部参考配置有关，但不构成对损失项或所有阶段的独立因果消融。53.00 / 12.92 没有被冻结为黄金基准。

## 源码与权重

后端源码与非数据配置清单包含 **710 个路径**，整体 SHA-256 为 `58d33a919aaf882e177bf2e889791a21a677777269c98f6a74a130e3016b5dc2`；逐文件摘要及策略见 [backend_source_manifest.json](evidence/backend_source_manifest.json)。预检已把此清单加入后端验证结果和续跑指纹。范围是声明的源码扩展名及配置文件；排除生成结果、模型、数据及单独部署环境，不跟随目录符号链接，也不覆盖所有运行环境或外部库；该后端本身不是 Git 仓库。用户因此仍不能把摘要解释为全部依赖的完整锁定。

预检对五个模型／检测器文件和人体配置必须提供有效期望 SHA-256，缺少、格式无效或与实际内容不符均拒绝；`cached_sha` 已改名为 `file_sha256`，每次读取内容计算。

## 相机数值对照

对 cam3、cam4 标定各使用 17×11 网格、CPU float32 Torch 和 NumPy 路线。[cam3](evidence/camera_cam3.json) / [cam4](evidence/camera_cam4.json) 的原始类实现之间，所测投影和反投影最大差为 0；但运行时拟合用的 `stable_world2camera` 与 backend 原始 Torch 投影的最大**单坐标**差分别为 **1.40558 px** / **1.56192 px**。在该采样下，稳定函数与 backend NumPy 投影最大差约 0.000061 px，原始 Torch 与 NumPy 的差分别约 1.40560 / 1.56192 px。当前不能声称运行时两种投影数值等价；这些结果不覆盖 GPU、全部点或完整原生网络执行。

## 验证与范围

完整拟合使用原始提交 `32119e2`；本轮新源码树哈希与严格哈希变更在拟合完成后加入，并通过 [真实预检](evidence/preflight.txt) 和 [单元测试](evidence/pytest.txt)。未用新增校验版本重新跑完整拟合。脱敏运行记录和日志在 evidence；完整输入与模型仍在开发机。历史差异去重不属于本轮。
