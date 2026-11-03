# 缓存路线三次重复与 provenance 修复

## 范围和状态

`session_hand6/sample_01_cam3` 的同一自动预测缓存、同一标定与资源，使用开发机 egosmplx Python 3.8 和 GPU 0，顺序执行三次独立完整拟合（没有 `--resume`；不重新执行 EgoSMPLX、Sapiens2 或 WiLoR 网络）。三次被测代码均为干净提交 `5996cba4a1037de455180d927e6d513790959d62`。均使用 guided 默认流程、头部参考权重 0.1；结果不应与其他配置的旧指标混为一谈。

## 结果

三次逐帧 validation 均 `passed=true`，匹配两只手。身体自动点 RMSE 均为 **53.0014831348791 px**，手部自动点 RMSE 均为 **12.916784053749964 px**；它们反映与自动预测的差距，不是人工真值精度。三次身体、融合、OBJ 重载最大误差均为 0 m。逐点结果见 [metrics.json](evidence/metrics.json)。

按同名文件与字段比较 10 个 NPZ，共 214 个字段：212 个数值字段在三次两两比较中最大绝对差均为 **0**，2 个非数值字段相等，字段集合及形状均一致。见 [array_comparison.json](evidence/array_comparison.json)，比较脚本见 [collect.py](evidence/collect.py)。这只是同机、同环境和单帧的结果，不证明跨设备一致或算法精度。

每轮 controller、frame、reference 日志和 [脱敏 provenance](evidence/provenance.redacted.json) 在 evidence 中；完整配置和 patch 仅保留在开发机各自的输出 `provenance/`，不公开。已用 `<WORKSPACE>` 替换日志中的开发机工作区路径。当前证据并未列出每个中间 worker 日志；完整输出留在本机对应 run 目录。

## 实现与校验

- 本地 provenance 保存展开配置、pipeline Git 二进制 patch、未跟踪文件内容摘要；后端 Git 仓库时也执行同样处理。当前 `EgoSMPLX` 后端不是 Git 仓库，未能生成后端 Git patch，记录中明确标记不可用。
- 预检计算并核对 EgoSMPLX、Sapiens pose、Sapiens seg、WiLoR、检测器及人体配置的实际 SHA-256；配置了期望摘要但不一致时失败。三次运行版本使用元信息缓存；随后测试发现时间戳不可靠，已改为每次预检重新计算实际摘要。
- 预检依据 `docs/backend_contract.json` 逐项比较后端文件；指纹纳入实际资源摘要及契约摘要。并非所有后端依赖与运行环境都已纳入，无法据此保证完整环境复现。
- 三次实际运行时契约列有原 4 个文件，均匹配。运行完成后静态核查发现默认 `UndistortViT` 的 `UndistortPatch` 相对导入 backend 内部鱼眼相机；它不保证被顶层 `FishEyeCalibrated` 替换。该文件当时实际 SHA-256 与 main 相机相同，随后已加入契约，扩充契约后的 [preflight](evidence/preflight-new-contract.txt) 通过。三次缓存运行不验证网络相机的动态调用。
- [三次运行时的单元测试](evidence/pytest.txt) 记录 31 项通过；完成契约、无缓存摘要与额外测试后 [33 项通过](evidence/pytest-final.txt)。三次拟合执行的是上面固定的 `5996cba`；无缓存校验和扩展契约是拟合结束后的变更，未再重复三次完整拟合。

主网络权重是 8 月 9 日 8 GPU 训练的 snapshot_11，SHA-256 `c7d30f04d13ed80f1941226458a43dfd50fc0ab0e5902701b6c9895e1f92a3b3`。不是 v5_detach_fix 训练线；现有证据不足以确定当初未采用 v5 的原因，详见 `docs/downsight/CHECKPOINT_zh.md`。

历史变体去重、quotepath 问题和 head01/optimized_v4 来源核实不在本轮范围。
