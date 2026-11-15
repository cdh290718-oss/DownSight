# D 方法首页与黄金样本固定

将仓库 README 改为以 D 方法、黄金样本与投影为入口，并将此前首页逐字保留在 [README_LEGACY.md](../../README_LEGACY.md)。保留既有默认 EgoSmplx 流程的说明和入口，未改变默认配置或拟合代码。

黄金样本 D 固定为初始化消融中合成 `000001` 的标准体型／模板姿态 D；明确不是早期同名 D，也不是近期真实图像实验。锁定归档提交 `d26b9ee292204506492fb9e1d221d683531465ea`、代码发布提交、样本/配置/检查点、资源 SHA-256 和外部访问说明，见 [golden_d.json](../../experiments/synthetic_template_head_stereo_20261109/golden_d.json) 及既有 [manifest.json](../../experiments/synthetic_template_head_stereo_20261109/manifest.json)。黄金样本是固定研究参考，不声称满足既有黄金回归方案对重复性、容差或跨环境验证的要求；也不等于部署流程。Head真值、生成相机、单一姿态和真值均值体型、未通过的稳定性门槛及固定手指限制均在首页和原归档中保留。

本轮实际执行基线和开始时远端 main 为 `a93ed44f4409fcdc213a007de4f9bfa1c5266aa7`；执行时工作树存在先前无关未跟踪项及本轮新增文件。未运行拟合、检测或模型推理；不是新的复现或精度评估。旧首页字节一致和固定版本见[evidence/identity.json](evidence/identity.json)。

[tools/check_golden_d.py](../../tools/check_golden_d.py) 检查组别身份、关键元数据、外部资源记录一致性及公开文件与固定提交的字节／SHA-256；实际结果见[evidence/golden_check.txt](evidence/golden_check.txt)。通过注入模拟损坏但不改写文件，验证检查器会拒绝被修改的源码，见[evidence/check_verifier.txt](evidence/check_verifier.txt)。本机归档输入和输出资源与清单哈希比对见[evidence/external_hashes.json](evidence/external_hashes.json)；该检查不审计权重、后端环境或重复性。首页和旧首页相对链接检查见[evidence/links.json](evidence/links.json)。

## 推送前自检

检查工作区、暂存内容、拟推送提交及远端状态；本轮不加入任何既有无关未跟踪内容。增量不包含权重、数据集、视频、NPZ/NPY、完整原始输出；单文件小于50MB；无凭据、令牌、`.env`；日志与报告只含仓库相对路径或占位符；无完整本地 provenance 和 patch。逐项证据见[evidence/prepush.json](evidence/prepush.json)。审计仅覆盖本轮增量，不代表历史已全面审计。
