# D组标准体型/模板姿态归档

已将用户指定的D组整理到 [experiments/synthetic_template_head_stereo_20261109](../../experiments/synthetic_template_head_stereo_20261109/README.md)。本轮只归档，没有重跑拟合、检测或评测，没有修改源run或默认流程。

## 内容

新增入口、方法依据、结果与对照、边界、复现说明、精确设置和D组配置、精选数值证据、带SHA-256的代码/图片/外部资产索引。直接引用原投影图，唯一拟合实现继续保留在源run，避免出现两套漂移的拟合器。

明确区分同设置的网络初值对照A和上一轮旧配置；D主结果固定为20000步预算内训练目标选出的19960步检查点，保留3400步等预算及其他预定预算结果。真值Head/生成相机、单姿态与零beta真值优势、未通过的严格稳定性标志、手指固定与足部未观测问题均保留。

D仅拟合与原四组prepare的依赖分别说明：D不需要EgoSmplx预测参数，但四组prepare仍会为A/B/C提取网络因素；原批量评测也依赖全部条件。归档没有声称提供或验证新的D专用runner。

## 来源与检查

实际归档基线及开始时远端main均为 `485fc71ca37d3c68a74bf99338fe247c197bf6eb`。源拟合的实际基线为 `6d5994401b33450ea3343d20f44f2cc54e9ac5d6`，新增工作树代码的执行哈希已由源run保存。当前工作树只有本轮新增目录和既有无关未跟踪point_comparison_followup.json，后者未改动或提交。源review待Claude；本轮review同样待审。

归档提取命令、退出状态和源码哈希见evidence/archive_generation.json；核对结果见evidence/archive_check.json。检查源提交字节/哈希、精确设置、D身份、各预算数值、读取隔离记录、外部输入/输出哈希、主表数字及本地文档/图片链接。没有把公开证据核对称为新的实验复现。

外部资产访问条件、生成数据固定提交、后端与模型限制见实验的REPRODUCE.md及manifest.json。模型权重和全部外部环境没有在本轮重新审计；拟合后端不是Git工作树的源记录保留。无法取得外部资产只能审核公开证据。

## 推送前自检

逐项检查工作区、暂存、本次提交与远端main；收据摘要为evidence/prepush.json，完整收据留在本轮outputs。确认本轮增量不含权重、数据集、视频、NPZ/NPY或完整原始输出；不含凭据、令牌、.env；单文件小于50MB；机器路径用 `<WORKSPACE>` / `<BODY_PYTHON>` 等占位符表示；不发布完整本地provenance或patch。

只发布当前归档run和新experiment目录；源代码、图片和旧实验不改动。本自检仅覆盖本轮增量，不代表已审计全部历史。提交自检通过后才推送。
