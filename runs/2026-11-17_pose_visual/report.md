# 姿态预测图片对照

执行基线 `365b286f46889f388cf9c9ee2592251d59d8a2e7`；工作树包含既有训练脚本修改和未跟踪实验，本轮未改动。根据用户要求生成本地单页网页：同人 S6 留出动作和跨人 S5 验证各选两张，各组按帧四分位及四分之三位选择，依次左眼、右眼。选择结果、图片哈希检查和每图关节数见 [build.json](evidence/build.json)。

网页在同一原图上绘制原模型、全骨干第 1 轮和数据集投影标注，提供图层切换、鼠标悬停、全部被选关节的像素坐标表。标注不代表可见性，缺失鼻点不显示。S6 来自官方训练划分的本地留出序列；S5 来自官方验证划分；图片是单序列样例，不代表整体。坐标与预测来自固定数据清单和相应训练保存输出；S6 预测重新推理，汇总见 [same_eval.json](evidence/same_eval.json)，与既有评估接近。

本地网页：`<WORKSPACE>/DownSight/outputs/pose_visual_20261117/index.html`，含真人原图和预测，仅本机保存，不上传。构建脚本在 [code/build.py](code/build.py)，推理脚本在 [code/evaluate.py](code/evaluate.py)。生成及图片 SHA-256 检查成功。环境中 `node` 不存在，JavaScript 语法和浏览器交互未能通过该方式验证；不声称浏览器测试通过。

## 推送前自检

仅公开代码、报告与脱敏摘要；不含数据集、图片、视频、权重、NPY/NPZ、完整输出、`.env`、凭据、完整 provenance 或 patch，单文件小于 50 MB，路径脱敏。检查工作区、暂存内容、待推提交及远端状态，见 [prepush.json](evidence/prepush.json)；既有未跟踪文件排除。本轮增量自检不代表历史审计。
