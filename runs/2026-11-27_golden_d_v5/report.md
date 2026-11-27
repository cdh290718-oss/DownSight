# 黄金 D v5 定版

用户将 v4 身体阶段与已选的透视 WiLoR 固定指姿、双目只转腕方案指定为当前黄金方法 `golden-d-2d-v5`。首页和协作指南已指向 v5，固定清单记录了身体父版本、手部算法及其输入、实现和参考证据的指纹。[架构与方法 README](../../docs/downsight/golden-d-v5/README.md) · [当前清单](../../configs/golden_d.current.json) · [v5 固定清单](../../configs/golden_d.2d-v5.json)

从冻结的四个真实身体起点和逐眼 WiLoR 候选重新运行手部求解。001 至 004 的双手候选来源、得分和选择逐项复核，四份全身参数数组与参考解完全一致，四份 OBJ 的 SHA-256 分别与归档相同。公开回放命令为 `bash runs/2026-11-27_golden_d_v5/replay.sh`。机器路径以仓库相对位置或占位符表示，数据、模型和原始输出留在开发机。[逐帧核验](evidence/verification.json) · [原图、全身与手部渲染](../2026-11-27_perspective_wrist_gallery/report.md)

v4 清单以原字节归档；原 v4 复现、核验和真实视频拟合入口均显式读取该归档。使用既存 v4 输出复核了 24 对历史选择与参数文件指纹。此次四帧新回放从身体起点和手部缓存开始，未重新处理原始视频。[v4 归档清单](../../configs/golden_d.2d-v4.json) · [执行记录](evidence/execution.json)

执行及定版基线为 927341aa7ef18dffa96ac1564be2cc9c60b17e9d；工作树有既存无关改动。发布前核对工作区、暂存、待推提交和远端。增量包含方法文档、配置、脚本和脱敏核验结果；检查文件大小、凭据、本机路径及数据类型。[增量发布检查](evidence/prepush.json)
