# 二维干预研究汇总与网页发布

状态：完成。既有实验、指标澄清与论文参考已整理为 [summary.md](summary.md)；网页已部署并验证。DownSight 提交完成发布前自检后推送。

- 网页：https://cdh290718-oss.github.io/egofishpose-demo/2d-to-3d-summary-20261117/
- 网页源码仓库：`cdh290718-oss/egofishpose-demo`，新增 `2d-to-3d-summary-20261117/index.html`。
- 网页提交：`9bb86404d60430476c47a02440ae74141c1ec21a`。
- 部署：https://github.com/cdh290718-oss/egofishpose-demo/actions/runs/44062643538，GitHub Pages 结论 `success`。

## 执行范围与基线

本轮基线 `42c647b479623486192959ddea88e57c615c2af1`，开始时本地与 GitHub main 一致。工作树已有其他轮次的修改和未跟踪文件，全部保留且不纳入本轮提交。本轮仅新增此 run；没有新增拟合或训练，也没有安装包、下载模型或数据、改动外部后端。

读取本地已提交实验的派生 evidence；源文件标识与 SHA-256 见 [build.json](evidence/build.json)。数字从既有分析与逐样本结果提取，汇总数据见 [summary_data.json](evidence/summary_data.json)。实验源快照与网页源提交分开记录，不能把本轮整理称为独立复现实验。

## 汇总内容

汇总说明数据为 UnrealEgo-RW 的 S9 固定序列小样本；主实验为直接二维重投影，辅助为固定混合，均非原样 D。包含二维坐标修正的逐点欧氏距离定义、PA 与不对齐指标的区别、全身／局部效果、逐关节与逐样本例外、收敛与同时统计支持限制，以及参考二维仍有残差时的待排查因素。

论文参考与本地诊断分开展示，不提供混合协议的公平排名。来源为数据集官方页面与指定版本预印本；保留训练数据、时序／附加输入、作者本地评测与未复现状态、划分版本和开发期测试集使用披露。核查日期、原始表行、来源 URL 和响应摘要见 [literature_baselines.json](evidence/literature_baselines.json)。不公开完整抓取页面。

## 网页交互与验证

页面为自包含静态 HTML，不依赖前端外部包。支持直接二维／固定混合、标准／延长预算、不对齐／PA 的切换；提供全身修正对照、局部效果矩阵、关节同时区间、样本细表与源证据链接。无脚本时仍可阅读核心结果和表格。

实际执行了 8 种方法／预算／指标组合、704 个条件／样本细表选择，逐项对照保存的指标；矩阵点击与输入修正数值检查通过。检查命令、退出状态见 [page_checks.json](evidence/page_checks.json)。复用既有 Node v22.22.1，未安装依赖。没有实际浏览器渲染环境，因而没有完成真实视口截图或指针布局验证；已完成结构、标签、数值及 DOM 适配执行检查。

GitHub Pages 完成构建与部署，原始状态见 [pages_deployment.json](evidence/pages_deployment.json)。线上规范 URL 返回 HTTP 200，响应 SHA-256 为 `025908fadfae6abc931d96f3b8b65e180d2a92d1bb1acee24b8d6be7f4b9bc98`，与已审查本地 HTML、实际推送文件逐字节一致，见 [online_verified.json](evidence/online_verified.json)。

## 失败与修正

首次来源表格解析未覆盖网页内的 JavaScript 表格片段；第二次定位同时匹配了预印本的总体摘要行，已改为同时匹配模型名称和数值。新页面尚未存在时，空稀疏检出也曾失败，随后保留根 .nojekyll 文件，未改动原站点。具体命令、退出状态及修正见 [initial_setup_failures.json](evidence/initial_setup_failures.json)。

最初网页请求在部署传播阶段返回错误状态，所有请求均保留在 [online_attempts.json](evidence/online_attempts.json)，最终响应已验证成功。原实验的检查点保存失败、修复与重新运行记录在实验报告中保留并提供链接；本轮没有重新运行或覆盖它们。

## 发布与安全检查

站点沿用既有公开 GitHub Pages 发布方式；只新增本次页面路径，其他站点文件与公开受众保持不变。GitHub 连接器对该仓库仅有读取权限，发布使用开发机既有 Git 访问；未申请新的连接权限。网页发布前检查了新工作树、暂存 HTML、实际待推送增量和远端状态，见 [pages_prepush.json](evidence/pages_prepush.json)。

DownSight 工作树、暂存内容与本次提交增量的发布检查见 [prepush.json](evidence/prepush.json)。仅包含本轮新文件；不含原始图像／坐标、权重、NumPy 数组、视频、完整原始输出、凭据、开发机绝对路径、完整本地 provenance 或 patch；单文件小于规定限制。开发机路径使用仓库相对路径或占位符。检查仅覆盖本轮增量，不代表历史已全面审计。

## 交接

本轮完成汇总和网页发布目标；待 Claude 依据公开证据审核。后续原样 D 与实际检测器／论文基线的同协议实验仍待用户决定，不能把本轮网页整理误解为新增了那些对照。
