# UnrealEgo-RW：以真实二维身体点检验 D 方法

此前误将用户希望作为二维输入的非 Head 关节三维标注直接输入拟合；本轮按更正重跑前四对。输入是两目官方 GT15 二维标注，经鱼眼双目三角化生成 D 的身体三维目标；非 Head 的官方三维标注只在拟合结束后用于评测。Head 位置使用同样本三维标注；Head 朝向仍来自历史拟合而非真值。四对均属于同一受试者小样本，不能代表全数据集。

| 样本 | 本轮 GT15 PA mm | 本轮 GT15 不对齐 mm | 输入二维点的左 / 右投影平均距离 px | 上轮直接输入身体三维 GT 的 PA mm |
|---|---:|---:|---:|---:|
| 01 | 29.26 | 31.74 | 5.81 / 6.21 | 27.01 |
| 02 | 29.28 | 30.96 | 6.41 / 7.45 | 27.58 |
| 03 | 31.70 | 32.95 | 6.87 / 8.06 | 29.68 |
| 04 | 29.42 | 29.44 | 6.44 / 7.89 | 26.66 |
| 四对平均 | 29.92 | 31.27 | — | 27.73 |

精确数字、逐点误差、不同条件参照和公式见[汇总](evidence/summary.json)及[结果](evidence/results.json)。二维距离是已参与拟合的点的投影残差；3D GT15不含Head，使用网页基准 toe-to-heel proxy 对应，3D标签只用于评测。PA 对齐消除尺度和刚体位姿，不等于绝对误差。上轮条件不仅更换目标，还更换了三角化协方差，不能把差值单独归因于二维点。历史网页拟合在同四对的 GT15 PA均值为109.42 mm，但本轮拥有准确标注二维点、3D Head和标注来源双目标定；不能视为公平的实际检测器或无标注方法对比。

投影绿点为官方2D目标，红点为模型投影：[01 左](assets/body01-left.jpg) · [右](assets/body01-right.jpg)；[02 左](assets/body02-left.jpg) · [右](assets/body02-right.jpg)；[03 左](assets/body03-left.jpg) · [右](assets/body03-right.jpg)；[04 左](assets/body04-left.jpg) · [右](assets/body04-right.jpg)。原图本身可见头部，头部入画不代表失败；渲染不模拟场景遮挡并忽略跨相机平面或过长三角形。二维点近不证明形体、深度或旋转均正确。

## 条件与限制

执行基线 `a2866aca92bcdfb8eb9cdf1ecefc3cd77df76562`，工作树包含本轮代码及此前无关未跟踪文件。保留 D 模板初始化、真实 Head 位置/估计朝向软约束、身体观测分组损失和体型可观测子空间；不提供 WiLoR 手点；内部 source 字段沿用 sapiens 兼容标签，实际身体目标来自官方二维标注。原网页 GT15 二维目标按已知图像适配变换，脚点为 heel proxy。双目相对位姿来自同数据集另一 S9 序列的标注配准，非独立设备标定；因此不能把结果称为完全无标注或真实部署精度。Head 与 SMPL-X 的语义差及相机安装误差尚未验证。

前处理先测试沿用历史8px检测噪声假设，样本01–03各9/15、04为11/15通过，不拟合该条件。看到筛选结果后，为匹配用户提出的准确二维点条件，将假设改为理想二维观测按1px噪声、8mm尺度下限、0.5m最大三角化标准差重新估计协方差；四对均15/15通过。1px是工程假设而非标注测得精度；保留前者与后者的筛选和重投影摘要分别见[原筛选](evidence/preparation.json)、[理想条件](evidence/preparation_ideal.json)、[执行证据](evidence/execution.json)。GT2D 坐标、三角化完整点、3D GT只存本机 outputs，发布派生残差而非数据集原件。

每组按训练目标选择可行检查点；四组各开放一个体型方向，最终beta非零；四组均未通过原严格稳定性诊断。其他模型固有限制包括固定手指、工程姿态范围和缺少自碰撞约束。投影 NumPy/PyTorch 数值一致性检查见[projection_check.json](evidence/projection_check.json)；首次检查因未设置 PYTHONPATH 失败，修正后通过，见[check_initial_failure.txt](evidence/check_initial_failure.txt)。训练命令和输出见[execution.json](evidence/execution.json)与[fit_log.txt](evidence/fit_log.txt)。

## 发布

输入资源和完整参数只留在开发机，本目录代码为执行源码，绝对路径以 `<WORKSPACE>` 等占位；公开证据不包含原数据、权重、完整 provenance 或 patch。推送前审计见[evidence/prepush.json](evidence/prepush.json)，仅覆盖本轮增量，不代表既有历史已审计。无法获取外部数据和模型的审核者只能审阅公开证据。
