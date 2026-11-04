# 腕部长尾误差：筛选、原因与候选修复

- Run ID：`2026-11-04_wrist-tail-gating`。拟合基线 `3fc4751d2cc698953cebbf2e81aeba61439fe8be`，开始时工作树干净；本轮新增代码运行时未提交，代码、输入、权重和后端摘要见 [inputs.json](evidence/inputs.json)。
- 结论：原 stereo-keypoint-gating **单独接入仍有左腕长尾**。逐帧证据显示，若干被标为左手的指尖在两目和三维中靠近右手指尖，原筛选未对指尖做左右重合检查；这些冲突的指尖约束与腕部观测共同进入拟合。增加保守的跨手指尖剔除后，本批测试帧的左腕 p90 明显下降；固定损失分母后改善仍存在。
- 限制：新规则是在检查本批测试结果和图像后提出的**事后候选**，不是经独立数据验证的生产修复。补充早期帧仍有漏检和画外腕的大误差；DFM 是模型参考，非独立验证的人体真值。

## 公平对照与筛选接入

所有主对照使用与上一轮零扰动相同的后 10 个时刻、图像估计双目外参、相同自动检测、同一躯干点云、头位置及初始化；零扰动头位置是上一轮由前 10 帧 DFM 头关节汇总出的**模拟标定值**，逐帧 DFM 腕点仅用于事后评测。重新运行的 500 步基线与上轮零扰动腕结果逐点一致，最大三维差为零，见 [metrics_final.json](evidence/metrics_final.json)。不因筛选或预测失败删去 DFM 有效的评测帧。

将已发布 gating 判定接入既有拟合器：按原规则保留／丢弃二维和三维候选，并在当前拟合所用的图像标定下重新计算已接受点的三维坐标及协方差。原规则产生的接受判定与在当前标定下重算整套规则的结果分别报告，避免把不同外参混在同一拟合中。原规则可能新增肘部三维点，不只是删除指尖；拟合器消费 cam2 的二维点与双目三维点，若出现仅 cam3 的二维点则不消费，实际计数见 [validation_final.json](evidence/validation_final.json)。所有主测试帧的左右腕在原已发布筛选中均为三维候选，故该规则本身并没有剔除这些腕点。

跨手候选从原筛选结果出发：如果某指尖及左右腕都有有效自动三角化，指尖到同侧腕距离大于 15 cm，且另一侧腕至少近 5 cm，则移除该指尖的二维和三维约束，**不自动交换身份**。阈值是本轮事后探索值，不是普适人体尺度或正确身份的证明。后 10 帧共移除 12 个左手指尖约束，前 10 帧移除 1 个；逐点证据见 [cross_hand_decisions.json](evidence/cross_hand_decisions.json)。双手相接、检测噪声和腕点错误均可能触发或漏掉该规则。

## 修复前后：相同后十帧

每个指标是 cam2 下拟合腕关节相对 DFM 腕关节的**三维欧氏距离**，每侧 10 帧；p90 使用这 10 个数的第 90 百分位，不是独立数据上的泛化保证。逐帧数据和沿视线指标见 [metrics_final.json](evidence/metrics_final.json)。

| 条件 | 步数 | 左腕中位 / p90 cm | 右腕中位 / p90 cm |
|---|---:|---:|---:|
| 原拟合 | 500 | 3.34 / 16.36 | 2.07 / 3.43 |
| 已发布 gating 判定 | 500 | 2.61 / 15.03 | 1.75 / 2.48 |
| 当前标定下重算原 gating | 500 | 2.26 / 15.08 | 1.66 / 3.44 |
| 原 gating + 跨手剔除候选 | 500 | 1.93 / 3.27 | 1.65 / 2.67 |
| 不使用任何指尖观测 | 500 | 1.77 / 2.66 | 1.60 / 2.03 |
| 原拟合 | 1500 | 3.14 / 16.69 | 1.72 / 2.51 |
| 已发布 gating 判定 | 1500 | 2.27 / 14.83 | 1.69 / 1.99 |
| 原 gating + 跨手剔除候选 | 1500 | 1.86 / 2.50 | 1.52 / 1.92 |
| 不使用任何指尖观测 | 1500 | 1.87 / 2.73 | 1.51 / 2.18 |

原损失对有效点求平均：删除点后，其余点相对权重会改变。为分离这一影响，以下消融固定原始分母：原拟合二维／三维分母为 122／98；跨手候选固定到已发布 gating 的 103／103。分母、逐帧和结果见 [fixedden_metrics.json](evidence/fixedden_metrics.json)。

| 固定分母对照 | 步数 | 左腕中位 / p90 cm | 右腕中位 / p90 cm |
|---|---:|---:|---:|
| 原 gating + 跨手剔除 | 500 | 1.97 / 3.38 | 1.71 / 2.48 |
| 原 gating + 跨手剔除 | 1500 | 2.02 / 2.41 | 1.52 / 1.91 |
| 只保留指尖二维项 | 500 | 4.34 / 16.06 | 1.97 / 3.02 |
| 只保留指尖三维项 | 500 | 2.30 / 11.46 | 2.05 / 2.95 |
| 指尖二维、三维均移除 | 500 | 1.63 / 2.76 | 1.65 / 2.79 |

跨手剔除的改善在固定分母后仍在，说明不能只归因于重加权；但仅保留二维或仅保留三维指尖项仍有左腕尾部，说明两类项都可能参与冲突。非线性拟合、其他项和初始化仍可能影响单帧结果；不能把全部误差简单分配给某一项。

## 后十帧：每个坏腕的归类和叠加图

本表把原拟合、原 gating、跨手候选或无指尖四个条件在 500／1500 步中任一结果相对 DFM 达到或超过 5 cm 定为坏腕；这只是诊断阈值，所有 10 帧和两侧腕都保留在总体指标中。表中数值为 500 步 **原拟合 / 原 gating / 跨手候选 / 无指尖** 的三维距离 cm。图例：绿 DFM，黄腕检测，红原拟合，青原 gating，白跨手候选，洋红无指尖；橙色为有效的原指尖检测。全画面的身份图中橙色连线代表检测器的左手归属、青色为右手，白圈是跨手候选剔除点。图像显示投影，不单独证明三维深度或真实解剖身份。每帧 500／1500 图和逐点证据见 [case_classification.json](evidence/case_classification.json)。

| 帧 / 腕 | 三维差 cm | 原因归类与证据 | 图 |
|---|---:|---|---|
| 02700 左 | 4.15 / 5.65 / 5.43 / 3.48 | 原 gating 关联的拟合恶化；原腕三角化对 DFM 为 2.62 cm，1500 步候选降至 3.81 cm。原因未能唯一定位。 | [腕](evidence/wrist_crops/frame_02700.jpg) · [身份](evidence/identity_overlays/frame_02700.jpg) |
| 02970 左 | 15.42 / 2.08 / 2.11 / 1.78 | 左手多根二维指尖靠近右手同名点，原 gating 因双目不一致剔除它们；腕三角化距 DFM 1.71 cm，拟合冲突明显。 | [腕](evidence/wrist_crops/frame_02970.jpg) · [身份](evidence/identity_overlays/frame_02970.jpg) |
| 03240 左 | 16.26 / 17.93 / 1.43 / 1.21 | 四对左右同名指尖双目位置近乎重合、三维差 0.48–1.64 cm；原 gating 接受这些候选，跨手规则移除左侧四点。腕三角化距 DFM 1.38 cm。 | [腕](evidence/wrist_crops/frame_03240.jpg) · [身份](evidence/identity_overlays/frame_03240.jpg) |
| 03780 左 | 17.28 / 14.71 / 1.46 / 1.47 | 四个左指尖三维点更靠近右腕，多对左右指尖近乎重合；原 gating 保留四点，跨手候选剔除。腕三角化距 DFM 1.50 cm。 | [腕](evidence/wrist_crops/frame_03780.jpg) · [身份](evidence/identity_overlays/frame_03780.jpg) |
| 04320 左 | 7.60 / 7.89 / 1.97 / 2.57 | 四个左指尖到右腕明显更近而被跨手候选剔除；手与物体重叠，身份和遮挡不能单凭图像完全确认。腕三角化距 DFM 3.03 cm。 | [腕](evidence/wrist_crops/frame_04320.jpg) · [身份](evidence/identity_overlays/frame_04320.jpg) |
| 04860 右 | 6.43 / 4.64 / 4.49 / 1.94 | 右腕三角化距 DFM 1.89 cm；无指尖和更长优化均改善，但本规则没有确认该侧跨手误认，具体机制未定。 | [腕](evidence/wrist_crops/frame_04860.jpg) · [身份](evidence/identity_overlays/frame_04860.jpg) |

上面指尖近邻与距离证据来自无 DFM 的 [identity_audit.json](evidence/identity_audit.json)，原筛选和当前标定重算判定见 [gating_decisions.json](evidence/gating_decisions.json)。这些迹象支持跨手归属冲突，但不能代替人工标注。

## 前十帧补充检查：仍有失败

新规则依据后十帧诊断提出，另对前十帧重跑相同拟合。它们曾用于计算 DFM 头位置模拟标定，也在此前项目探索中出现过；因此是补充检查，**不是独立受试者或无泄漏基准**。每侧均评估 10 帧，未丢弃筛选失败的腕。结果和逐帧值见 [first10_metrics.json](evidence/first10_metrics.json)。

| 条件 | 步数 | 左腕中位 / p90 cm | 右腕中位 / p90 cm |
|---|---:|---:|---:|
| 原拟合 | 500 | 2.79 / 12.35 | 2.53 / 8.93 |
| 原 gating + 跨手剔除 | 500 | 1.42 / 4.68 | 2.44 / 8.04 |
| 原拟合 | 1500 | 2.31 / 12.35 | 2.20 / 8.75 |
| 原 gating + 跨手剔除 | 1500 | 1.55 / 4.11 | 2.44 / 7.73 |

下表列出原拟合或候选在任一步数达到 5 cm 的每个腕；数值为 500 步 **原拟合 / 候选** cm。补充帧腕图展示 1500 步结果，表格为 500 步；身份图展示原检测。详细证据见 [first10_classification.json](evidence/first10_classification.json)。

| 帧 / 腕 | 三维差 cm | 原因归类与证据 | 图 |
|---|---:|---|---|
| 00000 左 | 7.78 / 7.91 | 左右同名指尖近乎重合；两腕接近时距离差不足 5 cm，新阈值没有触发，属于漏检。 | [腕](evidence/first10_crops/frame_00000.jpg) · [身份](evidence/first10_identity_overlays/frame_00000.jpg) |
| 00270 左 | 5.44 / 4.32 | 腕三角化距 DFM 2.22 cm，拟合存在偏差；另一侧同时出现明显归属歧义，不能唯一归因。 | [腕](evidence/first10_crops/frame_00270.jpg) · [身份](evidence/first10_identity_overlays/frame_00270.jpg) |
| 00270 右 | 40.14 / 32.45 | DFM 右腕投影在 cam2 画外，cam2 的右腕检测却接近左腕；原三角化无效、已发布 gating 丢弃右腕。缺少可靠约束，仍有严重误差。 | [腕](evidence/first10_crops/frame_00270.jpg) · [身份](evidence/first10_identity_overlays/frame_00270.jpg) |
| 00810 右 | 5.46 / 5.32 | 腕三角化距 DFM 3.16 cm，拟合与参考更远；无已确认跨手误认，原因未定。 | [腕](evidence/first10_crops/frame_00810.jpg) · [身份](evidence/first10_identity_overlays/frame_00810.jpg) |
| 01080 左 | 22.45 / 0.97 | 多根左指尖在一视角近乎复制右手，另一视角不一致；原 gating 剔除这些指尖，结合候选后改善。 | [腕](evidence/first10_crops/frame_01080.jpg) · [身份](evidence/first10_identity_overlays/frame_01080.jpg) |
| 02160 左 | 11.23 / 3.53 | 两手在图像中重叠，候选剔除一个可疑左指尖；原 gating 同时改变其他点，无法把改善全部归因于该单点。 | [腕](evidence/first10_crops/frame_02160.jpg) · [身份](evidence/first10_identity_overlays/frame_02160.jpg) |

## 验证、授权和后续限制

所有 28 次拟合（主对照、固定分母、补充帧）及结构检查通过；两个步数、固定头锚点和筛选输入见 [validation_final.json](evidence/validation_final.json)、[first10_validation.json](evidence/first10_validation.json)、[fixedden_metrics.json](evidence/fixedden_metrics.json)及[日志摘录](evidence/fit_checkpoints.txt)。首次重算 gating 时，某点无穷协方差标准差导致严格 JSON 序列化失败；该点已标为无效，随后将非有限诊断值记为 null 并在新输出目录重跑成功，未覆盖旧结果。完整记录见 [execution_notes.txt](evidence/execution_notes.txt)。

图像公开依据：用户明确确认持有允许公开 Gen 帧及派生叠加图的授权。公开的 [Gen-HumanWholeBody 数据集卡](https://huggingface.co/datasets/genrobot2025/Gen-HumanWholeBody)将许可标为 `other`，具体访问条件需登录；本轮没有独立取得具体许可条款，授权来源按用户确认记录。发布的仅是这些坏帧的派生叠加或裁剪，原始全分辨率帧、数据集、权重和二进制输出保留在开发机；图像无 EXIF，摘要见 [images_manifest.json](evidence/images_manifest.json)。

建议将跨手剔除作为候选保留在实验层，下一步用独立序列、佩戴者及人工身份／可见性标注验证，尤其测试双手接触、两腕接近、画外和缺失双目腕点。当前没有将它合入默认生产流程，也不能据这 20 帧认定右腕长尾已经解决。

## 推送前自检

本轮增量按仓库规范检查：仅包含代码、脱敏证据和用户确认授权的坏帧派生叠加图；无权重、数据集、视频、`.npz`／`.npy`、完整原始输出、凭据或 `.env`，每个文件小于 50 MB。开发机绝对路径已从报告和日志替换；未提交完整本地 provenance 或 patch。代码摘要、图像 EXIF、报告链接和证据 JSON 检查见 [verification.txt](evidence/verification.txt)，暂存内容与远端核对见 [prepush-check.txt](evidence/prepush-check.txt)。实验输入和原始输出留在忽略的本地输出目录，此检查不等于对既有历史文件的审计。
