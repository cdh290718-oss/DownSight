# v0 + 手部二阶段：双目网格投影

- Run ID：`2026-11-04_mesh-projection-review`；基线 `d9312f1`，开始时工作树干净。实际运行的是本轮新增、当时尚未提交的可视化代码；资源和源码摘要见 [inputs.json](evidence/inputs.json)。
- [**打开 12 帧图像与指标索引**](evidence/index.md)。每张图包含 cam2、cam3，v0 与二阶段全图，以及两只手的同区域放大对比。
- 本轮只读取已有模型参数、自动观测和已发布的逐帧评测，重建网格用于渲染；没有重新拟合、改动方法或改写此前评测结果。选帧用途为诊断，不代表数据集的无偏抽样。

## 选帧与图例

12 帧从上一轮 179 个有预测的时刻中按双腕对 DFM 三维误差的较大值排序：约 1%、3%、5% 分位各一帧作为较好，45%、50%、55% 各一帧作为中等，95%、97%、99% 各一帧作为较差；另加指定的画外右腕 00270、DFM 两手最近点距 <5 cm 的首个时刻 03060、左手无可用 cam2 二维或双目三维手点的中位误差时刻 04770。冲突时跳过已选帧并选最近分位；详细逻辑和逐帧理由、数值见 [selection.json](evidence/selection.json) 和[索引](evidence/index.md)。选帧发生在已有评测之后，不用于选择或调整模型。

网格为青色，头部顶点为洋红色；绿色线是 DFM 肩、肘、腕和 MediaPipe 顺序 21 点手骨架，**DFM 是模型参考，不是独立真值**；橙色圈是 Sapiens 自动腕检测，蓝点是上一轮匹配并通过二维筛选的 WiLoR 手点。蓝点也画在 v0 对照图上，但 v0 拟合没有使用 WiLoR；cam3 的单目蓝点不一定被二阶段优化消费。放大区域以 DFM 参考为中心，严重偏离或画外的预测可能不在局部图内，应结合全图看。

## 遮挡、投影与局限

使用上轮图像估计双目标定、Double Sphere 投影和已有 SMPL-X 模型；网格从已保存的 v0 和二阶段参数按原批大小重建。12 帧重建的 v0 顶点、v0 关节与二阶段关节相对保存值的最大坐标绝对差均为 **0 m**，见 [reconstruction_validation.json](evidence/reconstruction_validation.json)。模型有 10,475 个顶点、20,908 个三角面；以头关节最大蒙皮权重标记 3,345 个头部顶点，颜色在面上插值。

仓库原有线框绘制不做自遮挡，原有实心版本只按面平均深度排序；本轮复用投影与网格重建，增加逐像素相机射线—三角面求交和深度缓冲，只绘制该像素最近的网格表面，网格不是点云。重叠三角面顺序无关的测试见 [rasterizer_test.txt](evidence/rasterizer_test.txt)，双目整流 / 鱼眼投影的 36 个合成点一致性检查见 [geometry_test.json](evidence/geometry_test.json)。为避免相机平面附近的反向大面，含非正深度顶点的面和投影边长超过 250 px 的面不绘制；鱼眼三角形的候选范围由顶点包围盒近似，因此轮廓边缘并非解析无误差的曲线。

真实物体对模型的遮挡使用显示用途的保守双目 SGBM：左右视差一致、纹理和距离筛选，比较场景与模型的射线距离，并留 4 cm 裕量。**这只能处理有可靠深度的部分像素，不能保证真实场景中所有被物体挡住的模型面都被去除，也可能因匹配错误误删。** 48 个全图面板中，场景深度覆盖模型可见像素的比例中位数为 16.14%，被该掩码去除的比例中位数为 1.39%；每面板数据见 [render_stats.json](evidence/comparisons/render_stats.json)，深度匹配数见 [scene_depth_stats.json](evidence/scene_depth_stats.json)。这些深度图只用于绘图，不参与模型拟合或评测。DFM 骨架、自动检测点作为诊断标记直接叠在图像上，不声称其经过深度遮挡处理。

48 个全图面板中只有 10 个有可见头部像素；其余头部在相机前向范围外、被模型或场景遮挡、或被上述近相机面裁剪，因此没有人为绘制不可见头部点。具体每面板像素数见 [render_stats.json](evidence/comparisons/render_stats.json)。图像是半透明表面叠加，只用于观察对齐；图像中的衣物、遮挡物与人体形状都可能使可视差异不等同于关节误差。DFM 参考元信息亦提到既往外参拟合，双目重投影差异不应自动解释成解剖真值误差。

## 执行与授权

未安装或升级任何包；使用已有 EgoSMPLX 环境及其 NumPy、OpenCV、Numba。首次单帧重建未通过与原批次结果的一致性断言，随后以原批大小在新目录重建；首次光栅器测试和场景深度运行的实现错误也已修复并重新运行，保留原因和退出状态于 [execution_notes.txt](evidence/execution_notes.txt)。原数据、权重和原始模型输出留在本机。

用户已明确确认持有允许发布 Gen 帧及派生叠加图的授权；公开 [Gen 数据集页面](https://huggingface.co/datasets/genrobot2025/Gen-HumanWholeBody)标注许可证为 `other`，本轮没有独立取得具体许可条款。本轮发布 12 张带标注、缩放后的派生图，未包含 EXIF；摘要见 [images_manifest.json](evidence/images_manifest.json)。

## 推送前自检

本轮增量仅包含代码、脱敏证据和 12 张用户确认授权的派生叠加图；不含权重、数据集、视频、`.npz` / `.npy`、完整原始输出、凭据、`.env`、完整 provenance 或 patch，单个文件均小于 50 MB。报告和日志中的开发机路径已移除；图像、逐帧索引、渲染与几何检查见 [verification.txt](evidence/verification.txt)，工作区、暂存内容和远端核对见 [prepush-check.txt](evidence/prepush-check.txt)。本轮增量自检不构成对既有历史的审计。
