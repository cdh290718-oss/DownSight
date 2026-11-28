# Sapiens2 双目轮廓 · 12 对测试样本

沿用指定图库的前 12 对测试样本。每张图在原鱼眼图上叠加当前 Sapiens2 双目轮廓拟合使用的边界点，点击图片可放大。

白色表示人体前景外缘，青色表示左右手，黄色表示左右前臂，橙色表示左右上臂，绿色表示左右大腿。部位颜色覆盖同位置的白色前景边界。

边界提取来自归档的 Sapiens2-1B 分割概率，原图每 4 像素采样，显示的点是参与比较的采样位置。按历史拟合规则，前景使用 1 减背景概率，部位使用最大类别概率，阈值均为 0.55；可信区域至少 12 个采样点且边界至少 6 点才显示。画框边缘排除。人体前景包含服装等非背景类。未显示的部位可能是不可见、出画或未满足筛选。

| 人物 / 样本 | 左目 | 右目 |
|---|---|---|
| S1 / pair001 | [![左目](assets/pair001_left.jpg)](assets/pair001_left.jpg) | [![右目](assets/pair001_right.jpg)](assets/pair001_right.jpg) |
| S1 / pair022 | [![左目](assets/pair022_left.jpg)](assets/pair022_left.jpg) | [![右目](assets/pair022_right.jpg)](assets/pair022_right.jpg) |
| S1 / pair028 | [![左目](assets/pair028_left.jpg)](assets/pair028_left.jpg) | [![右目](assets/pair028_right.jpg)](assets/pair028_right.jpg) |
| S1 / pair034 | [![左目](assets/pair034_left.jpg)](assets/pair034_left.jpg) | [![右目](assets/pair034_right.jpg)](assets/pair034_right.jpg) |
| S9 / pair036 | [![左目](assets/pair036_left.jpg)](assets/pair036_left.jpg) | [![右目](assets/pair036_right.jpg)](assets/pair036_right.jpg) |
| S9 / pair037 | [![左目](assets/pair037_left.jpg)](assets/pair037_left.jpg) | [![右目](assets/pair037_right.jpg)](assets/pair037_right.jpg) |
| S9 / pair045 | [![左目](assets/pair045_left.jpg)](assets/pair045_left.jpg) | [![右目](assets/pair045_right.jpg)](assets/pair045_right.jpg) |
| S9 / pair055 | [![左目](assets/pair055_left.jpg)](assets/pair055_left.jpg) | [![右目](assets/pair055_right.jpg)](assets/pair055_right.jpg) |
| S11 / pair068 | [![左目](assets/pair068_left.jpg)](assets/pair068_left.jpg) | [![右目](assets/pair068_right.jpg)](assets/pair068_right.jpg) |
| S11 / pair078 | [![左目](assets/pair078_left.jpg)](assets/pair078_left.jpg) | [![右目](assets/pair078_right.jpg)](assets/pair078_right.jpg) |
| S11 / pair088 | [![左目](assets/pair088_left.jpg)](assets/pair088_left.jpg) | [![右目](assets/pair088_right.jpg)](assets/pair088_right.jpg) |
| S11 / pair094 | [![左目](assets/pair094_left.jpg)](assets/pair094_left.jpg) | [![右目](assets/pair094_right.jpg)](assets/pair094_right.jpg) |

各眼图像和概率文件分别与归档 SHA-256 核对，颜色图保留原图坐标。此图展示预测轮廓，未与人工分割真值比较。[输入核查和各通道点数](evidence/results.json)

执行基线 bbc60839d3635be7da15690f48c1f89e3dbb3cb5；工作树含已有无关修改和本轮新增文件。使用已归档的分割缓存与双目原图，在 CPU 上生成派生叠图，命令为 OPENBLAS_NUM_THREADS=1 python -B runs/2026-11-28_sapiens_contour12/render.py，退出状态为 0。源码和证据中的开发机路径以仓库路径或占位符表示。

推送前检查工作区、暂存区、待推送提交及远端。既有无关文件与数据留在工作区，发布本轮代码、脱敏核查和派生叠图；不包含权重、数据集、视频、数组、完整原始输出、凭据、环境文件、机器路径及完整 provenance 或 patch。每个文件小于 50 MB。历史文件未在本轮审计。[发布自检](evidence/prepush.json)
