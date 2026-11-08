# 完整头部位姿＋躯干轮廓：投影比较

[数值报告](report.md)

绿色点是真值，红点是预测；青色是身体，紫色是头部。显示统一使用真值 Brown–Conrady 相机，近面交叉或过长的三角形省略。新结果未做 MANO 手替换。

## left

实际 Sapiens 轮廓候选（腿部未启用）

![](assets/left_contour_targets.jpg)

| 上轮位置＋交替 | 本轮完整位姿＋轮廓联合优化 |
|---|---|
| ![](assets/left_previous_position_alternating.jpg) | ![](assets/left_se3_contour_joint.jpg) |

完整位姿但无轮廓对照

![](assets/left_se3_joint.jpg)

原最终（含 MANO）

![](assets/left_original_final.jpg)

真值

![](assets/left_truth.jpg)

## right

实际 Sapiens 轮廓候选（腿部未启用）

![](assets/right_contour_targets.jpg)

| 上轮位置＋交替 | 本轮完整位姿＋轮廓联合优化 |
|---|---|
| ![](assets/right_previous_position_alternating.jpg) | ![](assets/right_se3_contour_joint.jpg) |

完整位姿但无轮廓对照

![](assets/right_se3_joint.jpg)

原最终（含 MANO）

![](assets/right_original_final.jpg)

真值

![](assets/right_truth.jpg)
