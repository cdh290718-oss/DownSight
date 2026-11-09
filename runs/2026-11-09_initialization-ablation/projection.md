# 网络初值消融：投影与骨架

[返回完整报告](report.md)

A=网络体型/网络姿态；B=标准体型/网络姿态；C=网络体型/模板姿态；D=标准体型/模板姿态。新四组目标和约束相同，全部身体/手部观测为自动值，Head为真值。

## 收敛

![convergence](assets/convergence.png)

## left / A与D / 3400步

绿：真值；蓝：A；橙：D。同一相机两种预算保持相同轴范围；侧视米制等比例。

![skeleton](assets/skeleton_left_3400.png)

## left / A与D / 20000步

绿：真值；蓝：A；橙：D。同一相机两种预算保持相同轴范围；侧视米制等比例。

![skeleton](assets/skeleton_left_20000.png)

## right / A与D / 3400步

绿：真值；蓝：A；橙：D。同一相机两种预算保持相同轴范围；侧视米制等比例。

![skeleton](assets/skeleton_right_3400.png)

## right / A与D / 20000步

绿：真值；蓝：A；橙：D。同一相机两种预算保持相同轴范围；侧视米制等比例。

![skeleton](assets/skeleton_right_20000.png)

## 原图网格投影

绿点：真值；红点：预测；网格为预测。部分跨相机平面或过长投影三角形由渲染器跳过；关键点接近不代表轮廓、头部可见性或手指形态准确。

### left / 原流程参考

![original](assets/left_original.jpg)

### left / A / 3400步

![projection](assets/left_A_3400.jpg)

### left / B / 3400步

![projection](assets/left_B_3400.jpg)

### left / C / 3400步

![projection](assets/left_C_3400.jpg)

### left / D / 3400步

![projection](assets/left_D_3400.jpg)

### left / A / 20000步

![projection](assets/left_A_20000.jpg)

### left / B / 20000步

![projection](assets/left_B_20000.jpg)

### left / C / 20000步

![projection](assets/left_C_20000.jpg)

### left / D / 20000步

![projection](assets/left_D_20000.jpg)

### right / 原流程参考

![original](assets/right_original.jpg)

### right / A / 3400步

![projection](assets/right_A_3400.jpg)

### right / B / 3400步

![projection](assets/right_B_3400.jpg)

### right / C / 3400步

![projection](assets/right_C_3400.jpg)

### right / D / 3400步

![projection](assets/right_D_3400.jpg)

### right / A / 20000步

![projection](assets/right_A_20000.jpg)

### right / B / 20000步

![projection](assets/right_B_20000.jpg)

### right / C / 20000步

![projection](assets/right_C_20000.jpg)

### right / D / 20000步

![projection](assets/right_D_20000.jpg)
