# 只改髋目标：前后对比

[返回报告](report.md)

绿色是真值，蓝色是原条件复跑，橙色是仅替换两髋目标。骨架图同一相机内使用相同坐标轴，侧视坐标为米且横纵单位比例一致；两目不代表独立样本。

## cam3 骨架

![left skeleton](assets/skeleton_left.png)

## cam4 骨架

![right skeleton](assets/skeleton_right.png)

## 原图投影

绿色点为真值，红色点为预测，半透明网格为预测；渲染器可能略去相机平面相交或过长投影三角形。

### left / 原条件复跑

![left control_left](assets/left_control_left.jpg)

### left / 仅替换两髋目标

![left gt_hips_left](assets/left_gt_hips_left.jpg)

### right / 原条件复跑

![right control_left](assets/right_control_left.jpg)

### right / 仅替换两髋目标

![right gt_hips_left](assets/right_gt_hips_left.jpg)
