# RW20000：脚掌平行于地面的 12 / 15 点双目拟合

四个 S9 姿态，逐帧双肩双髋初始化，头随上胸；双目共享一个人体。左列自由脚、右列双脚平放，地面侧视图中的绿线是地毯估计高度。两行依次为 12、15 点。[下载 16 个 OBJ](https://github.com/cdh290718-oss/DownSight/releases/download/rw-s9-flat-feet-20261119/rw_s9_flat_feet_OBJ.zip)

| 条件 | 实际脚底平均倾角 (°) | 15 点非 PA (mm) | 15 点 PA (mm) | 共同 12 点非 PA (mm) | 共同 12 点重投影 (px) |
|---|---:|---:|---:|---:|---:|
| 12 点，自由脚 | 17.46 | 51.59 | 40.34 | 53.00 | 12.69 |
| 12 点，双脚平放 | 0.91 | 52.88 | 41.67 | 53.00 | 12.69 |
| 15 点，自由脚 | 57.83 | 65.77 | 52.82 | 52.55 | 13.09 |
| 15 点，双脚平放 | 1.63 | 65.02 | 53.67 | 52.02 | 13.42 |

倾角逐只脚从导出 OBJ 读回，按脚跟、前脚掌两侧三个顶点簇构成的平面与本帧估计地面计算，均值涵盖 8 只脚。脚掌参考坐标系硬约束为水平，SMPL-X 蒙皮与姿态变形会让实际顶点产生小残差。共同 12 点的投影误差始终对照官方画布二维标注。15 点非 PA 为左相机坐标三维平均欧氏距离，PA 在逐帧相似对齐后计算，所有组采用相同关节映射。[逐只脚与逐帧证据](evidence/results_auto.json) · [均值](evidence/summary.json)

12 点组加平脚后，共同 12 个关节的三维误差与重投影误差均保持原值。这些关节止于踝，改变脚掌旋转不会移动它们；用 15 个评测关节时则会计入脚趾根。15 点拟合把脚趾观测纳入损失，固定脚倾角后其余关节也会调整。body02 的 15 点自由组有明显脚尖下垂，平脚组侧面能直接看到改善。

还有独立的接地高度问题：平脚组的脚底取样中心低于估计地面 4.9–17.7 cm，图中绿线保留其估计高度，没有作图像平移或将两脚抹到同一高度。地面高度亦未经过外部测量。若目标是消除明显竖脚，本次四帧优先选 12 点平脚；若输出需要落脚位置，还需要另行确定接触状态与地面高度。

## 逐帧网格与原图

鱼眼总览按行依次为 12 点自由脚、12 点平脚、15 点自由脚、15 点平脚。每行从左到右为两个 OBJ 视角、左目和右目原图叠加。下方的地面侧视图采用另一套共享的水平参考，适合判断脚掌和地面的夹角。

| 样本 | 地面侧视图 | 四组网格和双目叠加 | 12 点平脚全尺寸 | 15 点平脚全尺寸 |
|---|---|---|---|---|
| body01 | [脚与地面](assets/body01__floor.jpg) | [总览](assets/body01__auto.jpg) | [网格](assets/body01/flat12__obj.jpg) · [左](assets/body01/flat12__left.jpg) · [右](assets/body01/flat12__right.jpg) | [网格](assets/body01/flat15__obj.jpg) · [左](assets/body01/flat15__left.jpg) · [右](assets/body01/flat15__right.jpg) |
| body02 | [脚与地面](assets/body02__floor.jpg) | [总览](assets/body02__auto.jpg) | [网格](assets/body02/flat12__obj.jpg) · [左](assets/body02/flat12__left.jpg) · [右](assets/body02/flat12__right.jpg) | [网格](assets/body02/flat15__obj.jpg) · [左](assets/body02/flat15__left.jpg) · [右](assets/body02/flat15__right.jpg) |
| body03 | [脚与地面](assets/body03__floor.jpg) | [总览](assets/body03__auto.jpg) | [网格](assets/body03/flat12__obj.jpg) · [左](assets/body03/flat12__left.jpg) · [右](assets/body03/flat12__right.jpg) | [网格](assets/body03/flat15__obj.jpg) · [左](assets/body03/flat15__left.jpg) · [右](assets/body03/flat15__right.jpg) |
| body04 | [脚与地面](assets/body04__floor.jpg) | [总览](assets/body04__auto.jpg) | [网格](assets/body04/flat12__obj.jpg) · [左](assets/body04/flat12__left.jpg) · [右](assets/body04/flat12__right.jpg) | [网格](assets/body04/flat15__obj.jpg) · [左](assets/body04/flat15__left.jpg) · [右](assets/body04/flat15__right.jpg) |

## 平脚如何实现

人工在各左目图上选择裸露地毯区域，用标定的鱼眼双目匹配特征、三角化并用 RANSAC 初估平面，再用归一化光度误差细化。在两组交错图像块上分别拟合时，两个法向的夹角范围为 0.19°–2.32°；这是重复性检查，绝对地面倾角缺少独立测量。使用的是背景纹理，不使用这四个样本身体或脚的三维 GT 求地面。[地面估计](evidence/floor.json) · [区域与特征](floor.py) · [光度细化](plane.py)

中性 SMPL-X 模板双脚在脚跟、前足两侧分别选取低处顶点簇，建立脚底坐标系。拟合时将每只脚的参考法向固定到本帧地面法向，脚尖方位保留在水平面上的投影，膝、髋与踝位置可动；踝的物理局部旋转由目标世界朝向和膝关节旋转反推。15 点条件的脚趾根代理在二维损失内随平脚约束更新。导出前验证全 SMPL-X 前足位置与求解器中约束后的前向运动学一致。脚的高度和接触状态均未固定，离地动作、脚跟抬起和踮脚也被统一施加水平方向。[拟合实现](experiment.py) · [物理姿态导出与验证](assess.py)

仅使用既有 RW20000 检测预测，无新训练；12 点为双肩、肘、腕、髋、膝、踝；15 点再加同一检测结果的颈和双脚趾根代理。各条件共用同一组相机、标注 Head 位置、初始人体朝向、零权重绝对头朝向、固定颈头局部旋转与均值体型。补跑的自由脚 12 点与上一轮肩髋起点非 PA 差 0.00000 mm。四帧同一受试者，不能推断所有动作的落脚状态。[输入核对](evidence/inputs.json)

## 执行与发布

执行基线 `f8fc5ae32df369f92bc269cb9c7cb538d1f7ef4d`，工作树含本轮脚本和其他既有改动。每组 20,000 步，选择最低可行目标检查点，0 / 16 通过原严格末段收敛诊断。适用范围为相同预算下的结果。[执行记录](evidence/execution.json)

本轮公开增量仅为代码、报告、精简证据和授权的派生 JPG，原图、数据、权重、完整输出和机器路径均排除。检查工作区、暂存与本次待推提交以及远端状态。命令使用相对路径与占位符脱敏。OBJ 位于发布附件，按坐标、三角面、来源和成员哈希回读验证。ZIP SHA-256 为 `7338f2602e5286f09e7b1876c74929c0bb0624d465ddbcfefde385d4243c7e45`。[附件检查](evidence/attachment_audit.json) · [发布前自检](evidence/prepush.json)
