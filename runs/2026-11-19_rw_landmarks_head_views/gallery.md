# 四对 RW 的实际网格与原图叠加

[下载全部 OBJ](https://github.com/cdh290718-oss/DownSight/releases/download/rw-s9-landmarks-head-20261119/rw_s9_body01-04_landmarks_head_OBJ.zip) · [实验报告](report.md)

每张总览按行显示拟合条件，每行从左到右为同一 OBJ 的两个正交视角、左目叠加和右目叠加。下表可打开完整分辨率的各个视图。叠加中粉色为头、青色为身体，只计算人体自身遮挡，未重建房间深度。OBJ 以左相机 CV 坐标存储，单位是米，X 向右、Y 向下、Z 向前。文件按 body01–04 分目录，文件名中的条件对应图表。

| 对照 | body01 | body02 | body03 | body04 |
|---|---|---|---|---|
| 官方二维点，统一历史初始化 | [总览](assets/body01__current.jpg) | [总览](assets/body02__current.jpg) | [总览](assets/body03__current.jpg) | [总览](assets/body04__current.jpg) |
| 官方二维点，相机代理初始化 | [总览](assets/body01__camera.jpg) | [总览](assets/body02__camera.jpg) | [总览](assets/body03__camera.jpg) | [总览](assets/body04__camera.jpg) |
| RW20000 检测点，历史朝向先验 | [总览](assets/body01__automatic.jpg) | [总览](assets/body02__automatic.jpg) | [总览](assets/body03__automatic.jpg) | [总览](assets/body04__automatic.jpg) |
| RW20000 检测点，相机代理初始化 | [总览](assets/body01__automatic_head.jpg) | [总览](assets/body02__automatic_head.jpg) | [总览](assets/body03__automatic_head.jpg) | [总览](assets/body04__automatic_head.jpg) |
| 旧 D，原三角化输入与原体型设置 | [总览](assets/body01__old.jpg) | [总览](assets/body02__old.jpg) | [总览](assets/body03__old.jpg) | [总览](assets/body04__old.jpg) |

## 官方二维点，统一历史初始化

| 拟合条件 | body01 | body02 | body03 | body04 |
|---|---|---|---|---|
| 12 点，历史朝向先验 | [网格](assets/body01/baseline__obj.jpg) · [左](assets/body01/baseline__left.jpg) · [右](assets/body01/baseline__right.jpg) | [网格](assets/body02/baseline__obj.jpg) · [左](assets/body02/baseline__left.jpg) · [右](assets/body02/baseline__right.jpg) | [网格](assets/body03/baseline__obj.jpg) · [左](assets/body03/baseline__left.jpg) · [右](assets/body03/baseline__right.jpg) | [网格](assets/body04/baseline__obj.jpg) · [左](assets/body04/baseline__left.jpg) · [右](assets/body04/baseline__right.jpg) |
| 15 点，历史朝向先验 | [网格](assets/body01/15history__obj.jpg) · [左](assets/body01/15history__left.jpg) · [右](assets/body01/15history__right.jpg) | [网格](assets/body02/15history__obj.jpg) · [左](assets/body02/15history__left.jpg) · [右](assets/body02/15history__right.jpg) | [网格](assets/body03/15history__obj.jpg) · [左](assets/body03/15history__left.jpg) · [右](assets/body03/15history__right.jpg) | [网格](assets/body04/15history__obj.jpg) · [左](assets/body04/15history__left.jpg) · [右](assets/body04/15history__right.jpg) |
| 12 点，相机软先验 | [网格](assets/body01/12camera__obj.jpg) · [左](assets/body01/12camera__left.jpg) · [右](assets/body01/12camera__right.jpg) | [网格](assets/body02/12camera__obj.jpg) · [左](assets/body02/12camera__left.jpg) · [右](assets/body02/12camera__right.jpg) | [网格](assets/body03/12camera__obj.jpg) · [左](assets/body03/12camera__left.jpg) · [右](assets/body03/12camera__right.jpg) | [网格](assets/body04/12camera__obj.jpg) · [左](assets/body04/12camera__left.jpg) · [右](assets/body04/12camera__right.jpg) |
| 15 点，相机软先验 | [网格](assets/body01/15camera__obj.jpg) · [左](assets/body01/15camera__left.jpg) · [右](assets/body01/15camera__right.jpg) | [网格](assets/body02/15camera__obj.jpg) · [左](assets/body02/15camera__left.jpg) · [右](assets/body02/15camera__right.jpg) | [网格](assets/body03/15camera__obj.jpg) · [左](assets/body03/15camera__left.jpg) · [右](assets/body03/15camera__right.jpg) | [网格](assets/body04/15camera__obj.jpg) · [左](assets/body04/15camera__left.jpg) · [右](assets/body04/15camera__right.jpg) |
| 12 点，头随胸廓 | [网格](assets/body01/12torso__obj.jpg) · [左](assets/body01/12torso__left.jpg) · [右](assets/body01/12torso__right.jpg) | [网格](assets/body02/12torso__obj.jpg) · [左](assets/body02/12torso__left.jpg) · [右](assets/body02/12torso__right.jpg) | [网格](assets/body03/12torso__obj.jpg) · [左](assets/body03/12torso__left.jpg) · [右](assets/body03/12torso__right.jpg) | [网格](assets/body04/12torso__obj.jpg) · [左](assets/body04/12torso__left.jpg) · [右](assets/body04/12torso__right.jpg) |
| 15 点，头随胸廓 | [网格](assets/body01/15torso__obj.jpg) · [左](assets/body01/15torso__left.jpg) · [右](assets/body01/15torso__right.jpg) | [网格](assets/body02/15torso__obj.jpg) · [左](assets/body02/15torso__left.jpg) · [右](assets/body02/15torso__right.jpg) | [网格](assets/body03/15torso__obj.jpg) · [左](assets/body03/15torso__left.jpg) · [右](assets/body03/15torso__right.jpg) | [网格](assets/body04/15torso__obj.jpg) · [左](assets/body04/15torso__left.jpg) · [右](assets/body04/15torso__right.jpg) |

## 官方二维点，相机代理初始化

| 拟合条件 | body01 | body02 | body03 | body04 |
|---|---|---|---|---|
| 15 点，固定相机相对朝向 | [网格](assets/body01/camera_lock__obj.jpg) · [左](assets/body01/camera_lock__left.jpg) · [右](assets/body01/camera_lock__right.jpg) | [网格](assets/body02/camera_lock__obj.jpg) · [左](assets/body02/camera_lock__left.jpg) · [右](assets/body02/camera_lock__right.jpg) | [网格](assets/body03/camera_lock__obj.jpg) · [左](assets/body03/camera_lock__left.jpg) · [右](assets/body03/camera_lock__right.jpg) | [网格](assets/body04/camera_lock__obj.jpg) · [左](assets/body04/camera_lock__left.jpg) · [右](assets/body04/camera_lock__right.jpg) |
| 15 点，头随胸廓 | [网格](assets/body01/torso_caminit__obj.jpg) · [左](assets/body01/torso_caminit__left.jpg) · [右](assets/body01/torso_caminit__right.jpg) | [网格](assets/body02/torso_caminit__obj.jpg) · [左](assets/body02/torso_caminit__left.jpg) · [右](assets/body02/torso_caminit__right.jpg) | [网格](assets/body03/torso_caminit__obj.jpg) · [左](assets/body03/torso_caminit__left.jpg) · [右](assets/body03/torso_caminit__right.jpg) | [网格](assets/body04/torso_caminit__obj.jpg) · [左](assets/body04/torso_caminit__left.jpg) · [右](assets/body04/torso_caminit__right.jpg) |

## RW20000 检测点，历史朝向先验

| 拟合条件 | body01 | body02 | body03 | body04 |
|---|---|---|---|---|
| 12 个检测点 | [网格](assets/body01/auto12__obj.jpg) · [左](assets/body01/auto12__left.jpg) · [右](assets/body01/auto12__right.jpg) | [网格](assets/body02/auto12__obj.jpg) · [左](assets/body02/auto12__left.jpg) · [右](assets/body02/auto12__right.jpg) | [网格](assets/body03/auto12__obj.jpg) · [左](assets/body03/auto12__left.jpg) · [右](assets/body03/auto12__right.jpg) | [网格](assets/body04/auto12__obj.jpg) · [左](assets/body04/auto12__left.jpg) · [右](assets/body04/auto12__right.jpg) |
| 15 个检测点 | [网格](assets/body01/auto15__obj.jpg) · [左](assets/body01/auto15__left.jpg) · [右](assets/body01/auto15__right.jpg) | [网格](assets/body02/auto15__obj.jpg) · [左](assets/body02/auto15__left.jpg) · [右](assets/body02/auto15__right.jpg) | [网格](assets/body03/auto15__obj.jpg) · [左](assets/body03/auto15__left.jpg) · [右](assets/body03/auto15__right.jpg) | [网格](assets/body04/auto15__obj.jpg) · [左](assets/body04/auto15__left.jpg) · [右](assets/body04/auto15__right.jpg) |

## RW20000 检测点，相机代理初始化

| 拟合条件 | body01 | body02 | body03 | body04 |
|---|---|---|---|---|
| 12 点，固定相机相对朝向 | [网格](assets/body01/auto12_camera__obj.jpg) · [左](assets/body01/auto12_camera__left.jpg) · [右](assets/body01/auto12_camera__right.jpg) | [网格](assets/body02/auto12_camera__obj.jpg) · [左](assets/body02/auto12_camera__left.jpg) · [右](assets/body02/auto12_camera__right.jpg) | [网格](assets/body03/auto12_camera__obj.jpg) · [左](assets/body03/auto12_camera__left.jpg) · [右](assets/body03/auto12_camera__right.jpg) | [网格](assets/body04/auto12_camera__obj.jpg) · [左](assets/body04/auto12_camera__left.jpg) · [右](assets/body04/auto12_camera__right.jpg) |
| 12 点，头随胸廓 | [网格](assets/body01/auto12_torso__obj.jpg) · [左](assets/body01/auto12_torso__left.jpg) · [右](assets/body01/auto12_torso__right.jpg) | [网格](assets/body02/auto12_torso__obj.jpg) · [左](assets/body02/auto12_torso__left.jpg) · [右](assets/body02/auto12_torso__right.jpg) | [网格](assets/body03/auto12_torso__obj.jpg) · [左](assets/body03/auto12_torso__left.jpg) · [右](assets/body03/auto12_torso__right.jpg) | [网格](assets/body04/auto12_torso__obj.jpg) · [左](assets/body04/auto12_torso__left.jpg) · [右](assets/body04/auto12_torso__right.jpg) |
| 15 点，固定相机相对朝向 | [网格](assets/body01/auto15_camera__obj.jpg) · [左](assets/body01/auto15_camera__left.jpg) · [右](assets/body01/auto15_camera__right.jpg) | [网格](assets/body02/auto15_camera__obj.jpg) · [左](assets/body02/auto15_camera__left.jpg) · [右](assets/body02/auto15_camera__right.jpg) | [网格](assets/body03/auto15_camera__obj.jpg) · [左](assets/body03/auto15_camera__left.jpg) · [右](assets/body03/auto15_camera__right.jpg) | [网格](assets/body04/auto15_camera__obj.jpg) · [左](assets/body04/auto15_camera__left.jpg) · [右](assets/body04/auto15_camera__right.jpg) |
| 15 点，头随胸廓 | [网格](assets/body01/auto15_torso__obj.jpg) · [左](assets/body01/auto15_torso__left.jpg) · [右](assets/body01/auto15_torso__right.jpg) | [网格](assets/body02/auto15_torso__obj.jpg) · [左](assets/body02/auto15_torso__left.jpg) · [右](assets/body02/auto15_torso__right.jpg) | [网格](assets/body03/auto15_torso__obj.jpg) · [左](assets/body03/auto15_torso__left.jpg) · [右](assets/body03/auto15_torso__right.jpg) | [网格](assets/body04/auto15_torso__obj.jpg) · [左](assets/body04/auto15_torso__left.jpg) · [右](assets/body04/auto15_torso__right.jpg) |

## 旧 D，原三角化输入与原体型设置

| 拟合条件 | body01 | body02 | body03 | body04 |
|---|---|---|---|---|
| 历史朝向权重 | [网格](assets/body01/old_history__obj.jpg) · [左](assets/body01/old_history__left.jpg) · [右](assets/body01/old_history__right.jpg) | [网格](assets/body02/old_history__obj.jpg) · [左](assets/body02/old_history__left.jpg) · [右](assets/body02/old_history__right.jpg) | [网格](assets/body03/old_history__obj.jpg) · [左](assets/body03/old_history__left.jpg) · [右](assets/body03/old_history__right.jpg) | [网格](assets/body04/old_history__obj.jpg) · [左](assets/body04/old_history__left.jpg) · [右](assets/body04/old_history__right.jpg) |
| 朝向权重乘 0.1 | [网格](assets/body01/old_weak__obj.jpg) · [左](assets/body01/old_weak__left.jpg) · [右](assets/body01/old_weak__right.jpg) | [网格](assets/body02/old_weak__obj.jpg) · [左](assets/body02/old_weak__left.jpg) · [右](assets/body02/old_weak__right.jpg) | [网格](assets/body03/old_weak__obj.jpg) · [左](assets/body03/old_weak__left.jpg) · [右](assets/body03/old_weak__right.jpg) | [网格](assets/body04/old_weak__obj.jpg) · [左](assets/body04/old_weak__left.jpg) · [右](assets/body04/old_weak__right.jpg) |
| 关掉朝向项 | [网格](assets/body01/old_off__obj.jpg) · [左](assets/body01/old_off__left.jpg) · [右](assets/body01/old_off__right.jpg) | [网格](assets/body02/old_off__obj.jpg) · [左](assets/body02/old_off__left.jpg) · [右](assets/body02/old_off__right.jpg) | [网格](assets/body03/old_off__obj.jpg) · [左](assets/body03/old_off__left.jpg) · [右](assets/body03/old_off__right.jpg) | [网格](assets/body04/old_off__obj.jpg) · [左](assets/body04/old_off__left.jpg) · [右](assets/body04/old_off__right.jpg) |
