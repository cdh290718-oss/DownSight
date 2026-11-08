# 下视鱼眼相机设备标定

- `cam3_left_lower.egosmplx.json`：左下 cam3 的内参和鱼眼投影多项式，对应双目标定的 `camera1`。
- `cam4_right_lower.egosmplx.json`：右下 cam4 的内参和鱼眼投影多项式，对应双目标定的 `camera2`。
- `stereoParams_20260604_fixed_native.json`：双目标定，包含两个相机的参数及 `poseCamera2` 相对位姿；`poseCamera2.A` 是 4×4 矩阵，平移单位为毫米。

两个单相机文件的 `extrinsic` 均为单位矩阵，不表示两相机之间的相对位姿。标定图像尺寸为 1280×720。历史双目外参尚未在 session_hand6 序列上验证；参见 [双目实验报告](../../runs/2026-11-04_session-hand6-v2/report.md)。
