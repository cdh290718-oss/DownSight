# session_hand6 的 100 张推理相机标定

该目录保存 100 张 head01 推理所引用的原始标定文件，未修改文件内容。原实验中 cam3、cam4 的图像分别独立推理。

- `cam3_left_lower.egosmplx.json`：左下 cam3 的内参和鱼眼投影多项式，对应原双目标定的 `camera1`。
- `cam4_right_lower.egosmplx.json`：右下 cam4 的内参和鱼眼投影多项式，对应原双目标定的 `camera2`。
- `stereoParams_20260604_fixed_native.json`：原双目标定，包含两个相机的参数和 `poseCamera2` 相对位姿；`poseCamera2.A` 是 4×4 矩阵，平移单位为毫米。

两个单相机 JSON 的 `extrinsic` 均为单位矩阵，不表示相机之间的相对位姿。输入图像尺寸应与单相机标定中的 1280×720 一致。历史双目外参尚未在 session_hand6 序列上验证；参见 [双目实验报告](../../runs/2026-11-04_session-hand6-v2/report.md)。

100 张推理的背景及采样说明见 [head01 文档](../../docs/HEAD_REFERENCE_01_zh.md)。文件摘要见 [本轮证据](../../runs/2026-11-07_session-hand6-calibration/evidence/verification.txt)。
