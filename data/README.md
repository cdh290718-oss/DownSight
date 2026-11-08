# 人体网格与下视鱼眼相机标定

- `meshes/`：最终融合人体网格，文件名为对应帧 ID，使用 MeshLab 的 **File → Import Mesh** 打开 `.obj`。这些网格来自同一段双相机视频的采样，不是独立场景；是身体和手部融合后的几何网格，不是一组完整的标准 SMPL-X 参数。
- `calibration/`：下视鱼眼相机设备的标定，说明见 [calibration/README.md](calibration/README.md)。

帧 ID、文件摘要、大小和源运行目录见 [网格清单](../runs/2026-11-08_publish-meshes/evidence/meshes.json)。网格对应各帧的最终运行结果；两帧使用成功的重试结果。
