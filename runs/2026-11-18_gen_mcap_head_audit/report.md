# Gen MCAP：头部朝向字段核查

用户提供的原始 MCAP 只读检查完成。文件哈希与摘要、主题及第一条样例见 [inspection.json](evidence/inspection.json)。MCAP 的 `robotx/motioncapture/human_pose_fuse_smooth_concat` 使用 `foxglove.PosesInFrame`，包含世界坐标 frame 的 22 个 Pose，首条样例每个 Pose 都有单位长度 orientation 四元数；按本项目先前假设的 SMPL 常见顺序，第 15 项是 Head 候选，但消息字段没有名称，索引尚未由原始规范证实。该候选有非空四元数，但四元数的坐标轴、旋转语义及与 SMPL-X Head 的对应关系尚未验证。

文件还包含 `/robot0/vio/eef_pose` 的带四元数 `PoseInFrame`（首条 frame_id 为 world）、相对姿态、IMU，以及 camera2/camera3 的 CameraCalibration（含 7 项 T_b_c）；仅凭字段名无法断定 VIO 的 eef 就是头部、T_b_c 的方向／分量顺序，或这些数据是真实人体头部朝向独立真值。

这说明上轮本地精简 NPZ 未保留的方向信息在原始文件中可找到候选；不能断言原始数据无朝向。当前结果仍是 DFM/跟踪系统的参考或估计，除非有独立外部标定依据，不应称为精确人工真值。只抽看每主题首个消息，尚未验证整段覆盖、时间同步、四元数变化、头部索引与相机坐标及虚拟转向的转换，不能直接替换既有实验 Head。

源文件 SHA-256 与大小见 inspection.json；解析器基于 MCAP 结构、现场内嵌 protobuf schema 与已安装库，未安装依赖、修改源文件或运行新拟合。解析出的完整消息与图像留在忽略输出，仅发布字段、统计与一项候选四元数以支持复核；不包含身份、图片或原始输出。执行基线 e2919cd，既有未跟踪工作未改。

## 推送前自检

只包含代码、报告和精简证据；无图像、数据集、视频、权重、NPY/NPZ、完整原始输出、`.env`、凭据、完整 provenance 或 patch；文件小于 50 MB，开发机路径脱敏。检查工作区、暂存与待推提交和远端，见 [prepush.json](evidence/prepush.json)；未审计历史。
