# Gen 原始头部与下视双目标定核查

已提取整段头部候选位置、四元数、VIO 设备位姿与原始双目标定，并生成当前 20 对样本的本地接入资产。原文件有方向信息；此前“仅有位置”的精简 NPZ 没有保留它。**提取完成不等于已经得到可直接替换黄金 D 的 SMPL-X Head 真值**：头部旋转轴定义仍未验证，且原始标定和历史缓存使用的标定不同。

执行基线 `abe0ea1`。工作树已有其他轮次的 `train.py` 修改及未跟踪文件，本轮均未改动；本轮新代码属于基线上的工作树增量，随本报告提交。读取上轮 audit 的 brief/report/review，review 尚待 Claude；本轮 brief 见 [brief.md](brief.md)。只读源文件，未安装依赖、训练、运行新黄金 D 拟合或覆盖旧评测。

## 数据覆盖与时间对应

源文件标识与 SHA-256、全部主题计数及解析命令见 [extraction.json](evidence/extraction.json)。使用原文件内嵌 protobuf schema 和现有系统 libzstd 解析；检查块长度、非零 CRC，未额外下载数据。下视相机为原始 camera2 / camera3，对应旧评测命名 cam3 / cam4。

- 左图、右图、BODY22、VIO 各 5,379 条，左图首末跨度 179.266998 秒；时间单调。
- BODY22 与左图的 MCAP log_time 全部一致；右图相对左图最大偏差 0.045 ms；VIO 最大绝对偏差 0.000384 ms。逐条最近时间匹配也验证与序号匹配一致。
- 图像 protobuf 的 timestamp 未填，不能把它的零值用于配对；本轮使用 MCAP log_time。
- 所有人体消息都有 22 个有限位置和单位长度四元数。**这只能证明字段完整，不证明姿态准确。**

数值见 [alignment.json](evidence/alignment.json)，最近时间对应检查见 [validation.json](evidence/validation.json)。

还发现有效性标记需要分别保存：5,379 条 `frame_validity` 均为 true，但 `time_range_validity` 排除了录制头尾。旧 20 对中的第 0 对位于无效时间段，原因涉及其他相机覆盖和 IMU 边界；其余 19 对通过时间段检查。第 0 对仍完整导出并显式标记，不悄悄换样或删掉；这也不等于其下视双目图像本身坏了。后续严格测试应单列或排除该对，同时对所有模型使用相同清单。见 [export.json](evidence/export.json) 的逐对标记及 validation 中的时间段摘要。

## 头部位置、朝向分别是什么

1. **人体 Head 候选**：取人体消息 `poses[15]`（零基）。它和既有 BODY22/SMPL 顺序一致，官方代码的骨架连接包含 `(12,15)`；当前消息没有关节名称，因此不能声称获得了额外的、带名称的独立 Head GT。它在所有旧样本上的位置能通过一致刚体变换重建原 NPZ：只从首帧确定旧坐标变换，再配合逐帧 VIO，可重建其余样本，全部 880 个双目身体点最大残差约 `1.27e-12 mm`。这是同源数据对应检查，不是准确率。
2. **人体 orientation 字段**：按内嵌 `PosesInFrame` schema 的声明，以世界系四元数 `x,y,z,w` 原样保存，再输出按该声明转换到两虚拟相机的旋转矩阵。矩阵有限、正交与往返检查通过；仍未验证该生产程序实际写入的旋转是否遵循全局语义、以及它与 SMPL-X Head 局部轴是否一致。
3. **VIO 设备位姿**：官方说明明确它是 DAS-Ego 设备的 6D 位姿，原点是 IMU 几何中心。它不是人体 Head 关节位置，不能直接把两者当成同一个点。[官方数据说明](https://docs.genrobot.ai/guides/das-ego-data-introduction)。本段 DFM Head 候选与 VIO 原点平均相距 137.78 mm（该统计不等于固定解剖偏置，见 validation）。

旋转谨慎接入有实际证据：即使换到**同一个旧虚拟相机坐标系**，MCAP 头部四元数按 schema 解释后与旧 EgoSMPLX 网络 Head 旋转的平均夹角仍为 140.82°。这不是“网络朝向误差”的测量，可能包含坐标轴/关节旋转定义差异和估计偏差。测试直接全局解释及局部 FK 等候选，尚未得到足够证据消除语义歧义（完整统计见 alignment 的 orientation_hypotheses）。不能为了得到小夹角，用旧预测或评测人体姿态拟合一个修正旋转后宣称是真值。

资产明确设置 `ready_as_smplx_anchor=false`。公开的 [Gen 数据卡](https://huggingface.co/datasets/genrobot2025/Gen-HumanWholeBody) 说明人体轨迹由 DFM 重建；这里应称为 DFM/跟踪参考，不能当独立动作捕捉真值。

## 下视相机内参和外参

原图均为 1600 × 1300，`distortion_model=ds`。D 的含义为 `fx, fy, cx, cy, xi, alpha`，不能当普通径向畸变系数。

| 原始相机（旧命名） | fx | fy | cx | cy | xi | alpha |
|---|---:|---:|---:|---:|---:|---:|
| camera2（cam3，左） | 511.198967 | 513.310971 | 832.742062 | 648.045631 | -0.000594509 | 0.570042257 |
| camera3（cam4，右） | 511.794498 | 513.749120 | 750.168941 | 650.758649 | -0.000684119 | 0.570219564 |

完整精度的 K、D、T_b_c 在 [alignment.json](evidence/alignment.json)。原文件还含另一套低分辨率 camera_info_resize；本轮原图映射未使用它。

官方 proto 明确 `T_b_c` 顺序是 `[tx,ty,tz,qx,qy,qz,qw]`，方向为 camera → calibration base；官方解析程序还使用固定轴变换：

```text
R_BODY = [[0,0,-1], [1,0,0], [0,-1,0]]
T_world_from_raw_camera = T_world_from_vio @ R_BODY_4 @ T_b_c
T_world_from_virtual = T_world_from_raw_camera @ block_diag(R_virtual_from_raw.T, 1)
T_right_virtual_from_left_virtual = inverse(T_vio_from_right_virtual) @ T_vio_from_left_virtual
```

出处及固定工具包提交 `cf541b96fe9eff4cef50c7f3dcd44f6bdf0fc8a5` 见 [official_sources.json](evidence/official_sources.json)。官方脚本中有一条借助原生 `_cam2` 人体主题的交叉验证，本文件没有该主题，不能声称运行了该交叉验证。本轮验证的是上述公开转换链的数值自洽及与旧数据的差异。

旧虚拟图为 1280 × 720，主点 `(639.5,359.5)`，左右图分别转向 28.86693° / 28.54935°；旋转直接读取旧 NPZ 保存的 `R_virtual_from_gen`。左右原图首帧解码后按该模型重采样，与旧图的平均 8-bit 像素差为 1.99 / 1.84；支持同帧、同虚拟视角的对应，但不是逐像素完全一致，尚未证明差异仅来自解码或重采样实现。DS 投影/反投影往返最大约 `2.55e-13 px`。见 export。

## 发现旧标定与原始文件不一致

| 项目 | 本次 MCAP 原始标定 | 历史评测缓存 |
|---|---:|---:|
| 双目基线 | 59.88346 mm | 55.66274 mm |

两套相对外参相差 1.06604°，平移向量差 6.88889 mm。补齐官方固定轴变换后，单相机与旧缓存隐含标定仍相差：左 2.65088° / 16.06607 mm，右 1.62104° / 10.46561 mm。**单纯虚拟转向不会改变相机基线长度，不能把基线差归因于裁切或旋转。**

这不是帧号错位造成的猜测：旧数据与原始 BODY22、逐帧 VIO 的固定转换在全部选定帧上达到数值精度一致，但该转换隐含的相机外参不是 MCAP 内的外参。尚未找到旧生成器的原始标定来源，因此不武断判断哪套物理标定更准确。

只替换标定、保持同一世界人体轨迹时：

| 对旧缓存的变化 | 左虚拟相机 | 右虚拟相机 |
|---|---:|---:|
| 20 帧 Head 位置平均位移 | 17.53 mm | 11.58 mm |
| 身体参考投影平均变化 | 11.75 px | 6.50 px |

投影统计使用每视图各自旧参考在图内且新投影有效的全部 BODY22 点（左 324、右 323 个）；不是当前检测器 12 个评测点的误差，也不是标定准确率。证据在 export / validation。

因此后续采用原始标定时，应同时从世界坐标重建所有模型共同使用的 2D/3D 参考、Head 位置、双目外参和参考掩码。不能把新外参与旧缓存的参考点混用。既有四模型结果保留为“历史缓存标定协议”，本轮没有把它改名为原始标定结果。

## 交付物、复现与后续接入

本地（仓库数据盘内、Git 忽略）：

- `outputs/gen_head_camera_alignment_20261118/head_all_frames.npz`：整段 5,379 帧世界 Head 候选位置/四元数、VIO 设备位姿、两虚拟相机下的候选位置/旋转。未按有效时间段过滤。
- `outputs/gen_head_camera_alignment_20261118/head_camera_20pairs.json`：选定 20 对的时间戳、有效性、原始/虚拟内外参、头部候选值，以及显式分开的旧坐标兼容候选。未验证旋转有阻止直接接入的标记。
- 本地首帧重采样图及标定投影叠加图用于检查，未上传图像/原始数据。

两项主要资产的 SHA-256、大小、位置占位符和访问要求见 export 的 asset_manifest。公开仓库仅保留源码与汇总证据；没有资产访问权限者不能声称独立复现。

执行顺序（使用已有含 numpy / scipy / OpenCV / protobuf 的 Python 环境与系统 libzstd）：

```sh
python runs/2026-11-18_gen_head_camera_alignment/code/extract.py
python runs/2026-11-18_gen_head_camera_alignment/code/analyze.py
python runs/2026-11-18_gen_head_camera_alignment/code/export_assets.py
python runs/2026-11-18_gen_head_camera_alignment/code/validate.py
```

各脚本命令和退出状态记录在对应 evidence JSON 中。验证通过：源 MCAP 哈希不变、最近时间匹配、原始转换链位置往返、旋转 SO(3)、首帧固定旧标定对全部样本的重建、旧 DS 投影复算。公开路径中的本机前缀替换为 `<WORKSPACE>` / `<PYTHON_ENV>`；保留所有统计数值及未解决异常。早期不含 R_BODY 的外参假设结果也留在 alignment 中供复核，不能据此忽略后来确定的官方转换。

下一步应先确认 DFM orientation 的生产端定义或获得设备→SMPL-X Head 的独立标定。位置与相机参数已经具备重建输入的条件；头部方向尚不满足“可直接作为 SMPL-X 全局 Head 朝向”的验收条件，本轮没有冒充已通过，也没有启动使用未经验证朝向的重跑。

## 推送前自检

仅发布本轮代码、文档、精简证据；无权重、图片、数据集、视频、NPY/NPZ、完整原始输出、凭据、`.env`、完整 provenance 或 patch；文件均小于 50 MB。检查工作区、暂存内容、待推提交和远端状态，结果见 [prepush.json](evidence/prepush.json)。既有修改、未跟踪历史实验及用户 MCAP 保持原样且不暂存；本轮增量检查不代表历史仓库审计。
