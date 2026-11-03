# 冗余分支与工作树清理

用户确认后，已移除此前误建的独立工作树、删除本地 `experiment/gen-stereo-diagnostics-20261103` 分支，并删除 GitHub 同名远程分支。清理前核对该工作树没有未提交或忽略文件、提交已在 main 历史中、没有进程以该工作树为当前目录，远程分支与记录一致。

main 中 `experiments/gen_stereo_analysis_20261030_20261103/` 的实验报告及归档保留；没有改动旧 `EgoFishPose` 或另一项未跟踪实验。前后状态和引用核对见 [verification.txt](evidence/verification.txt)。没有运行模型或修改环境。

推送前自检：本轮仅提交本目录中的简短 Markdown 和文本证据；文件均小于 50 MB，没有权重、数据集、视频、`.npz` / `.npy`、完整原始输出、`.env`、完整 provenance 或 patch；未发现已知凭据模式或报告和日志中的未脱敏开发机路径。检查覆盖本次增量，未重新审计历史；模式扫描不能保证发现所有敏感内容。详见 [prepush-check.txt](evidence/prepush-check.txt)。
