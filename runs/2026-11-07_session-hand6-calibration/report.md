# 运行报告

- 目标：整理 100 张推理的三份标定文件并同步 GitHub。
- 基线：`b5d1f45bf5f946cdb375f67dd8c010720af2c7e8`，与开始时远端 main 一致；执行时工作树含已有未跟踪文件 `runs/2026-11-04_session-hand6-v2/evidence/point_comparison_followup.json`，未改动或提交。
- 输入：100 张实验清单中引用的 cam3、cam4 单相机 JSON，以及它们所标识的原双目标定 JSON。复制文件与源文件逐字节一致；JSON 格式、对应关系和相对位姿字段检查见 [证据](evidence/verification.txt)。没有运行推理或重新标定。
- 结果：见 [标定目录](../../data/calibration/README.md)。单相机文件的外参为单位矩阵；原双目标定 `poseCamera2` 提供相对位姿，单位为毫米。历史外参未在该序列验证。
- 脱敏：证据中的开发机路径替换为 `<WORKSPACE>`；不提交完整本地 provenance 或 patch。
- 推送前检查：见 [检查记录](evidence/prepush.txt)；检查本轮工作区、暂存内容、待推送提交与远端状态，不构成对已发布历史的审计。
