# 运行报告：2026-11-04_archive-20261028

- 目标：归档用户指定的 10 月 28 日推理和同批后续结果，供 GitHub 审核。
- 状态：归档完成；未重新执行推理。DownSight 归档基线为 `21cdefc2a9cb58cf3dbc4072ec878742a4b90ad2`；工作树开始时已有其他任务的未提交改动，本次仅提交新归档。历史输入清单记录 EgoSmplx 提交 `269ec28d10b779045136b907d1fe2f9ee2acf0f7`；未独立复核历史运行环境。
- 样本和配置：左右视图逐张处理；[历史配置和结果目录](../../experiments/inference_20261028/README.md)、[输入记录](evidence/historical_inputs_manifest.json)、[原始摘要](evidence/historical_summaries.json)。附图为历史派生可视化，未修改。配置仅记录历史参数，不足以独立复现。
- 结果：用户图中的 0002、0003、0005 六个视图均未通过局部表面检查；0008 两个视图通过。后续 0100、0200 四个视图均未通过。RMSE 是相对自动目标点而非人工真值的误差；具体数值见摘要证据。历史检查的范围有限，不证明全身无碰撞。
- 原始数据与完整输出保留在 `<WORKSPACE>/EgoFishPose/cui_local_computer_files/inference_github_269ec28d_20261028`；资源标识、输入和图像 SHA-256、获取及访问方式见[资源清单](../../experiments/inference_20261028/resource_manifest.json)。模型、权重、标定与运行环境未完整锁定。
- 脱敏：配置和输入清单中的开发机用户目录替换为 `<WORKSPACE>`；保留数值、资源名称及异常。未上传完整 provenance 或 patch。
- 推送前自检：见 [检查记录](evidence/prepush_check.txt)。检查范围是本轮增量、暂存区、可见工作区及将推送的提交，不代表历史提交已全面审计。派生对比图按用户请求发布，未包含单独输入数据、权重或完整原始输出。
