# 2026-10-28：左右视图逐帧推理归档

这是历史结果整理，本次没有重新推理。历史 [输入清单](../../runs/2026-11-04_archive-20261028/evidence/historical_inputs_manifest.json)记录使用 [EgoSmplx 269ec28d](https://github.com/cdh290718-oss/EgoSmplx/tree/269ec28d10b779045136b907d1fe2f9ee2acf0f7)，不能将结果视为当前 DownSight 流程的验证。

- [用户提供的八张图：四阶段对比](eight_stage_comparisons.jpg)：左右视图各四张，编号 0002、0003、0005、0008。
- [同批后续四张图：四阶段对比](four_stage_comparisons_0100_0200.jpg)：左右视图各两张，编号 0100、0200。
- [逐帧原始摘要](../../runs/2026-11-04_archive-20261028/evidence/historical_summaries.json)与[本次归档报告](../../runs/2026-11-04_archive-20261028/report.md)。
- [历史配置](configs/)已将开发机路径根替换为 `<WORKSPACE>`，是记录而非可直接运行的配置。[资源清单](resource_manifest.json)含输入及汇总图哈希和获取位置。

四列依次为 Raw EgoSMPLX、Sapiens2 两阶段身体拟合、MANO 引导的身体拟合、引导身体加原始接缝。历史摘要的误差是对 Sapiens2 和 WiLoR 自动目标点的像素误差，不是对人工真值的精度。用户图中 0002、0003、0005 的六个视图均未通过局部表面检查；0008 两个视图通过。0100 和 0200 的四个视图均未通过。局部检查不构成全身碰撞验证。

输入图、权重、模型、完整原始输出仍在开发机，只分享派生对比图与精简证据。旧存档根位置为 `<WORKSPACE>/EgoFishPose/cui_local_computer_files/inference_github_269ec28d_20261028`；需向数据持有人申请访问。原始逐帧网格和报告位于 `work/<帧名>/`，索引为 `index.html`。历史配置记录部分权重哈希，但没有证明全部权重、标定、后端版本和运行环境；本归档不足以独立复现。
