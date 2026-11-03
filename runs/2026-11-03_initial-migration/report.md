# 初始迁移与验证

- 目标：建立 DownSight 独立工作区，使用公开 EgoSmplx 基线，建立后续双端协作入口。
- 状态：初始基线已整理并推送；历史实验和后端迁移尚未完成。
- 被测代码提交：`fce5e3c0deaaf902d64864a41727805a280d27bf`。
- 原始基线：`cdh290718-oss/EgoSmplx` 的 `269ec28d10b779045136b907d1fe2f9ee2acf0f7`。
- 测试：开发机现有 egosmplx Python，`PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider`；27 passed。未运行神经网络、黄金样本或完整端到端推理。
- 新增：单次运行的初步 provenance、工作流程、报告模板、一个双目头部先验探索脚本与报告摘录。
- 限制：模型后端与权重仍在仓库外；未完成所有历史源码副本差异审查及迁移。双目实验摘录不是独立可运行流程。provenance 的后端、环境、产物和续跑覆盖仍不完整，详见 `docs/downsight/WORKFLOW_zh.md`。
- 下一步：锁定实际后端版本与资源，逐一审查历史源码差异；冻结黄金样本后，执行缓存路线和原生推理验证，再迁移其他实验模块。
