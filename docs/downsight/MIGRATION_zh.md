# 迁移状态

- 基线：公开 `cdh290718-oss/EgoSmplx` 提交 `269ec28d10b779045136b907d1fe2f9ee2acf0f7` 的跟踪文件。`DownSight` 是独立 Git 工作区。
- backend：当前配置默认仍需单独部署现有 EgoSMPLX 后端，仓库没有打包它，也未验证能完全摆脱旧工作路径；应依 `docs/backend_contract.json` 核对文件和资源。Sapiens2 / WiLoR 和模型文件不在仓库中。
- 未迁移：10月工程加速的未提交代码和 body10、Gen 等历史变体尚待逐 patch 审查；其效果不能因为文件存在而认定已验证。
- 新增探索记录：`experiments/stereo_head_prior_20261103` 仅归档部分脚本与报告，不作为默认模块。
- 待完成：固定回归输入与权重，补齐后端资源与版本，执行缓存及端到端回归，逐个把适合的历史实验重构成默认关闭的模块，核对完整数据迁移，再将旧目录退出运行路径。不能声称全部历史资料或训练结果已迁移。
- 历史 pipeline 目录和相对基线的 Python 文件差异数见 `historical-pipeline-inventory.csv`；文件路径只标识原开发机历史位置。其他后端运行快照、打包文件和新增非 Python 文件未完整纳入。
- 候选黄金样本及尚未验证的公差见 `GOLDEN_REGRESSION_zh.md`；不要将其中的跨环境候选公差视为已经验证。
