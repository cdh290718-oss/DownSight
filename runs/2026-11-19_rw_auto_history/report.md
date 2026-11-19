# RW20000 自动 12 点：补全逐帧历史起点对照

头随上胸、其他条件相同，采用逐帧历史朝向作人体起点，四帧平均的校准 15 点非 PA 为 **52.25 mm**，PA 为 **40.79 mm**。[逐帧结果](evidence/results_auto.json) · [下载四组实际 OBJ](https://github.com/cdh290718-oss/DownSight/releases/download/rw-s9-auto-history-20261119/rw_s9_auto_history_OBJ.zip)

## 对照

四种起点在同一批运行。共用 RW20000 自动身体 12 点、标注 Head 位置、相机、掩码和固定均值体型。历史组逐帧替换人体根旋转初值，与其他组一样用 Head 位置设置初始平移。拟合过程中绝对头朝向权重为零，局部颈和头旋转锁定。四组均按相同模型端映射的 15 个关节评测。非 PA 为左相机坐标内的平均欧氏距离；PA 做逐例相似对齐。[输入核查](evidence/inputs.json)

| 人体起点 | 校准 15 非 PA (mm) | 校准 15 PA (mm) | 共同 12 非 PA (mm) | 共同 12 重投影 (px) |
|---|---:|---:|---:|---:|
| 逐帧历史拟合 | 52.25 | 40.79 | 53.47 | 12.69 |
| 相机代理 | 51.85 | 40.64 | 53.37 | 12.69 |
| 统一固定 | 53.08 | 41.35 | 54.79 | 12.67 |
| 双肩和双髋 | 51.59 | 40.34 | 53.00 | 12.69 |

| 历史起点样本 | 15 点非 PA (mm) | 15 点 PA (mm) | 四组 OBJ 正侧面、左右目叠加 |
|---|---:|---:|---|
| body01 | 50.55 | 34.84 | [总览](assets/body01__auto.jpg) · [历史 OBJ 图](assets/body01/history__obj.jpg) · [左目](assets/body01/history__left.jpg) · [右目](assets/body01/history__right.jpg) |
| body02 | 48.49 | 32.40 | [总览](assets/body02__auto.jpg) · [历史 OBJ 图](assets/body02/history__obj.jpg) · [左目](assets/body02/history__left.jpg) · [右目](assets/body02/history__right.jpg) |
| body03 | 61.64 | 57.04 | [总览](assets/body03__auto.jpg) · [历史 OBJ 图](assets/body03/history__obj.jpg) · [左目](assets/body03/history__left.jpg) · [右目](assets/body03/history__right.jpg) |
| body04 | 48.31 | 38.86 | [总览](assets/body04__auto.jpg) · [历史 OBJ 图](assets/body04/history__obj.jpg) · [左目](assets/body04/history__left.jpg) · [右目](assets/body04/history__right.jpg) |

总览从上到下依次为历史、相机代理、固定、肩髋。三个重跑对照与上一轮归档的两项 15 点均值最大差为 0.003037 mm。[汇总和重跑差值](evidence/summary.json)

上轮对照表中，采用历史起点和官方 15 点输入时的非 PA / PA 为 40.21 / 34.96 mm；本轮补全同一行的自动 12 点输入列。与本轮肩髋起点相比，历史起点的非 PA 和 PA 分别高 0.66 和 0.45 mm；与统一固定相比则分别低 0.84 和 0.56 mm。相机代理重跑出现保留两位小数时的末位差异，完整精度已保留。[上一轮汇总](../2026-11-19_rw_position_only/evidence/summary.json)

## 历史拟合用的是哪个网络

旧流程由 **EgoSMPLX** 全图左目身体网络提供初始人体参数，配置记录的权重标识为 `output/train_5datasets_8gpu_camtrans_w10_12ep_fixed_20260809_20260809_163018/model_dump/snapshot_11.pth.tar`。已比对四帧的实际网络推理 manifest 与各自配置，并在本轮读取权重字节核对 SHA-256 `c7d30f04d13ed80f1941226458a43dfd50fc0ab0e5902701b6c9895e1f92a3b3`。[四帧模型来源](evidence/model.json) · [历史权重说明](../../docs/downsight/CHECKPOINT_zh.md)

本次复用该网络下游旧流程已拟合的 SMPL-X 参数。body01、02、04 取融合阶段，body03 取身体拟合阶段。将每帧根、三段脊柱、颈和头的旋转按骨架顺序相乘，得到的矩阵与官方 15 点历史初始化使用的缓存逐元素一致。RW20000 在本轮提供自动二维身体点。全过程复用历史推理结果，没有重新训练或重新推理旧网络。[逐帧参数来源和核验](evidence/inputs.json)

## 执行、限制与自检

基线为 `00cfb139fb5dfe800ba210fe84d0b9aca8d75be9`，在有既有无关改动的工作树中新增本轮文件，使用现有 CPU 环境。每个条件拟合 20,000 步，四个样本全部来自 S9。0 / 16 个结果通过原有严格收敛诊断。模型转换、输入来源、点掩码和关节语义沿用上轮，这些三维关节误差不能用作真实头部朝向精度。[执行记录](evidence/execution.json)

仅提交本轮精简证据、脚本和网格派生 JPG；完整原始输出、权重和数据留在开发机，开发机路径以相对标识和占位符脱敏。增量自检覆盖工作区、暂存、待推提交与远端状态；网格按既有要求通过 GitHub Release 发布。16 个 OBJ 的坐标、拓扑、渲染源和附件成员均已核对，附件 SHA-256 为 `d4e16bd64fffe937e88adae4ad87b454afdde0f3c8d9826470da8c7dd332c9c6`。[附件审计](evidence/attachment_audit.json) · [推送自检](evidence/prepush.json)
