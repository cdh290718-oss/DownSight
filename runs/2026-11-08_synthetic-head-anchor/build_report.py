"""Make the public report directly from evidence and completed stage records."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/synthetic_head_anchor_20261108'
read = lambda p: json.loads(p.read_text())
metrics = read(RUN / 'evidence/metrics.json')
config = read(RUN / 'settings.json')
execution = read(RUN / 'evidence/execution.json')
labels = ['original_rigid', 'original_body', 'original_final', 'anchor_only_rigid', 'requested_rigid', 'cycle4_rigid']
display = {'original_rigid': '原刚体', 'original_body': '原身体姿态', 'original_final': '原最终（含 MANO）',
           'anchor_only_rigid': '仅加头锚点的刚体', 'requested_rigid': '完整修改的刚体', 'cycle4_rigid': '交替优化末轮'}
lines = [
    '# 真值头部位置锚点与交替优化', '',
    '- 基线：`68fa5cb13d3dc4470b95b408000c8404448289ff`，开始时与 GitHub main 一致；原工作区已有无关未跟踪文件，未修改或提交。执行时新增本 run 的诊断脚本，原流水线代码、原结果、外部后端未改。',
    '- 数据为原来一个合成姿态的 cam3 / cam4，两视角独立优化；真值头位置是输入，因此结果是 Oracle 诊断，不是无真值的实际推理精度。数据提交及资源沿用[原实验](../2026-11-08_synthetic-original/report.md)。',
    '- 只改变身体拟合，网络观测直接复用。未重新运行 WiLoR/MANO 替换、接缝和后续引导；新结果手部为固定参数的原 SMPL-X 手，不能以此声称完整手部重建已经改善。原最终结果仅作额外参照；对应阶段的主要对照为原身体姿态。',
    '- 优化中唯一输入的真值是 native SMPL-X Head 关节（索引 15）的相机三维位置，不输入其朝向，也不输入真值姿态、体型、其他关节或可见性。不是整片真值头网格约束。', '',
    '[投影图比较](projection.md)', '',
    '**本轮是部分改善：头部和骨盆光轴深度明显更准，但骨盆 Y 方向误差增大、肘部精度下降，不能认定为全面优于原流程。** 只加入头部锚点、姿态保持固定时，腕肘误差会明显增大；交替优化恢复了其中一部分。', '',
    '## 固定设置', '',
    f"头部损失为 `0.01 * sum(((Head - target)/0.015)^2)`。1.5 cm 是本轮预先设定的误差尺度，不是测得的佩戴误差，也不是硬边界。肩点保留，径向 {config['shoulder_tolerance_px']:.0f} px 内不处罚，超过部分才进入 Smooth L1；肩组系数 {config['shoulder_group_weight']}，其他身体组与腕组各为 1，组内按分数平方归一化。容差与权重是首轮假设，未按评测结果调参。",
    '',
    '完整修改停用鼻点，不按真值入画性筛点；无 40 px 边缘排除。取消全网格 2 cm 硬/软限制，也没有沿用旧刚体头网格参考。仅要求参与重投影的关节点 z ≥ 0.1 mm 以保证数值有效，头部锚点直接做三维比较。没有设备实体模型，故本轮没有碰撞约束。', '',
    '体型、手、足部固定参数及表情保持原网络值；姿态增量范围沿用原身体阶段。保留原 0.85 倍腕距下限，以本轮完整修改刚体的腕距为参考，避免无关改动；整体平移 z 的原范围亦保留。初始刚体 1800 步；随后 4 轮，每轮固定整体位姿优化姿态 400 步，再固定姿态优化整体位姿 200 步。姿态步数合计与原身体阶段相同，但额外增加了刚体步数，不能把收益全部归因于交替策略。', '',
    '仅加头锚点的刚体对照保留原鼻、肩权重及原近深度软惩罚；完整修改从相同 raw 独立开始。因此该对照用来展示仅加入位置锚点的作用，不是对所有修改进行完整因素分解。', '',
    '## 结果', '',
    '所有误差以独立重建的保存参数计算；骨盆 z 误差为预测减真值，正值表示偏远。身体二维误差统一用渲染真值的 Brown–Conrady 相机评价，拟合仍用原多项式相机。二维统计仅限真值入画点，未检查遮挡；无效预测数量单独保留在证据中。', '',
    '| 视角 / 阶段 | Head 误差 mm | 骨盆 z 误差 mm | 身体25点 mm | 双腕均值 mm | 双肘均值 mm | 入画身体点二维均值 px |',
    '|---|---:|---:|---:|---:|---:|---:|',
]
assessment = []
for view in metrics['views']:
    for label in labels:
        s = view['stages'][label]
        lines.append('| %s / %s | %.2f | %+.2f | %.2f | %.2f | %.2f | %.2f |' % (
            view['side'], display[label], s['head_anchor_error_mm'], s['per_joint'][0]['error_xyz_mm'][2],
            s['body25_absolute_mean_mm'], s['wrists_absolute_mean_mm'], s['elbows_absolute_mean_mm'],
            s['body25_inframe_projection_px']['mean']))
    new = view['stages']['cycle4_rigid']
    old = view['stages']['original_body']
    final = view['stages']['original_final']
    assessment.append(dict(side=view['side'],
        absolute_pelvis_depth_error_change_vs_original_body_mm=abs(new['per_joint'][0]['error_xyz_mm'][2])-abs(old['per_joint'][0]['error_xyz_mm'][2]),
        wrist_mean_change_vs_original_body_mm=new['wrists_absolute_mean_mm']-old['wrists_absolute_mean_mm'],
        elbow_mean_change_vs_original_body_mm=new['elbows_absolute_mean_mm']-old['elbows_absolute_mean_mm'],
        wrist_mean_change_vs_original_final_mm=new['wrists_absolute_mean_mm']-final['wrists_absolute_mean_mm'],
        elbow_mean_change_vs_original_final_mm=new['elbows_absolute_mean_mm']-final['elbows_absolute_mean_mm']))
lines += ['', '## 每轮交替变化', '',
          '| 视角 / 阶段 | Head 误差 mm | 骨盆 z 误差 mm | 双腕均值 mm | 双肘均值 mm |', '|---|---:|---:|---:|---:|']
for view in metrics['views']:
    for label, s in view['stages'].items():
        if not (label.startswith('cycle') or label == 'requested_rigid'):
            continue
        lines.append('| %s / %s | %.2f | %+.2f | %.2f | %.2f |' % (view['side'], label,
            s['head_anchor_error_mm'], s['per_joint'][0]['error_xyz_mm'][2], s['wrists_absolute_mean_mm'], s['elbows_absolute_mean_mm']))
lines += ['', '逐关节误差、肩部投影宽度、头部入画数、SMPL-X 手指指标及全部阶段见 `evidence/metrics.json`。', '',
          '## 腕肘逐点对照', '',
          '| 视角 / 关节 | 原身体姿态 mm | 原最终 mm | 交替优化末轮 mm |', '|---|---:|---:|---:|']
for view in metrics['views']:
    for index in [10, 11, 12, 13]:
        joint = view['stages']['cycle4_rigid']['per_joint'][index]['name']
        values = [view['stages'][label]['per_joint'][index]['absolute_mm'] for label in ['original_body', 'original_final', 'cycle4_rigid']]
        lines.append('| %s / %s | %.2f | %.2f | %.2f |' % (view['side'], joint, *values))
lines += ['', '均值不能替代逐侧检查：例如 cam3 左腕变差，即使双腕均值相对原身体阶段有所下降。因此本轮没有达到“腕肘都不明显变差”的目标。', '',
          '## 骨盆三维误差与边界', '',
          '| 视角 / 阶段 | 骨盆 X 误差 mm | Y 误差 mm | Z 误差 mm | 骨盆三维误差 mm |',
          '|---|---:|---:|---:|---:|']
for view in metrics['views']:
    for label in ['original_body', 'original_final', 'cycle4_rigid']:
        s = view['stages'][label]
        xyz = s['per_joint'][0]['error_xyz_mm']
        lines.append('| %s / %s | %+.2f | %+.2f | %+.2f | %.2f |' % (view['side'], display[label], *xyz, s['pelvis_absolute_error_mm']))
lines += ['',
          'Head 位置准确并不确定头部朝向、躯干方向或骨盆相对头部的位置。骨盆 Y 误差增大说明本轮尚未恢复正确的整体身体关系；剩余误差不能只用固定体型偏小解释。头到骨盆向量、姿态增量和约束状态均在证据中。多个髋、躯干、颈或锁骨姿态增量到达原上限；这是可能限制继续纠正的线索，尚未做解除上限的因果对照。', '',
          '本轮按预设预算完成 4 轮，目标仍在下降，没有宣称已达到充分收敛或全局最优。头部锚点、肩容差、损失分组和交替策略的效应尚未完全分离；图像中手指变化亦不能与原 MANO 手作同算法比较。', '',
          '## 数值核对与失败', '',
          '每一交替块仅保存同一目标下的最低损失可行状态；逐块核查固定参数未变化、目标不增大。保存参数独立通过 SMPL-X 前向重建；只对实际进入目标的关节点做像素重载核对，画外/镜头后点不用于该数值检查，三维网格重载仍检查全部顶点。详见 `evidence/verification.json`。', '',
          '首次左目尝试因 native joint tuple 索引方式错误退出，且启动器当时未创建 evidence 目录而未能写退出记录；第二次尝试完成头锚刚体后因数值保护的原地张量操作导致梯度错误退出，启动器同样在目录创建前结束。两次原日志和退出记录均保留为 `evidence/attempt*`；随后修正索引、非原地数值保护和目录初始化，新增梯度检查，保持目标、权重和步数不变后重跑。右目在这两次启动器失败中均未开始。', '',
          '## 复现与发布检查', '',
          '在原模型环境、原数据及观测可访问时，设置 `PYTHONPATH=<REPO>`，运行 `test_losses.py`、`run.py`、`evaluate.py`、`verify.py`、`build_report.py`。CPU 使用四线程，不安装依赖或修改外部后端。`run.py` 使用原 body Python，模型权重和输入资源获取要求见原实验；本轮代码 SHA-256、输入摘要和配置保存在证据中。各阶段直接调用模型，不是完整流水线执行，因此不生成完整 pipeline provenance。', '',
          '完整 NPZ、原始日志和中间参数留在仓库忽略的输出目录；只发布派生投影图、精简阶段日志和数值证据。开发机路径用 `<WORKSPACE>`、`<SYSTEM>` 等占位符脱敏。推送前核查工作区、暂存、待推送提交和远端；结果见 `evidence/prepush.json`。不把本次增量检查视为历史全量审计。首次暂存检查因 projection.md 文件末尾多余空行失败，未提交或推送；修正生成器后重新检查，记录见 `evidence/prepush_attempt1.txt`。']
(RUN / 'report.md').write_text('\n'.join(lines) + '\n')
(RUN / 'evidence/assessment.json').write_text(json.dumps(assessment, indent=2) + '\n')
page = ['# 真值头部锚点：投影比较', '',
        '绿色点为真值关节，红点为预测；青色为身体网格，紫色为头部。新结果是身体阶段的 SMPL-X 网格；原最终图含 MANO 替换。所有图统一采用真值 Brown–Conrady 投影。近面交叉或过长三角形会被省略，不能把显示缺口解释为模型本身有孔洞。', '',
        '[数值报告](report.md)', '']
for view in metrics['views']:
    page += ['## ' + view['side'], '',
             '| 原身体姿态阶段 | 头部锚点＋交替优化 |', '|---|---|',
             '| ![](' + view['stages']['original_body']['image'] + ') | ![](' + view['stages']['cycle4_rigid']['image'] + ') |', '',
             '真值', '', '![](' + view['truth_image'] + ')', '']
    for label in ['original_rigid', 'anchor_only_rigid', 'requested_rigid', 'original_final']:
        page += [display[label], '', '![](' + view['stages'][label]['image'] + ')', '']
(RUN / 'projection.md').write_text('\n'.join(page).rstrip() + '\n')
# Publish compact per-stage diagnostics, not local path-bearing provenance.
for side in ['left', 'right']:
    for name in ['inputs', 'stages']:
        payload = read(OUT / side / (name + '.json'))
        (RUN / 'evidence' / (side + '_' + name + '.json')).write_text(json.dumps(payload, indent=2) + '\n')
print('report and projections indexed')
