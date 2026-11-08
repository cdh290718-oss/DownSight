"""Generate reviewable tables and projection page from measured evidence."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];RUN=Path(__file__).resolve().parent;OUT=ROOT/'outputs/synthetic_head_se3_contour_20261108'
read=lambda p:json.loads(p.read_text());metrics=read(RUN/'evidence/metrics.json');prep=read(RUN/'evidence/preparation.json')
labels=['original_body','original_final','previous_position_alternating','se3_joint','se3_contour_joint']
display={'original_body':'原身体阶段','original_final':'原最终（MANO）','previous_position_alternating':'上轮仅位置＋交替','head_se3_initial':'Head完整位姿对齐初始化','se3_joint':'完整位姿＋联合优化','se3_contour_joint':'完整位姿＋轮廓＋联合优化'}
lines=['# 真值 Head 完整位姿与 Sapiens 躯干轮廓', '',
 '- 基线及实际执行时 HEAD：`70d2d8554e53083e76d1d1eef49ff6022648969d`，开始时与远端 main 一致。执行使用本轮尚未提交的新增脚本；运行版本哈希见 `evidence/execution.json`。工作树不是完整干净状态：本轮新增内容及原有无关未跟踪文件均存在；后者未改动或提交。本轮只新增 run 内脚本和结果，未改动原流水线或外部后端。',
 '- 数据仍是同一个合成姿态的 cam3/cam4（left=cam3，right=cam4），两目分别优化。输入模型和原自动观测复用[原实验](../2026-11-08_synthetic-original/report.md)；不重新推理，不输入真值体型、身体姿态、其他关节或真值轮廓。唯一优化真值是 Head 的完整位姿。本实验是 Oracle，不能等同实测设备上的精度。',
 '- 体型、手指和面部参数固定。本轮只运行身体拟合，未重跑 MANO 手替换和接缝；原最终结果为额外参考，主要阶段对照为原身体阶段及上轮身体优化结果。', '',
 '[查看投影与实际轮廓目标](projection.md)', '',
 '## 有效轮廓', '',
 '**实际启用的是躯干轮廓；两目的腿部轮廓均因分割支持不足关闭。** 这是输入质量限制，不是把缺失观测解释为零误差。', '',
 '| 视角 / 组 | 原语义像素数 | 置信度及排除后像素数 | 有效外边缘候选点 | 启用 |', '|---|---:|---:|---:|---|']
for row in prep['records']:
 for name,q in row['targets'].items():lines.append('| %s / %s | %d | %d | %d | %s |'%(row['side'],name,q['class_pixels'],q['eligible_pixels'],q['boundary_points'],q['active']))
lines += ['',
 '躯干取 Sapiens Torso 与 Upper_Clothing（22/23）并集。右目上衣标签包含袖子，因此用 Sapiens 的肩—肘—腕线段构造 60 px 半径排除区；只保留原分割中直接与背景相邻的轮廓，排除区本身不制造新边缘。腿部仅取上下腿标签（8/12/17/21），不把脚、鞋或衣服误当作可靠腿轮廓。概率阈值 0.55、至少 256 个有效语义像素和 32 个外边缘候选；每组均匀抽取最多 256 点。所有阈值在评测前固定，分割概率不等于校准后的正确率。', '',
 '模型侧采用 320×180 全网格深度测试，选可见人体外轮廓并按最大蒙皮权重对应的三角面分为躯干/腿；不使用内部语义边界。外轮廓的三角形重心坐标与最近点对应每 20 步更新，随后对表面投影坐标求梯度。这是轮廓对应更新，**每个优化步仍同时更新全部位姿参数，不是交替冻结优化**。两方向距离均使用鲁棒损失，按组归一化。低分辨率轮廓、部分语义对应及手臂排除区仍是近似，轮廓改善不保证三维真值误差改善。', '',
 '## Head 6D 与联合优化设置', '',
 'Head 使用 native SMPL-X 关节 15 的位置和局部坐标框架的相机系朝向。朝向由 global→Spine1→Spine2→Spine3→Neck→Head 的旋转链，再乘对象到世界、世界到相机的旋转得到；没有用全局躯干朝向冒充头部朝向。prepare.py 只将这个位置和旋转矩阵提供给拟合器，其余真值仅在评测中使用。Head 固定不代表躯干不能在颈椎处转动。', '',
 '先把网络预测 Head 框架作一次解析刚体对齐，然后用同一初始化进行两个对照：完整位姿锚点，以及额外加入可用轮廓。每个对照 3400 个联合 Adam 步；整体旋转/平移和原来的 19 个身体关节增量同时优化。保留原身体增量上限，整体旋转增量范围为 60°；因此相对上轮除加入朝向、轮廓以外，初始化、优化调度和全局姿态范围也改变，不能将全部差值归因于单一因素。两种本轮对照之间仅轮廓项是否参与优化不同。', '',
 '头部位置误差尺度 1.5 cm、朝向尺度 3°，两项权重均 0.01。朝向用旋转矩阵弦距离，小角度时等价角度平方标准化，避免 acos 的数值梯度问题。保留鼻停用、肩 30 px 容差及 0.25 组权重；无画幅 40 px 排除规则。无全网格 2 cm 软/硬限制，无设备碰撞模型；仅实际重投影点有正深度数值保护。沿用 0.85 倍腕距下限，以本轮共同初始化为参考。', '',
 '## 误差', '',
 '骨盆 XYZ 为预测减真值；身体/腕/肘为三维绝对误差。投影评价统一使用生成真值的 Brown–Conrady 相机，拟合与轮廓使用原多项式相机。入画统计未检查遮挡。', '',
 '| 视角 / 阶段 | Head mm | Head角度 ° | 骨盆 X/Y/Z mm | 骨盆三维 mm | 身体25点 mm | 双腕 mm | 双肘 mm | 身体二维 px |',
 '|---|---:|---:|---|---:|---:|---:|---:|---:|']
assessment=[]
for v in metrics['views']:
 for label in labels:
    q=v['stages'][label];xyz=q['per_joint'][0]['error_xyz_mm']
    lines.append('| %s / %s | %.2f | %.2f | %+.1f / %+.1f / %+.1f | %.2f | %.2f | %.2f | %.2f | %.2f |'%(v['side'],display[label],q['head_anchor_error_mm'],q['head_orientation_error_deg'],*xyz,q['pelvis_absolute_error_mm'],q['body25_absolute_mean_mm'],q['wrists_absolute_mean_mm'],q['elbows_absolute_mean_mm'],q['body25_inframe_projection_px']['mean']))
 a=v['stages']['se3_joint'];b=v['stages']['se3_contour_joint'];prior=v['stages']['previous_position_alternating']
 assessment.append(dict(side=v['side'],contour_vs_no_contour={k:b[k]-a[k] for k in ['body25_absolute_mean_mm','pelvis_absolute_error_mm','wrists_absolute_mean_mm','elbows_absolute_mean_mm']},requested_vs_previous={k:b[k]-prior[k] for k in ['body25_absolute_mean_mm','pelvis_absolute_error_mm','wrists_absolute_mean_mm','elbows_absolute_mean_mm']}))
lines += ['', '## 判断', '',
 '四组拟合均正常退出，保存参数和独立 Head 旋转链复核通过。相对上轮，骨盆及身体平均误差改善，但手臂没有全面改善；因此本轮不能判定为整体成功。', '']
for v in metrics['views']:
 a=v['stages']['previous_position_alternating'];b=v['stages']['se3_contour_joint'];c=v['stages']['se3_joint']
 lines.append('- %s：身体25点 %.2f → %.2f mm，骨盆 %.2f → %.2f mm，骨盆深度偏差 %+.2f → %+.2f mm；双腕 %.2f → %.2f mm，双肘 %.2f → %.2f mm。相对本轮无轮廓对照，加入轮廓使身体平均误差增加 %.2f mm。'%(v['side'],a['body25_absolute_mean_mm'],b['body25_absolute_mean_mm'],a['pelvis_absolute_error_mm'],b['pelvis_absolute_error_mm'],a['per_joint'][0]['error_xyz_mm'][2],b['per_joint'][0]['error_xyz_mm'][2],a['wrists_absolute_mean_mm'],b['wrists_absolute_mean_mm'],a['elbows_absolute_mean_mm'],b['elbows_absolute_mean_mm'],b['body25_absolute_mean_mm']-c['body25_absolute_mean_mm']))
v=next(v for v in metrics['views'] if v['side']=='right');a=v['stages']['previous_position_alternating']['per_joint'][12];b=v['stages']['se3_contour_joint']['per_joint'][12]
lines += ['',
 '最明显的退步是 right/cam4 左腕：%.2f → %.2f mm，其中本轮 Z 方向偏差 %+.2f mm；但真值二维投影误差却从 %.2f 降到 %.2f px。这直接说明投影贴合不能保证腕部深度准确。完整头部位姿只锚定头部，颈椎、躯干和手臂仍能相互补偿。'%(a['absolute_mm'],b['absolute_mm'],b['error_xyz_mm'][2],a['projection_brown_px'],b['projection_brown_px']), '',
 '本轮仅有头部的三维真值约束，手腕仍是单目二维观测；体型固定，肩与锁骨等姿态增量存在上限，right 轮廓组还保留较大的分割→模型残差。这些是后续需要区分的限制，不能仅凭本轮断定某一个是唯一原因。两个本轮对照支持的结论是：当前轮廓项改善了对 Sapiens 的轮廓匹配，没有改善身体三维平均误差。没有可靠腿部观测，也不能据此判断腿轮廓约束有效或无效。', '',
 '本轮所有点仍使用原观测和原多项式标定；没有新增双目手部深度锚点。当前结果不足以替换原流程，尤其 cam4 左腕退步需要保留为失败证据。', '',
 '## 腕肘逐侧与轮廓', '', '| 视角 / 关节 | 原最终 mm | 上轮位置＋交替 mm | 完整位姿 mm | 完整位姿＋轮廓 mm |', '|---|---:|---:|---:|---:|']
for v in metrics['views']:
 for i in [10,11,12,13]:
    vals=[v['stages'][k]['per_joint'][i]['absolute_mm'] for k in ['original_final','previous_position_alternating','se3_joint','se3_contour_joint']]
    lines.append('| %s / %s | %.2f | %.2f | %.2f | %.2f |'%(v['side'],v['stages']['se3_joint']['per_joint'][i]['name'],*vals))
lines += ['', '| 视角 / 阶段 | 躯干模型→分割 px | 分割→模型 px |', '|---|---:|---:|']
for v in metrics['views']:
 for label in ['previous_position_alternating','se3_joint','se3_contour_joint']:
    q=v['stages'][label]['contour_diagnostics']['torso']
    if q.get('valid'):lines.append('| %s / %s | %.2f | %.2f |'%(v['side'],display[label],q['model_to_target_mean_px'],q['target_to_model_mean_px']))
lines += ['',
 '以上轮廓指标相对于 Sapiens 观测，不是真值轮廓误差。完整每关节误差、头部入画、正深度状态、头到骨盆向量及参数摘要见 `evidence/metrics.json`；优化轨迹见两目各对照的 `evidence/*optimization.json`。最后结果按预设目标在每次刷新轮廓对应的检查点选择，不按真值评价挑选。', '',
 '## 验证、复现与公开检查', '',
 '数学测试覆盖独立旋转链、零旋转梯度、肩容差、近表面遮挡和缺失腿部门控。保存参数经 SMPL-X 独立重建；检查体型/手部固定、两对照初始化完全一致、Head 朝向由独立 NumPy 旋转链复核。代码与输入摘要、退出状态及测试见 evidence。最终报告角度使用保存参数的独立双精度旋转链；优化轨迹的单精度 acos 诊断在极小角度时可能舍入为零，不改变优化所用弦距离。', '',
 '使用原 body Python 环境，设置 `PYTHONPATH=<REPO>`；依次运行 `test_math.py`、`prepare.py`、`run.py`、`evaluate.py`、`verify.py`、`build_report.py`。CPU 四线程，未安装依赖、未改动外部后端、未重新推理网络。准备阶段读取真值以生成 Head 6D 目标；拟合器只读取该目标和自动观测。运行目录保存完整本地输出，公开仅派生图、源码、配置和精简数值记录。', '',
 '推送前检查工作区、暂存、待推送提交和远端；不含权重、数据集原始图片、NPZ/NPY、完整原始输出、完整 provenance/patch 或凭据。路径用 `<WORKSPACE>`、`<SYSTEM>` 等占位符替换。见 `evidence/prepush.json`，不把增量检查当作整个历史审计。']
(RUN/'report.md').write_text('\n'.join(lines)+'\n');(RUN/'evidence/assessment.json').write_text(json.dumps(assessment,indent=2)+'\n')
page=['# 完整头部位姿＋躯干轮廓：投影比较','','[数值报告](report.md)','','绿色点是真值，红点是预测；青色是身体，紫色是头部。显示统一使用真值 Brown–Conrady 相机，近面交叉或过长的三角形省略。新结果未做 MANO 手替换。','']
for v in metrics['views']:
 page+=['## '+v['side'],'','实际 Sapiens 轮廓候选（腿部未启用）','','![](assets/'+v['side']+'_contour_targets.jpg)','','| 上轮位置＋交替 | 本轮完整位姿＋轮廓联合优化 |','|---|---|','| ![]('+v['stages']['previous_position_alternating']['image']+') | ![]('+v['stages']['se3_contour_joint']['image']+') |','','完整位姿但无轮廓对照','','![]('+v['stages']['se3_joint']['image']+')','','原最终（含 MANO）','','![]('+v['stages']['original_final']['image']+')','','真值','','![]('+v['truth_image']+')','']
(RUN/'projection.md').write_text('\n'.join(page).rstrip()+'\n')
for side in ['left','right']:
 for case in ['se3_joint','se3_contour_joint']:
    payload=read(OUT/side/case/'optimization.json');(RUN/'evidence'/(side+'_'+case+'_optimization.json')).write_text(json.dumps(payload,indent=2)+'\n')
print('report and projection page written')
