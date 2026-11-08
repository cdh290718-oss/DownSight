"""Build report and projection page directly from measured evidence."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];RUN=Path(__file__).resolve().parent;OUT=ROOT/'outputs/synthetic_stereo_abcd_20261108'
read=lambda p:json.loads(p.read_text());m=read(RUN/'evidence/metrics.json');inputs=read(RUN/'evidence/inputs.json');tri=read(RUN/'evidence/triangulation.json');execution=read(RUN/'evidence/execution.json')
prev=read(ROOT/'runs/2026-11-08_synthetic-head-se3-contour/evidence/metrics.json')
labels={'A':'A 单目','B':'B 双目','C':'C 双目＋放宽硬范围','D':'D 双目＋真值体型'}
aL=m['views'][0]['stages']['A'];aR=m['views'][1]['stages']['A'];b,c,d=[m['views'][0]['stages'][k] for k in ['B','C','D']]
conclusions=[
 '主结果：双目共享人体明显改善腕部；单独放宽硬范围在本轮预算内未改善身体平均误差；真值体型对照改善最大。另一种初始化保持同样趋势。',
 'A 两目身体平均误差分别为 %.2f / %.2f mm，双腕 %.2f / %.2f mm；B 共享人体为 %.2f mm、双腕 %.2f mm。B 的 cam3 骨盆误差从 A 的 %.2f 变为 %.2f mm，并非所有部位同时改善。'%(aL['body25_absolute_mean_mm'],aR['body25_absolute_mean_mm'],aL['wrists_absolute_mean_mm'],aR['wrists_absolute_mean_mm'],b['body25_absolute_mean_mm'],b['wrists_absolute_mean_mm'],aL['pelvis_absolute_error_mm'],b['pelvis_absolute_error_mm']),
 'B→C：身体 %.2f → %.2f mm，双腕 %.2f → %.2f mm，双肘 %.2f → %.2f mm。放宽后本轮主结果没有触及新的硬上限，但不能因此证明已全局收敛。'%(b['body25_absolute_mean_mm'],c['body25_absolute_mean_mm'],b['wrists_absolute_mean_mm'],c['wrists_absolute_mean_mm'],b['elbows_absolute_mean_mm'],c['elbows_absolute_mean_mm']),
 'B→D：身体 %.2f → %.2f mm，双腕 %.2f → %.2f mm，双肘 %.2f → %.2f mm，骨盆 %.2f → %.2f mm。D 只换真值 betas，支持固定体型是本样本的重要限制；这些差值包含优化交互，不能分解成可加的误差份额，也不保证自由优化体型就能达到 D。'%(b['body25_absolute_mean_mm'],d['body25_absolute_mean_mm'],b['wrists_absolute_mean_mm'],d['wrists_absolute_mean_mm'],b['elbows_absolute_mean_mm'],d['elbows_absolute_mean_mm'],b['pelvis_absolute_error_mm'],d['pelvis_absolute_error_mm']),
 '建议下一轮保留双目共享与原姿态范围，单独检验带先验的体型优化；轮廓继续关闭。D 剩余骨盆偏差主要在相机 Y 轴，仍需检查髋/下肢观测及其内部关节定义，不能宣称全部问题已经解决。'
]
lines=['# 合成数据 A–D：共享双目人体、姿态范围和体型', '',
 '- 基线及实际执行 HEAD：`118c485f5b73c5767d050983d371a0315a9e7dba`，开始时与 GitHub main 一致；上一轮 review 待 Claude。执行时本轮脚本尚未提交，工作树包含本轮新增内容及原有无关未跟踪文件；后者未改动或提交。运行代码哈希见 `evidence/execution.json`。',
 '- 数据：同一合成姿态的 cam3/left 与 cam4/right，不能当作两个独立姿态。复用[原实验](../2026-11-08_synthetic-original/report.md)的网络、Sapiens、WiLoR 观测和资源；没有重新推理、安装依赖或修改外部后端。输入文件及相机摘要见 `evidence/inputs.json`。',
 '- 所有组均输入真值 Head 完整位姿与生成相机；D 额外输入真值体型。其他三维关节、身体姿态和网格真值仅用于评测。这是 Oracle 诊断，不代表单目或真实设备可部署精度。',
 '- 全部为身体拟合，手指/面部参数固定；没有 MANO 手替换、接缝或体型自由优化。D 是固定真值体型诊断，不能表述为从图像估计出了真值体型。', '',
 '[查看 A–D 投影对比](projection.md)', '',
 *[item for paragraph in conclusions for item in [paragraph,'']],
 '## 控制条件', '',
 '| 组 | 参与拟合的相机 | 姿态硬范围 | 体型 |', '|---|---|---|---|',
 '| A | 两目分别拟合，各只有本目二维目标 | 原范围 | 固定网络预测 |',
 '| B | 同时拟合两目，共享同一个人体 | 原范围 | 固定网络预测 |',
 '| C | 同 B | 原来的两倍 | 同 B |',
 '| D | 同 B | 同 B | 固定真值 |', '',
 '主对照全部从 cam3 网络预测初始化，参数统一存于 cam3 CV 坐标系。A 的两个单目拟合及 B/C/D 的初始 global_orient、body_pose、transl、手部参数完全相同，D 只替换 betas；D 没有额外重新对齐位置。整体方向通过预测 Head 的旋转链与真值 Head 框架解析对齐。B–D 再各从 cam4 网络预测独立重跑一次，作为初始化敏感性检查；不根据真值或最终目标值替换主结果。', '',
 '所有组使用生成时的 Brown–Conrady 内参、畸变及相机外参。一个共享 SMPL-X 网格先在 cam3 系生成，然后用外参投影到每个观测相机；两目的姿态和体型不是独立变量。B–D 的三维欧氏误差在两目应一致，仅误差 XYZ 分量和投影不同，数值不一致意味着坐标或评测问题。', '',
 '头部约束、停用鼻、肩 30 px 容差和 0.25 权重保持。取消全网格 2 cm 约束，不加轮廓、不交替冻结参数。每组 3400 步，整体位姿和 19 个身体关节同时更新。C 仅把这些关节的硬增量上限加倍；姿态先验仍使用原上限作为归一化尺度，权重不变。硬范围约束的是相对初始预测的增量，不等同解剖活动范围；没有新增已训练的人体姿态先验。', '',
 'Sapiens 仍保留原身体/腕/肩点；WiLoR 按手侧和相对 Sapiens 腕位置作一对一匹配，只取二维 21 点，每只手独立归一化、权重 0.1。没有使用 WiLoR 的三维手坐标、25000 px 抬升或 MANO 网格。A–D 观测政策完全相同；极线检查和三角化仅用于诊断，不参与 A 的点筛选或任何组的目标。手指固定，因此错误手形仍可能影响臂部，此限制没有在本轮单独消融。', '',
 '各相机内部按 body/wrist/shoulder/左右手分组归一化，双目对相机目标取平均；Head 项只计一次，避免双目自动把观测权重翻倍。沿用 0.85 腕距下限和实际投影点的正深度数值保护，D 与 B 使用相同腕距参考。完整配置见 `settings.json`。', '',
 '**相对上轮，A 除统一相机外，还加入了上述 WiLoR 二维手点，并统一主初始化。** 因此上轮到 A 的差值不能归因于相机单因素；本轮 A→B、B→C、B→D 才是对应控制对照。', '',
 '## 主结果', '',
 '单位 mm，骨盆 XYZ 为预测减真值；身体25点、腕肘均为绝对三维误差，不作骨盆、旋转或尺度对齐。二维评价使用真值入画标记，不额外判定遮挡。', '',
 '| 相机 / 组 | 身体25点 | 骨盆三维 | 骨盆 X/Y/Z | 双腕 | 双肘 | 手指40点 | 身体二维 px |', '|---|---:|---:|---|---:|---:|---:|---:|']
for v in m['views']:
 for k in ['A','B','C','D']:
    q=v['stages'][k];lines.append('| %s / %s | %.2f | %.2f | %+.2f / %+.2f / %+.2f | %.2f | %.2f | %.2f | %.2f |'%(v['camera'],labels[k],q['body25_absolute_mean_mm'],q['pelvis_absolute_error_mm'],*q['pelvis_error_xyz_mm'],q['wrists_absolute_mean_mm'],q['elbows_absolute_mean_mm'],q['hands40_absolute_mean_mm'],q['body25_inframe_projection_px']['mean']))
assessment=[]
for v in m['views']:
 for baseline,case in [('A','B'),('B','C'),('B','D')]:
    a,b=[v['stages'][k] for k in [baseline,case]];assessment.append(dict(view=v['side'],comparison=case+' minus '+baseline,delta={k:b[k]-a[k] for k in ['body25_absolute_mean_mm','pelvis_absolute_error_mm','wrists_absolute_mean_mm','elbows_absolute_mean_mm','hands40_absolute_mean_mm']},per_arm_joint_delta_mm={a['per_joint'][i]['name']:b['per_joint'][i]['absolute_mm']-a['per_joint'][i]['absolute_mm'] for i in [10,11,12,13]}))
lines+=['', '## 腕肘逐侧：三维 / 深度 / 投影', '', '| 相机 / 关节 | A mm / Z mm / px | B mm / Z mm / px | C mm / Z mm / px | D mm / Z mm / px |', '|---|---|---|---|---|']
for v in m['views']:
 for i in [10,11,12,13]:
    cells=[]
    for k in ['A','B','C','D']:
        q=v['stages'][k]['per_joint'][i];cells.append('%.2f / %+.2f / %.2f'%(q['absolute_mm'],q['error_xyz_mm'][2],q['projection_px']))
    lines.append('| %s / %s | %s |'%(v['camera'],v['stages']['A']['per_joint'][i]['name'],' | '.join(cells)))
lines+=['', '## 头部、肩宽和触限', '', '| 相机 / 组 | Head位置 mm | Head角度 ° | 投影肩宽 px | 触及硬上限的关节 |', '|---|---:|---:|---:|---|']
for v in m['views']:
 for k in ['A','B','C','D']:
    q=v['stages'][k];lines.append('| %s / %s | %.3f | %.3f | %.2f | %s |'%(v['camera'],k,q['head_position_error_mm'],q['head_orientation_error_deg'],q['shoulder_projected_width_px'],', '.join(q['saturated_pose_increments']) or '无'))
lines+=['', '## 初始化敏感性', '', '以下只看 cam3 系；B–D 的 cam4 三维欧氏误差等价。两种初始化均完整运行，主结果始终取预先指定的 cam3 初始化。', '', '| 组 / 初始化 | 身体25点 mm | 骨盆 mm | 双腕 mm | 双肘 mm | 目标函数 | 最优检查点 |', '|---|---:|---:|---:|---:|---:|---:|']
v=m['views'][0]
for k in ['B','C','D']:
 for seed,label in [('cam3',k),('cam4',k+'_right_seed')]:
    q=v['stages'][label];lines.append('| %s / %s | %.2f | %.2f | %.2f | %.2f | %.7f | %d |'%(k,seed,q['body25_absolute_mean_mm'],q['pelvis_absolute_error_mm'],q['wrists_absolute_mean_mm'],q['elbows_absolute_mean_mm'],q['objective'],q['best_step']))
lines+=['', '## 自动观测的三角化诊断', '', '使用自动二维点和生成标定，不使用真值点重建。这里的三维残差也包含检测定义与内部关节的差异，不能全部叫检测器误差。协方差只是每个像素坐标独立 2 px 噪声的线性传播，未含语义偏差或标定误差；不作为已校准置信度。全部点、极线残差、双目正深度与协方差见 `evidence/triangulation.json`。', '', '| 来源 / 点 | 三维误差 mm | cam3 Z误差 mm | 像素噪声假设下 Z标准差 mm | 两目重投影平均 px |', '|---|---:|---:|---:|---:|']
for q in tri['points']:
 if q['source']=='sapiens' or q['model_index'] in [12,13]:lines.append('| %s / %s | %.2f | %+.2f | %.2f | %.2f |'%(q['source'],q['name'],q['absolute_mm'],q['error_xyz_mm'][2],q['depth_std_mm'],q['reprojection_mean_px']))
lines+=['', '## 与上轮和原流程的参考比较', '', '原 final 包含 MANO 和后续身体调整；本轮均是身体拟合，不能把阶段差异当成某个单独约束的收益。', '', '| 相机 / 阶段 | 身体25点 mm | 骨盆 mm | 双腕 mm | 双肘 mm |', '|---|---:|---:|---:|---:|']
for v in prev['views']:
 for key,name in [('original_body','原 body'),('original_final','原 final / MANO'),('se3_joint','上轮 Head6D 无轮廓'),('se3_contour_joint','上轮 Head6D 加轮廓')]:
    q=v['stages'][key];lines.append('| %s / %s | %.2f | %.2f | %.2f | %.2f |'%(v['side'],name,q['body25_absolute_mean_mm'],q['pelvis_absolute_error_mm'],q['wrists_absolute_mean_mm'],q['elbows_absolute_mean_mm']))
lines+=['', '## 执行、验证与公开检查', '',
 '共 8 次拟合（A 两目＋B/C/D 各两种初始化），退出状态见 `evidence/execution.json`。最优参数按固定目标的可行检查点选择，不看真值误差；没有凭真值挑选初始化。保存参数独立重建，核验网格和投影、Head 旋转链、固定体型/手部、A–D 对照条件、共享参数以及两目欧氏误差的刚体变换不变性。投影实现对照 OpenCV 并做有限差分梯度检查及三角化往返测试。', '',
 '原始拟合 final.npz 继承了部分网络关节/投影缓存；评测始终从实际 SMPL-X 参数独立重建，没有使用这些缓存。另导出 parameters.npz，只保留权威参数并重建三维缓存；二维投影另存带有效标记的 projections.npz，允许镜头后方点无有效投影，拟合参数逐元素不变；复用结果应使用该导出。导出复核首次将重建网格要求为逐元素一致而失败；实际最大差在浮点重建容差内，现与原重载检查统一使用 1e-5 m 阈值，参数本身仍要求逐元素一致。失败记录保留在 `evidence/checks.json`。缓存差异和导出哈希见 `evidence/exports.json`，原文件保留供审计。', '',
 '固定迭代预算不等于证明优化收敛或找到全局最优。C 的部分轨迹在预算末仍下降，结论只适用于本轮预算和先验，不支持断言放宽范围永远无效。', '',
 '原环境 CPU 执行，每进程 4 线程、最多两个进程；运行时缓存位于仓库忽略输出目录。复现顺序：`prepare.py`、`test_math.py`、`run.py`、`triangulation.py`、`export.py`、`evaluate.py`、`verify.py`、`build_report.py`；body Python 和外部资产获取要求见原实验资源记录。完整本地输出不发布，公开精简数值及派生 JPG。', '',
 '推送前逐项检查工作区、暂存、本次提交与远端状态。禁止权重、原始数据集、NPZ/NPY、视频、完整原始输出、凭据、完整 provenance/patch；单文件小于 50 MB。日志中的机器路径替换为 `<WORKSPACE>` / `<SYSTEM>`，保留判断所需信息。见 `evidence/prepush.json`。这不是整个历史审计。']
(RUN/'report.md').write_text('\n'.join(lines)+'\n');(RUN/'evidence/assessment.json').write_text(json.dumps(assessment,indent=2)+'\n')
page=['# A–D 投影对比','','[数值报告](report.md)','','绿色点为真值，红色点为预测；青色为身体网格。所有图使用生成相机。B/C/D 两目显示同一人体的不同投影；手部未做 MANO 替换。','']
for v in m['views']:
 page+=['## '+v['camera'],'','| A 单目 | B 共享双目 |','|---|---|','| ![]('+v['stages']['A']['image']+') | ![]('+v['stages']['B']['image']+') |','','| C 放宽硬范围 | D 真值体型诊断 |','|---|---|','| ![]('+v['stages']['C']['image']+') | ![]('+v['stages']['D']['image']+') |','','[真值投影](../2026-11-08_synthetic-head-se3-contour/assets/'+v['side']+'_truth.jpg)','']
(RUN/'projection.md').write_text('\n'.join(page).rstrip()+'\n')
for q in execution:
 name=q['case']+'_'+q['seed'];data=read(OUT/name/'optimization.json');(RUN/'evidence'/(name+'_optimization.json')).write_text(json.dumps(data,indent=2)+'\n')
print('report and projection page generated')
