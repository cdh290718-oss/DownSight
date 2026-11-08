"""Publish measured shape-prior results without choosing weights by held-out GT."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];RUN=Path(__file__).resolve().parent;OUT=ROOT/'outputs/synthetic_shape_prior_20261108'
read=lambda p:json.loads(p.read_text());m=read(RUN/'evidence/metrics.json');shapes=read(RUN/'evidence/shape_metrics.json');cfg=read(RUN/'settings.json');execution=read(RUN/'evidence/execution.json')
labels={'B':'B 固定预测体型','E_weak':'弱先验 0.0001','E':'E 主配置 0.001','E_strong':'强先验 0.01','D':'D 真值体型参考'}
stages=m['views'][0]['stages'];b,e,d=[stages[k] for k in ['B','E','D']]
summary='主配置 E：身体25点 %.2f → %.2f mm，骨盆 %.2f → %.2f mm，双腕 %.2f → %.2f mm，双肘 %.2f → %.2f mm。D 的身体平均误差 %.2f mm 仅作为真值体型参考。'%(b['body25_absolute_mean_mm'],e['body25_absolute_mean_mm'],b['pelvis_absolute_error_mm'],e['pelvis_absolute_error_mm'],b['wrists_absolute_mean_mm'],e['wrists_absolute_mean_mm'],b['elbows_absolute_mean_mm'],e['elbows_absolute_mean_mm'],d['body25_absolute_mean_mm'])
lines=['# 双目共享人体：带先验的体型优化', '',
 '- 基线及实际执行 HEAD：`c8e19ad57cf5b5fdabc56bd684e819cd9ec92d7f`，开始时与 GitHub main 一致。上一轮 review 待 Claude。执行时使用本轮未提交脚本；代码哈希见 `evidence/execution.json`。工作树包含本轮新增文件及原有无关未跟踪文件，后者未改动或提交。',
 '- 数据仍为同一合成姿态的 cam3/cam4，两目共同拟合一个人体。沿用[上轮 B 的相机、Head 完整位姿及自动观测](../2026-11-08_synthetic-stereo-abcd/report.md)，未重新推理或修改外部后端。资源与权重获取要求沿用原实验；本轮输入哈希与配置见 evidence。',
 '- 新拟合只读取预先提取的 Head 位姿、生成相机及自动二维观测。进程读取检查禁止访问原始合成 dataset 和 oracle_shape.json。没有读取真值 betas、身体姿态或其他真值关节。D 是上轮真值体型结果，真值体型只在本轮独立评测脚本读取。所有组仍用真值 Head 和生成相机，不代表真实设备可部署精度。', '',
 '[查看投影对比](projection.md)', '', summary, '',
 '## 先验与控制条件', '',
 '两目共享 10 个 SMPL-X betas，与整体旋转/平移及 19 个身体关节增量在每个 Adam 步同时更新。不交替优化；手指和面部参数固定，无 MANO 替换或接缝。体型会影响手部几何，即使手指旋转固定，手形仍可能随 betas 改变。', '',
 '**体型先验为 `lambda * mean(betas ** 2)`，中心是模型的零系数体型，不是原网络预测。** 这是零中心、各向同性的二次正则假设；没有用本设备样本重新估计人口协方差，也不是关节比例或解剖合理性的保证。没有额外硬截断 betas。', '',
 '预先指定主权重 0.001；弱/强对照为 0.0001 / 0.01，每档各跑 cam3 和 cam4 网络初始化。主结果固定使用 cam3 初始化，完整报告其余结果，不按真值误差或总损失挑权重/起点。不同权重下的总目标不能直接用于比较优劣。体型学习率 0.003、3400 步，其他学习率与迭代预算保持上轮 B。', '',
 '其余设置保持 B：原姿态硬范围及先验、Head 位姿锚点、肩 30 px 容差和 0.25 权重、鼻停用、关闭轮廓、无全网格 2 cm 要求；实际投影点保留正深度数值保护，沿用 0.85 腕距下限。Sapiens 身体/腕/肩、WiLoR 每手 21 个二维点及分组权重不变；两目的观测目标取平均，头部及体型先验各只计一次。不另加三角化损失。', '',
 '额外运行冻结体型 B_check。其 betas、body_pose、global_orient、transl 和网格与上轮 B 完全一致，说明改造未改变冻结体型基线。所有新组的初始参数、观测、腕距参考及姿态范围均与对应初始化的 B 相同。', '',
 '## 主结果与先验强度', '',
 '下面为 cam3 系的三维绝对误差，单位 mm，不作骨盆、旋转或尺度对齐。共享同一人体时，cam4 的欧氏距离误差应相同，XYZ 分量与二维误差不同；不是两个独立样本。', '',
 '| 方案 | 身体25点 | 骨盆三维 | 骨盆 X/Y/Z | 双腕 | 双肘 | 手指40点 |', '|---|---:|---:|---|---:|---:|---:|']
for k in ['B','E_weak','E','E_strong','D']:
 q=stages[k];lines.append('| %s | %.2f | %.2f | %+.2f / %+.2f / %+.2f | %.2f | %.2f | %.2f |'%(labels[k],q['body25_absolute_mean_mm'],q['pelvis_absolute_error_mm'],*q['pelvis_error_xyz_mm'],q['wrists_absolute_mean_mm'],q['elbows_absolute_mean_mm'],q['hands40_absolute_mean_mm']))
assessment=[]
for k in ['E_weak','E','E_strong']:
 for suffix in ['', '_right_seed']:
    q=stages[k+suffix];base=stages['B'+suffix];assessment.append(dict(case=k+suffix,delta_vs_corresponding_B={n:q[n]-base[n] for n in ['body25_absolute_mean_mm','pelvis_absolute_error_mm','wrists_absolute_mean_mm','elbows_absolute_mean_mm','hands40_absolute_mean_mm']},per_arm_joint_delta_mm={q['per_joint'][i]['name']:q['per_joint'][i]['absolute_mm']-base['per_joint'][i]['absolute_mm'] for i in [10,11,12,13]}))
lines+=['', '## 腕肘逐侧', '', '| cam3 / 关节 | B 三维 / Z mm | E 三维 / Z mm | D 三维 / Z mm |', '|---|---|---|---|']
for i in [10,11,12,13]:
 cells=[]
 for k in ['B','E','D']:
    q=stages[k]['per_joint'][i];cells.append('%.2f / %+.2f'%(q['absolute_mm'],q['error_xyz_mm'][2]))
 lines.append('| %s | %s |'%(stages['B']['per_joint'][i]['name'],' | '.join(cells)))
lines+=['', '## 两目投影、Head 与肩宽', '', '| 相机 / 组 | 身体二维 px | Head位置 mm | Head角度 ° | 肩宽 px | 骨盆 X/Y/Z mm |', '|---|---:|---:|---:|---:|---|']
for v in m['views']:
 for k in ['B','E_weak','E','E_strong','D']:
    q=v['stages'][k];lines.append('| %s / %s | %.2f | %.3f | %.3f | %.2f | %+.2f / %+.2f / %+.2f |'%(v['camera'],k,q['body25_inframe_projection_px']['mean'],q['head_position_error_mm'],q['head_orientation_error_deg'],q['shoulder_projected_width_px'],*q['pelvis_error_xyz_mm']))
lines+=['', '## 体型是否更接近真值', '',
 '统一把身体、手部和面部姿态、全局旋转及表情置零，分别减去自身骨盆后比较，仅在评测时使用真值 betas。这用于隔离体型几何差异，不等同当前姿态的关节误差。Y 范围是零姿态网格包围盒范围，不是实测站立身高。', '',
 '真值 betas 的平方均值为 %.8f。先验向模型均值收缩是否恰好有利于此合成体型，需要结合这个数及下表判断；不能仅凭本样本推广到不同体型。'%shapes['truth_betas_squared_mean'], '',
 '| 方案 | betas平方均值 | betas到真值 RMSE | 零姿态网格平均误差 mm | 零姿态22关节平均误差 mm |', '|---|---:|---:|---:|---:|']
for q in shapes['cases']:
 lines.append('| %s | %.4f | %.4f | %.2f | %.2f |'%(q['case'],q['betas_squared_mean'],q['betas_error_rmse'],q['neutral_pelvis_centered_vertex_mean_mm'],q['neutral_native22_joint_mean_mm']))
lines+=['', '| 零姿态量 mm | 真值 | B | E弱 | E主 | E强 |', '|---|---:|---:|---:|---:|---:|']
byshape={q['case']:q for q in shapes['cases']}
for name in ['neutral_vertex_y_extent','shoulder_joint_width','head_to_pelvis','left_upper_arm','right_upper_arm','left_forearm','right_forearm','left_thigh','right_thigh','left_shin','right_shin']:
 vals=[shapes['truth_measures_mm'][name]]+[byshape[k]['measures_mm'][name] for k in ['B','E_weak_left','E_left','E_strong_left']];lines.append('| %s | %s |'%(name,' | '.join('%.2f'%x for x in vals)))
lines+=['', '## 初始化敏感性', '', '| 权重 / 初始化 | 身体25点 mm | 骨盆 mm | 双腕 mm | 双肘 mm | 总目标 | 最优检查点 |', '|---|---:|---:|---:|---:|---:|---:|']
for k in ['E_weak','E','E_strong']:
 for seed,suffix in [('cam3',''),('cam4','_right_seed')]:
    q=stages[k+suffix];lines.append('| %s / %s | %.2f | %.2f | %.2f | %.2f | %.7f | %d |'%(cfg['shape_prior_weights'][k],seed,q['body25_absolute_mean_mm'],q['pelvis_absolute_error_mm'],q['wrists_absolute_mean_mm'],q['elbows_absolute_mean_mm'],q['objective'],q['best_step']))
lines+=['', '## 验证、局限与复现', '',
 '共 7 次拟合：冻结 B 复跑 1 次，3 档先验各 2 个初始化。退出记录见 `evidence/execution.json`；相机数值/梯度、三角化往返、betas 对形状及旋转中心的有限差分、先验解析梯度均经过检查。保存参数独立重建，验证正深度、固定字段、Head 旋转链、共享参数及两目三维误差不变性，见 `evidence/verification.json`。', '',
 '参数按各自固定目标的可行检查点选择，不按真值选择。有限步数不保证收敛或全局最优。二维定义差异、固定手指、错误腿部观测和姿态先验仍可能被体型吸收；较小重投影误差或较小 betas 范数均不能证明形体正确。只有一个姿态，没有检验跨姿态稳定性或不同人的泛化。', '',
 '无继承的旧二维缓存：final.npz 保存权威参数及重新生成的有限三维缓存，投影在使用时按对应相机重新计算。本轮没有改动原流水线、外部后端或旧输出。原环境 CPU，每进程 4 线程、最多 2 个进程；缓存仅写仓库忽略的输出目录。', '',
 '复现顺序：`prepare.py`、`test_math.py`、`test_shape.py`、`run.py`、`evaluate.py`、`evaluate_shape.py`、`verify.py`、`build_report.py`。原始模型、数据和完整本地输出不发布；公开派生图、代码、配置和精简数值证据。', '',
 '推送前检查工作区、暂存、本次提交和远端状态。不包含权重、数据集原图、NPZ/NPY、视频、完整原始输出、凭据、完整 provenance/patch；单文件小于 50 MB。机器路径用 `<WORKSPACE>` / `<SYSTEM>` 替换。结果见 `evidence/prepush.json`，不把增量检查当作历史审计。']
if shapes['truth_betas_squared_mean']<1e-12:
 lines.insert(lines.index('## 先验与控制条件'), '**关键限制：这个合成样本的真值 betas 恰好全零，即模型平均体型；D 因而也是固定平均体型参考。强先验把结果拉近真值，不能独立证明模型从图像恢复了个体体型。需要非平均体型和更多姿态再验证权重，不能把本样本最好的权重直接定为通用配置。**\n')
(RUN/'report.md').write_text('\n'.join(lines)+'\n');(RUN/'evidence/assessment.json').write_text(json.dumps(assessment,indent=2)+'\n')
page=['# 双目体型优化投影比较','','[数值报告](report.md)','','绿色点为真值，红色点为预测；青色为网格。E 是预先固定的主权重，不按真值挑选。两目投影来自同一人体；手部未做 MANO 替换。','']
for v in m['views']:
 page+=['## '+v['camera'],'','| B 固定预测体型 | E 带先验优化体型 |','|---|---|','| ![]('+v['stages']['B']['image']+') | ![]('+v['stages']['E']['image']+') |','','| 弱先验 | 强先验 |','|---|---|','| ![]('+v['stages']['E_weak']['image']+') | ![]('+v['stages']['E_strong']['image']+') |','','D 真值体型参考','','![]('+v['stages']['D']['image']+')','','[真值投影](../2026-11-08_synthetic-head-se3-contour/assets/'+v['side']+'_truth.jpg)','']
(RUN/'projection.md').write_text('\n'.join(page).rstrip()+'\n')
for q in execution:
 name=q['case']+'_'+q['seed'];data=read(OUT/name/'optimization.json');(RUN/'evidence'/(name+'_optimization.json')).write_text(json.dumps(data,indent=2)+'\n')
print('report and projection page generated')
