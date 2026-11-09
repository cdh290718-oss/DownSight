"""Build numeric comparison before interpretation; all claims link to recorded evidence."""
import hashlib,json
import numpy as np
from common import ROOT,RUN,OUT,read
m=read(RUN/'evidence/metrics.json');shape=read(RUN/'evidence/shape_metrics.json');metrics=m['views'][0]['stages'];sm={x['case']:x for x in shape['cases']};execution=read(RUN/'evidence/execution.json');verify=read(RUN/'evidence/verification.json');assert len(execution)==4 and all(x['exit_status']==0 for x in execution) and verify['exit_status']==0
mesh={x['stage']:x for x in read(RUN/'evidence/mesh_metrics.json')['stages']}
opts={c:read(OUT/(c+'_left')/'optimization.json') for c in ['A','B','C','D']};fields=['body25_absolute_mean_mm','body_targets12_mean_mm','pelvis_absolute_error_mm','knees_mean_mm','ankles_mean_mm','wrists_absolute_mean_mm','elbows_absolute_mean_mm','hands40_absolute_mean_mm','feet6_mean_mm','feet6_ankle_aligned_mean_mm','head_position_error_mm','head_orientation_error_deg']
summary={label:{k:x[k] for k in fields} for label,x in metrics.items()}
for label in summary:
 summary[label]['neutral_shape_mesh_mm']=sm[label]['neutral_pelvis_centered_vertex_mean_mm']
 for k in ['posed_mesh_absolute_mean_mm','body_surface_without_head_neck_hands_mean_mm','native22_camera_rotation_mean_deg']:summary[label][k]=mesh[label][k]
contrasts={};pairwise={}
for step in read(RUN/'settings.json')['budget_steps']:
 rows={c:summary[c+'_'+str(step)] for c in opts};contrasts[str(step)]={name:{k:rows[u][k]-rows[v][k] for k in rows[u]} for name,u,v in [('B_minus_A_shape_with_network_pose','B','A'),('D_minus_C_shape_with_template_pose','D','C'),('C_minus_A_pose_with_network_shape','C','A'),('D_minus_B_pose_with_mean_shape','D','B'),('D_minus_A_joint_change','D','A')]}
 with np.load(OUT/'A_left'/('budget_'+str(step)+'.npz')) as a,np.load(OUT/'D_left'/('budget_'+str(step)+'.npz')) as d:
  difference=np.linalg.norm(a['smplx_joint_cam'][:65]-d['smplx_joint_cam'][:65],axis=1)*1000;pairwise[str(step)]=dict(body25_between_predictions_mm=float(difference[:25].mean()),hands40_between_predictions_mm=float(difference[25:].mean()),beta_coefficient_difference_l2=float(np.linalg.norm(a['betas']-d['betas'])),posed_mesh_between_predictions_mm=float(np.linalg.norm(a['smplx_mesh_cam']-d['smplx_mesh_cam'],axis=1).mean()*1000))
assessment=dict(exit_status=0,scope='Single posture; head/cameras oracle. All fitting body/hand points automatic. Network refers to EgoSmplx parameter inference, not Sapiens/WiLoR detectors or the SMPL-X body model.',summary=summary,contrasts=contrasts,pairwise_A_D=pairwise,convergence={c:q['convergence'] for c,q in opts.items()},elapsed_seconds={c:q['elapsed_seconds'] for c,q in opts.items()},validation_passed=True)
(RUN/'evidence/assessment.json').write_text(json.dumps(assessment,indent=2)+'\n')
text='''# 网络初值与标准体型/模板姿态：四组对照

[投影与骨架](projection.md) · [数值对照](evidence/assessment.json) · [逐关节与投影误差](evidence/metrics.json) · [验证](evidence/verification.json)

## 设计与结论口径

| 组 | 初始体型 | 初始身体姿态 |
|---|---|---|
| A | 网络体型投影到共同可观测子空间 | 网络姿态 |
| B | 标准体型 beta=0 | 网络姿态 |
| C | 网络体型投影到共同可观测子空间 | 固定零姿态模板 |
| D | 标准体型 beta=0 | 固定零姿态模板 |

A/B/C/D使用完全相同的损失、绝对旋转范围、自动三维目标、完整真值头部锚点和固定手/足/面部设置。D拟合不读取EgoSmplx网络参数；B只读取姿态，C只读取体型。Sapiens和WiLoR检测器及SMPL-X模型仍使用，并不意味着不需要任何神经网络。原流程作为单独参考，不属于仅初值变化的四组因子对照。这里的原流程/旧基线特指上一轮自动双目三维目标＋真值Head＋可观测体型的网络初值版本，不是最早的黄金样本流程。

先比较3400步等预算结果，再在10000、16000、20000步评测。各预算保存截至该步训练目标最小的可行检查点，未按真值误差挑选。后续学习率按预先固定的10000步×0.25、16000步×0.1调整，四组一致。没有对某一组单独调参、挑模板或增加重启。

## 等预算与延长预算

以下为cam3绝对误差，单位mm。两目是同一个三维人体的坐标变换，不是两个独立样本。原流程来自上一轮已验证的自动目标回放，预算3400步；使用原网络中心先验、较紧的增量上限及固定网络手指参数，不能把它与新四组的变化全部归因于初始化。

| 条件 | 身体25点 | 身体目标12点 | 骨盆 | 双膝 | 双踝 | 双腕 | 双肘 | 手指40点 | 脚部6点 | 体型网格 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
'''
columns=['body25_absolute_mean_mm','body_targets12_mean_mm','pelvis_absolute_error_mm','knees_mean_mm','ankles_mean_mm','wrists_absolute_mean_mm','elbows_absolute_mean_mm','hands40_absolute_mean_mm','feet6_mean_mm','neutral_shape_mesh_mm']
for label in ['original']+[c+'_'+str(n) for n in [3400,20000] for c in opts]:text+='| '+label+' | '+' | '.join('%.2f'%summary[label][k] for k in columns)+' |\n'
text+='\n体型网格误差：同一零姿态、各自骨盆居中后的顶点距离均值；没有额外尺度对齐。样本真值beta恰好为0，标准体型组从真实体型出发，这是外推限制。脚部6点为大脚趾、小脚趾与脚跟表面标志，不能叫足趾内部关节。\n'
text+='\n### 分离初值因素（负数表示减号前的组误差更低）\n\n| 预算 | 对比 | 身体25点差 mm | 体型网格差 mm | 手指40点差 mm |\n|---|---|---:|---:|---:|\n'
for n in [3400,20000]:
 for name,row in contrasts[str(n)].items():text+='| %d | %s | %+.3f | %+.3f | %+.3f |\n'%(n,name,row['body25_absolute_mean_mm'],row['neutral_shape_mesh_mm'],row['hands40_absolute_mean_mm'])
text+='\n## 网格与骨骼朝向\n\n| 条件 | 全网格绝对误差 mm | 身体表面（排除头颈手）mm | 22骨骼相机朝向平均误差 ° |\n|---|---:|---:|---:|\n'
for label in ['original']+[c+'_'+str(n) for n in [3400,20000] for c in opts]:text+='| '+label+' | '+' | '.join('%.3f'%summary[label][k] for k in ['posed_mesh_absolute_mean_mm','body_surface_without_head_neck_hands_mean_mm','native22_camera_rotation_mean_deg'])+' |\n'
text+='\n网格按对应顶点直接计算，不作刚体或尺度对齐。表面分组依据最大蒙皮权重所属关节；具体定义及逐关节朝向见mesh_metrics.json。朝向包含运动链前向旋转，不能仅靠关节位置误差判断扭转是否准确。这些评测只在拟合结束后读取真值。\n'
text+='\n## 收敛、头部和约束\n\n| 组 | 训练选中步 | 总loss | 最后1000步loss相对变化 | 目标关节最大位移 mm | 稳定性标志 | CPU拟合秒 | Head位置 mm | Head朝向 ° |\n|---|---:|---:|---:|---:|---|---:|---:|---:|\n'
for c,q in opts.items():
 z=q['convergence'];v=summary[c+'_20000'];text+='| %s | %d | %.9f | %.6f | %.4f | %s | %.2f | %.5f | %.5f |\n'%(c,q['best_step'],q['selected_loss'],z['relative_loss_change'],z['max_target_joint_displacement_mm'],z['diagnostic_pass'],q['elapsed_seconds'],v['head_position_error_mm'],v['head_orientation_error_deg'])
text+='\n稳定性标志预定义为最后1000步loss相对变化小于0.001、目标关节最大位移小于0.1 mm；只是数值轨迹诊断，不证明全局最优、所有未观测扭转稳定或跨样本成功率。CPU至多两个拟合子进程，各四线程；墙钟时间受并发和机器负载影响，不能据此直接宣称线上加速比。完整轨迹见optimization_summary.json。\n'
text+='\n| 预算 | A与D预测身体25点之间距离 mm | 手指40点之间距离 mm | beta差L2 | 两预测网格之间距离 mm |\n|---|---:|---:|---:|---:|\n'
for n,row in pairwise.items():text+='| '+n+' | '+' | '.join('%.5f'%v for v in row.values())+' |\n'
text+='\n新四组最终触及绝对旋转上限：\n\n'
for c in opts:text+='- '+c+': '+(', '.join(metrics[c+'_20000']['saturated_pose_increments']) or '无')+'。\n'
text+='''
## 实现依据与控制

- 双目观测、12个身体目标及10个WiLoR腕/MCP目标、组权重、完整三维协方差、鲁棒损失和Head锚点沿用自动目标基线。没有替换真值身体或手部目标，没有新增轮廓或足部观测。
- 相同信息矩阵产生相同体型子空间；网络体型只用于其中的初始系数。标准体型为0，后续仍在同一子空间内优化，四组体型先验均以0为中心。
- 模板为21个局部身体轴角全部为0的SMPL-X默认模板；网络姿态组仅保留其中活跃的19个关节初值。L_Foot/R_Foot固定为0，四组手、眼、下颌与表情输入均为0，外部模型层自己的手均值约定保留并记录。没有借用网络手/足/面部参数。
- 各组根据同一个完整Head锚点及自己的初始局部姿态/体型求初始根旋转与平移，使起始Head位置和朝向对齐。不读取网络global_orient/transl；这使用的是本来就允许的头部锚点，没有新增身体真值。
- 身体直接优化绝对局部轴角，使用共同弱零姿态先验；取消围绕网络初值的pose delta先验及根旋转/平移先验。绝对旋转范数上限见settings.json，最大肩175°、肘/膝165°。这些是保守工程范围，**不等于完整的解剖关节限制**，没有学习式姿态密度、自碰撞或方向性屈曲约束。轴角的分量、扭转与关节方向仍需独立检查。
- 保留已观测关节在两目正深度的约束；不恢复所有网格顶点必须在前方2cm的旧规则。没有腕距下限、鼻点、40px边缘删除或MANO替换。
- 优化不读取GT身体姿态、GT体型、GT手部、GT轮廓。输入读取清单与隔离审计见verification.json。使用准确生成相机和GT Head，所以不是完整可部署配置。

## 解释与下一步

ANALYSIS_PLACEHOLDER

## 执行、复现与发布

实际执行基线为 `6d5994401b33450ea3343d20f44f2cc54e9ac5d6`，origin/main开始时一致。执行时工作树包含本轮新增代码，以及既有无关未跟踪point_comparison_followup.json，后者未修改或提交。上一轮review待Claude，本轮review同样待审，不冒充独立审核。

输入沿用 `DownSightSmplxdata@192671271a74b99f41b9eb033d8f88b1dac77ada` 的000001样本cam3/cam4。来源哈希见preparation.json，执行源码/设置哈希、命令、退出状态见execution.json；评测与绘图见evaluation_execution.json。外部后端不是Git工作树，读取Git版本返回128，不能声称本轮确认了后端提交；执行后源码快照摘要见resources.json，完整清单留在本地。权重资产沿用基线资源标识，本轮未重新逐一计算其哈希。prepare.py从已有自动观测和网络raw参数中仅提取允许的初始化因素，run.py执行四组，evaluate_all.py独立评测和验证，最后build_report.py生成表格。需要先具备[基线外部资源](../../experiments/synthetic_oracle_head_shape_20261108/REPRODUCE.md)，本轮脚本不下载资产、不安装环境；无法访问外部资产不能声称独立复现。

中间NPZ、完整日志、模型路径和运行时目录保存在 `<WORKSPACE>/DownSight/outputs/initialization_ablation_20261109`，不提交。公开记录只含代码、配置、精简数值轨迹和派生图。机器路径替换为 `<WORKSPACE>`、`<BODY_PYTHON>` 等占位符；非致命运行警告和失败若出现均保留证据。

发布前分别检查工作区、暂存、本次提交和远端：不含权重、数据集、视频、NPZ/NPY、完整原始输出、凭据、.env、完整provenance或patch；单文件小于50MB。推送前自检见evidence/prepush.json，完整本地收据留在本轮outputs；提交审计通过后才push。本审计只覆盖本轮增量，不代表旧历史已全部审计。
'''
analysis=RUN/'analysis.txt'
text=text.replace('ANALYSIS_PLACEHOLDER',analysis.read_text() if analysis.exists() else '结果解释尚未完成，不应发布。')
(RUN/'report.md').write_text(text)
lines=['# 网络初值消融：投影与骨架','', '[返回完整报告](report.md)','', 'A=网络体型/网络姿态；B=标准体型/网络姿态；C=网络体型/模板姿态；D=标准体型/模板姿态。新四组目标和约束相同，全部身体/手部观测为自动值，Head为真值。','', '## 收敛','','![convergence](assets/convergence.png)','']
for side in ['left','right']:
 for n in [3400,20000]:lines+=['## '+side+' / A与D / '+str(n)+'步','','绿：真值；蓝：A；橙：D。同一相机两种预算保持相同轴范围；侧视米制等比例。','','![skeleton](assets/skeleton_'+side+'_'+str(n)+'.png)','']
lines+=['## 原图网格投影','', '绿点：真值；红点：预测；网格为预测。部分跨相机平面或过长投影三角形由渲染器跳过；关键点接近不代表轮廓、头部可见性或手指形态准确。','']
for side in ['left','right']:
 lines+=['### '+side+' / 原流程参考','','![original](assets/'+side+'_original.jpg)','']
 for n in [3400,20000]:
  for c in opts:lines+=['### '+side+' / '+c+' / '+str(n)+'步','','![projection](assets/'+side+'_'+c+'_'+str(n)+'.jpg)','']
(RUN/'projection.md').write_text('\n'.join(lines).rstrip()+'\n');print(json.dumps({k:v for k,v in summary.items() if k=='original' or k.endswith('_3400') or k.endswith('_20000')},indent=2))
