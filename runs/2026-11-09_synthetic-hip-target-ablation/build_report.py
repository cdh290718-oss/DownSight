"""Summarize the predeclared hip-only intervention, including non-target joints."""
import hashlib,json
from pathlib import Path
from common import ROOT,RUN,OUT,read
metrics=read(RUN/'evidence/metrics.json');shape=read(RUN/'evidence/shape_metrics.json');verification=read(RUN/'evidence/verification.json');execution=read(RUN/'evidence/execution.json');preparation=read(RUN/'evidence/preparation.json')
assert verification['exit_status']==0 and len(execution)==2 and all(q['exit_status']==0 for q in execution)
rows=[]
for v in metrics['views']:
 a=v['stages']['control_left'];b=v['stages']['gt_hips_left'];joints=[]
 for old,new in zip(a['per_joint'],b['per_joint']):
  joints.append(dict(name=old['name'],before_absolute_mm=old['absolute_mm'],after_absolute_mm=new['absolute_mm'],absolute_change_mm=new['absolute_mm']-old['absolute_mm'],before_error_xyz_mm=old['error_xyz_mm'],after_error_xyz_mm=new['error_xyz_mm'],prediction_displacement_xyz_mm=[y-x for x,y in zip(old['error_xyz_mm'],new['error_xyz_mm'])],before_pelvis_aligned_mm=old['pelvis_aligned_mm'],after_pelvis_aligned_mm=new['pelvis_aligned_mm']))
 fields=['body25_absolute_mean_mm','body25_excluding_hips_mean_mm','pelvis_knees_ankles_mean_mm','pelvis_absolute_error_mm','knees_mean_mm','ankles_mean_mm','wrists_absolute_mean_mm','elbows_absolute_mean_mm','hands40_absolute_mean_mm','head_position_error_mm','head_orientation_error_deg']
 rows.append(dict(camera=v['camera'],side=v['side'],summary={k:dict(before=a[k],after=b[k],change=b[k]-a[k]) for k in fields},pelvis_abs_Y_error_reduction_percent=100*(abs(a['pelvis_error_xyz_mm'][1])-abs(b['pelvis_error_xyz_mm'][1]))/abs(a['pelvis_error_xyz_mm'][1]),joints=joints))
q=dict(scope='Only two hip loss target coordinates changed. Primary left network seed fixed. Unchanged true Head anchor and known camera calibration. Hip metrics are not held-out, so report other joints separately.',views=rows,shape=shape['cases'],validation_passed=True,causal_scope='Within this fixed synthetic case and optimizer budget; does not identify semantic mismatch versus detector/triangulation error and is not a deployable GT-free correction.')
(RUN/'evidence/assessment.json').write_text(json.dumps(q,indent=2)+'\n')
v=rows[0];a=metrics['views'][0]['stages']['control_left'];b=metrics['views'][0]['stages']['gt_hips_left'];j=v['joints'];s={r['case']:r for r in shape['cases']}
text='''# 只替换左右髋三维目标的对照实验\n\n本轮对同一合成姿态执行两次拟合：复跑原 oracle_left 条件，以及仅把两侧髋的损失目标改为对应真值。两者均从原网络初值开始；没有从上一轮 final 继续优化。\n\n[骨架与投影对比](projection.md) · [数值汇总](evidence/assessment.json) · [逐关节指标](evidence/metrics.json) · [控制条件验证](evidence/verification.json)\n\n## 核心结果\n\n以下为 cam3 坐标，三维欧氏误差不做对齐；两目共享一个人体，cam4 是同一结果的坐标变换，不是另一个独立样本。\n\n| 指标 | 原条件复跑 | 只改真值髋 | 变化（后−前） |\n|---|---:|---:|---:|\n'''
labels={'body25_absolute_mean_mm':'身体25点 mm','body25_excluding_hips_mean_mm':'身体25点排除两髋 mm','pelvis_knees_ankles_mean_mm':'骨盆＋双膝＋双踝均值 mm','pelvis_absolute_error_mm':'骨盆 mm','knees_mean_mm':'双膝 mm','ankles_mean_mm':'双踝 mm','wrists_absolute_mean_mm':'双腕 mm','elbows_absolute_mean_mm':'双肘 mm','hands40_absolute_mean_mm':'手指40点 mm','head_position_error_mm':'Head位置 mm','head_orientation_error_deg':'Head朝向 °'}
for k,label in labels.items():
 r=v['summary'][k];text+='| '+label+' | %.4f | %.4f | %+.4f |\n'%(r['before'],r['after'],r['change'])
text+='\n骨盆 Y 偏差由 **%+.2f mm** 变为 **%+.2f mm**，绝对 Y 偏差减少 **%.2f%%**。注意髋本身已直接使用真值；骨盆、膝、踝才是判断偏差是否沿身体传播的重要额外指标。\n'%(j[0]['before_error_xyz_mm'][1],j[0]['after_error_xyz_mm'][1],v['pelvis_abs_Y_error_reduction_percent'])
text+='\n**结论：本轮支持原髋观测是骨盆偏差的重要驱动因素，但不支持下肢只是继承骨盆平移的解释。** 骨盆显著回正，而双膝、双踝绝对误差基本保留。此前根据侧视平行形态提出的共同位移推测，需要据此修正。左右腕肘的绝对误差变化均很小，具体数值如下；没有用均值掩盖单侧变化。\n'
text+='\n## 骨盆、下肢与手臂逐点\n\n坐标误差均为预测减真值，单位 mm。\n\n| 关节 | 绝对误差 前→后 | Y误差 前→后 | Z误差 前→后 |\n|---|---:|---:|---:|\n'
for i in [0,1,2,3,4,5,6,14,17,8,9,10,11,12,13]:
 r=j[i];text+='| '+r['name']+' | %.2f → %.2f | %+.2f → %+.2f | %+.2f → %+.2f |\n'%(r['before_absolute_mm'],r['after_absolute_mm'],r['before_error_xyz_mm'][1],r['after_error_xyz_mm'][1],r['before_error_xyz_mm'][2],r['after_error_xyz_mm'][2])
text+='\n中性体型平均顶点误差：%.2f → %.2f mm。允许的体型方向完全不变，但优化后的系数允许随损失改变；它属于干预引起的结果变化。\n'%(s['control_left']['neutral_pelvis_centered_vertex_mean_mm'],s['gt_hips_left']['neutral_pelvis_centered_vertex_mean_mm'])
text+='\n### 骨盆平移与下肢相对姿态\n\n下表先各自减去骨盆位置，再算误差，避免把共同位移误判成每段腿都变形。它是补充指标，不能代替绝对误差。\n\n| 关节 | 骨盆对齐后误差 前→后 mm | 本轮预测 Y 位移 mm |\n|---|---:|---:|\n'
for i in [3,4,5,6,14,17]:
 r=j[i];text+='| '+r['name']+' | %.2f → %.2f | %+.2f |\n'%(r['before_pelvis_aligned_mm'],r['after_pelvis_aligned_mm'],r['prediction_displacement_xyz_mm'][1])
text+='\n全部 XYZ 位移和另一个相机坐标下的结果见 assessment.json。身体整体均值不能替代左右腕肘分别验收；上表保留各侧，手指40点另行报告。\n'
ray=read(RUN/'evidence/ray_diagnostic.json')
text+='\n### 为什么膝、踝仍然偏：观测方向诊断\n\n作为拟合结束后的分析，将原自动三角化点沿 cam3 原点射线缩放到真值 Z，即 `p_auto × Z_GT/Z_auto`，不改变其射线方向。这里只使用真值深度分析已有观测，不再次拟合，也没有把这些点输入优化。\n\n| 关节 | 自动点沿射线移到真值深度后的 Y 偏差 mm | 只改髋后拟合 Y 偏差 mm |\n|---|---:|---:|\n'
for row in ray['rows']:text+='| '+row['name']+' | %+.2f | %+.2f |\n'%(row['ray_at_true_Z_error_xyz_mm'][1],row['gt_hips_error_xyz_mm'][1])
text+='\n结果与剩余下肢 Y 偏差相近：即使深度正确，原观测方向仍会给出偏高的膝踝位置。这支持继续检查膝踝观测方向，而不是继续整体平移人体。它仍不能区分关节定义、遮挡/定位错误或其他几何因素；三角化点的射线也不完全等于任一单目的原检测射线。原始数值、公式和来源哈希见 [ray_diagnostic.json](evidence/ray_diagnostic.json)。\n'
text+='''\n## 干预是否真的只改了髋目标\n\n- control：原自动三角化身体/手点、原完整真值 Head 锚点、原生成相机。\n- gt_hips：先运行同一个自动初始化与体型信息计算，再仅覆盖 Sapiens `left_hip` / `right_hip` 的三维 `xyz`。不替换骨盆、膝、踝或任何手点。\n- 两组设置文件与源 run 逐字节一致；协方差、权重、点分组、门控、采样、Huber 损失、均值先验、关节限幅、学习率、3400 步预算和检查点选择均不变。配置中的历史说明仍沿用源 run，实际新增条件以 cases.json 和 hip_targets.py 为准。\n- 初始化仍用原自动观测；没有把真值髋输入 Kabsch、骨长信息矩阵、体型子空间或根位姿先验。两组 initial 参数和初始化信息都与原 oracle_left 完全一致；control final 的全部核心参数及网格/关节缓存与原结果逐元素完全一致。\n- 拟合进程仍禁止读取整个 GT dataset；仅干预组允许读取离线导出的两个髋坐标。已知 Head 锚点仍单独读取，没有真值体型/身体姿态输入。\n\n准备与坐标变化见 [preparation.json](evidence/preparation.json)、[hip_targets.json](evidence/hip_targets.json)。原始三角化文件及相机/头部锚点按字节复制；三角化中的回投、协方差和筛选元数据仍描述原检测点，不能误解为真值替换后的新三角化质量。`hip_targets.py` 验证除了两个 xyz 外各字段不变。\n\n## 如何解释\n\n本轮对模型输入的干预显著修正了骨盆，膝踝却没有随之整体回正。髋/膝的优化增量均未达到其硬上限，不能把它们当前的残差直接说成被这些上限卡住；但预算末尾 loss 仍有下降，其他先验、手部约束与非线性耦合保留，因此也不能宣称优化已全局收敛或完全无影响。姿态和体型都可在既定范围内联合调整，干预响应不能分解成可相加的独立误差份额。

下一步建议分别替换膝、踝观测并保持同样控制条件，区分其各自的驱动作用；本轮尚未执行。应先验证这种观测方向偏差能否跨姿态重复，再决定学习偏移校正或改变观测定义。\n\n这不能单独证明偏差来自“检测标在体表、关节在内部”：检测定位、遮挡、双目对应和定义差异都可能使目标有偏。真值替换也不是一个可部署的偏移校正器，不能以此宣称已解决新人物/新姿态的检测问题。仍保留另一侧身体与手观测偏差、关节限幅、单姿态/均值体型优势、准确标定和真值 Head 条件；脚趾/脚跟目标仍缺失，踝/足方向未增加新观测。\n\n检查点按各自训练目标选取，不按真值误差选；目标坐标改变后不使用两组总 loss 大小作为精度比较。头部近零误差主要来自真值锚点。报告两目仅用于核对坐标和投影，不把它们当独立样本计算泛化成功率。\n\n## 执行与验证\n\n执行基线 HEAD / origin/main：`7a08c08b1afa9635447760cf0d140297137665dd`。运行时工作树包含本轮新脚本和既有无关未跟踪 point_comparison_followup.json，后者未改动或提交。原候选基线见 [experiments](../../experiments/synthetic_oracle_head_shape_20261108/README.md)。\n\n实际执行代码和设置 SHA-256、逐组命令、退出码和耗时见 [execution.json](evidence/execution.json)。两个 CPU 子进程、各四个线程；没有运行 GPU 推理或安装环境。每组参数重建、梯度有限和投影一致性沿用原检查；后验验证增加 control 精确回归、初始化/子空间逐项一致、仅处理组读取髋真值、两组损失独立重算。完整检查见 verification.json；精简优化摘要与日志见 optimization_summary.json、fit_log_summary.json；评测命令与退出码见 evaluation_execution.json。\n\n原始输入沿用 `DownSightSmplxdata@192671271a74b99f41b9eb033d8f88b1dac77ada` 的样本 000001。提取目标的源文件与哈希见 hip_targets.json；其他输入与资源访问要求沿用[归档资源说明](../../experiments/synthetic_oracle_head_shape_20261108/REPRODUCE.md)。模型权重、原始图和 NPZ 输出不上传；输出位于 `<WORKSPACE>/DownSight/outputs/synthetic_hip_target_ablation_20261109`，哈希见优化/评测证据。无法访问外部资产的审核者不能声称独立复现。\n\n## 推送前自检\n\n仅发布代码、配置、报告、精简数值证据与派生图；不含权重、数据集、视频、NPZ/NPY、完整原始输出、凭据或 .env；单文件小于 50 MB。实际开发机路径脱敏为 `<WORKSPACE>`、`<BODY_PYTHON>` 等占位符，不发布完整 provenance/patch。检查工作区、暂存内容、本次提交以及 origin/main；工作区/暂存收据见 evidence/prepush.json，完整本地收据保存在本轮 outputs。首次图片审核因 PIL 对 PNG 的 verify 调用顺序要求报错，已改为重新打开图片验证并复查；失败与修正见 evidence/development_failures.json。该检查器修正不改变实验结果。提交检查通过后才 push，仅覆盖本轮增量，不表示历史全部通过。\n'''
(RUN/'report.md').write_text(text)
lines=['# 只改髋目标：前后对比','', '[返回报告](report.md)','', '绿色是真值，蓝色是原条件复跑，橙色是仅替换两髋目标。骨架图同一相机内使用相同坐标轴，侧视坐标为米且横纵单位比例一致；两目不代表独立样本。','']
for side in ['left','right']:lines+=['## '+('cam3' if side=='left' else 'cam4')+' 骨架','', '!['+side+' skeleton](assets/skeleton_'+side+'.png)','']
lines+=['## 原图投影','', '绿色点为真值，红色点为预测，半透明网格为预测；渲染器可能略去相机平面相交或过长投影三角形。','']
for side in ['left','right']:
 for case,label in [('control_left','原条件复跑'),('gt_hips_left','仅替换两髋目标')]:lines+=['### '+side+' / '+label,'','!['+side+' '+case+'](assets/'+side+'_'+case+'.jpg)','']
(RUN/'projection.md').write_text('\n'.join(lines).rstrip()+'\n')
print(json.dumps(dict(summary=v['summary'],pelvis_Y_reduction_percent=v['pelvis_abs_Y_error_reduction_percent']),indent=2))
