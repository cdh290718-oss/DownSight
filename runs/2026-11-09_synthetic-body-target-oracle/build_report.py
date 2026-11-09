"""Summarize the existing-body-target oracle and its limits without GT selection."""
import json,hashlib
from common import ROOT,RUN,OUT,read
m=read(RUN/'evidence/metrics.json');s=read(RUN/'evidence/shape_metrics.json');diag=read(RUN/'evidence/target_residuals.json');v=read(RUN/'evidence/verification.json');execution=read(RUN/'evidence/execution.json');assert v['exit_status']==0 and len(execution)==3 and all(x['exit_status']==0 for x in execution)
labels=['control_left','gt_hips_left','gt_body_left'];views=[];shape={x['case']:x for x in s['cases']}
fields=['body25_absolute_mean_mm','body_targets12_mean_mm','other_body13_mean_mm','pelvis_absolute_error_mm','knees_mean_mm','ankles_mean_mm','feet6_mean_mm','feet6_ankle_aligned_mean_mm','wrists_absolute_mean_mm','elbows_absolute_mean_mm','hands40_absolute_mean_mm','head_position_error_mm','head_orientation_error_deg']
for view in m['views']:
 stages=view['stages'];rows=[]
 for i,p in enumerate(stages['control_left']['per_joint']):
  rows.append(dict(index=i,name=p['name'],cases={label:{k:stages[label]['per_joint'][i][k] for k in ['absolute_mm','error_xyz_mm','pelvis_aligned_mm']} for label in labels}))
 views.append(dict(camera=view['camera'],side=view['side'],summary={label:{k:stages[label][k] for k in fields} for label in labels},joints=rows,saturated={label:stages[label]['saturated_pose_increments'] for label in labels}))
a=m['views'][0]['stages']['control_left'];b=m['views'][0]['stages']['gt_body_left'];changes={k:b[k]-a[k] for k in fields}
q=dict(exit_status=0,scope='One synthetic pose, same primary left seed. Body oracle replaces exactly 12 existing Sapiens target xyz. It does not add all 25 evaluation joints or all65 joints. WiLoR hand targets remain automatic.',views=views,shape=s['cases'],gt_body_minus_control=changes,body25_reduction_percent=100*(a['body25_absolute_mean_mm']-b['body25_absolute_mean_mm'])/a['body25_absolute_mean_mm'],validation_passed=True)
(RUN/'evidence/assessment.json').write_text(json.dumps(q,indent=2)+'\n')
names={'body25_absolute_mean_mm':'身体25点','body_targets12_mean_mm':'直接替换目标的12点','other_body13_mean_mm':'其余13点（含头面、脚部）','pelvis_absolute_error_mm':'骨盆','knees_mean_mm':'双膝','ankles_mean_mm':'双踝','feet6_mean_mm':'脚趾/脚跟6点','feet6_ankle_aligned_mean_mm':'脚部6点踝对齐后','wrists_absolute_mean_mm':'双腕','elbows_absolute_mean_mm':'双肘','hands40_absolute_mean_mm':'手指40点','head_position_error_mm':'Head位置','head_orientation_error_deg':'Head朝向（°）'}
text='''# 全部现有身体三维目标替换为真值\n\n本轮在当前双目共享人体、联合姿态/体型优化中，仅替换当前 Sapiens 身体目标的三维坐标。全部身体目标是双侧肩、肘、腕、髋、膝、踝共 12 点；不是增加全部25个身体评测点或全部65个关节。原本没有的骨盆、脚趾/脚跟、鼻等目标没有新增，WiLoR 的10个腕/掌指观测继续使用原自动值。\n\n[骨架与原图投影](projection.md) · [精确数值](evidence/assessment.json) · [逐目标诊断](evidence/target_residuals.json) · [控制条件验证](evidence/verification.json)\n\n## 结果\n\n以下为 cam3，除朝向外单位 mm。欧氏关节误差不作对齐，除非指标明确写踝对齐。两目共享同一人体，cam4 数值用于核对坐标，不是独立样本。\n\n| 指标 | 原自动目标 | 仅真值髋 | 全部身体目标真值 |\n|---|---:|---:|---:|\n'''
for k,name in names.items():text+='| '+name+' | '+' | '.join('%.4f'%m['views'][0]['stages'][label][k] for label in labels)+' |\n'
text+='\n12 点的误差在 gt_body 中属于已输入目标的拟合精度，不能当作检测器泛化精度。其余13点中仍有受真值 Head 强约束的头面点，也不能全部叫独立测试点；脚部6点和骨盆分别报告。\n'
text+='\n## 逐关节结果\n\n| 关节 | 绝对误差 原→全真值 mm | XYZ误差 原 mm | XYZ误差 全真值 mm |\n|---|---:|---|---|\n'
for i in [0,1,2,3,4,5,6,8,9,10,11,12,13,14,15,16,17,18,19]:
 old=a['per_joint'][i];new=b['per_joint'][i];text+='| '+old['name']+' | %.2f → %.2f | '%(old['absolute_mm'],new['absolute_mm'])+', '.join('%+.2f'%x for x in old['error_xyz_mm'])+' | '+', '.join('%+.2f'%x for x in new['error_xyz_mm'])+' |\n'
text+='\n## 体型、触限与剩余拟合误差\n\n| 条件 | 中性体型网格误差 mm | 触限关节 |\n|---|---:|---|\n'
for label in labels:text+='| '+label+' | %.2f | '%shape[label]['neutral_pelvis_centered_vertex_mean_mm']+', '.join(m['views'][0]['stages'][label]['saturated_pose_increments'])+' |\n'
text+='\n体型信息子空间及初始化完全相同；最终 beta 可以在同一子空间内响应新目标。中性体型误差在统一零姿态、各自骨盆居中后评测；本样本真值 beta 为零，均值先验有天然优势。触限仅指相对网络初值的优化增量边界，不是人体解剖极限。\n'
text+='\n### 保留的自动手部目标\n\n全身体真值组仍同时拟合自动 WiLoR 腕/掌指点。下表的目标偏差与拟合残差用于发现剩余冲突，不能把标量距离直接相加。\n\n| 自动手部目标 | 目标到真值 mm | 拟合到目标 mm | 拟合到真值 mm |\n|---|---:|---:|---:|\n'
for row in next(x for x in diag['cases'] if x['case']=='gt_body_left')['targets']:
 if row['source']=='wilor':text+='| '+row['name']+' | %.2f | %.2f | %.2f |\n'%(row['target_to_GT_mm'],row['fit_to_target_mm'],row['fit_to_GT_mm'])
text+='''\n## 实验控制\n\n- control、gt_hips、gt_body 均从原网络 raw 参数和原自动三角化观测开始，先用相同初始化函数计算根位姿及体型子空间，再覆盖损失目标。不是从旧 final 继续优化。\n- 初始化信息、体型基向量及所有 initial 参数与原 oracle_left 逐元素一致；本轮 control final 精确复现原 oracle_left，gt_hips final 精确复现上轮只换髋的结果。\n- settings.json 与原配置逐字节一致。相机、完整真值 Head 锚点、协方差、组权重、检测置信度、门控、点集、姿态先验、硬上限、学习率、3400 步预算和检查点选择均不改。\n- 变化仅为 2 或 12 个身体点的 xyz；对应的三角化质量与协方差元数据仍描述原自动观测，不能误称为真值点的不确定度。\n- 拟合进程禁止读取整个合成 GT dataset 或真值形状文件；各条件只能读取自己允许的目标提取文件。原条件不读身体真值；髋组只读两点；body组只读12点。真值体型、身体姿态、完整网格未输入优化。\n- 双目 Head 锚点与三维身体目标各只计一次。手部仍按一对一左右手匹配，使用自动三角化的腕和四个掌指点。没有新增轮廓、足部观测、碰撞、MANO 接缝或手指优化。\n\n点集和严格变化清单见 preparation.json、body_targets.json、hips_targets.json；loss 独立重算、原条件回归与读取清单见 verification.json。标量总 loss 在不同目标下不作为质量排名，检查点只按本组目标选择，不按真值误差挑选。\n\n## 解释边界与后续方向\n\n这个诊断能够检验当前观测坐标偏差对拟合的影响，但同时替换了12点，不能把变化拆成各关节的独立贡献。即便真值身体目标改善结果，也不能据此认定所有偏差都来自关节定义；检测定位、遮挡、匹配和投影方向误差仍需分开检验。\n\n它不是部署方法，也不是证明优化器在所有条件下正确：保留自动手目标、固定手指和足部未观测自由度、姿态/根位姿先验、关节限幅及有限迭代预算；相机仍使用准确生成标定、Head仍是真值，单姿态且真值体型为均值。现有12个身体点即使拟合准确，也不保证脚的朝向、人体轮廓、粗细或手指网格准确。\n\n### 本轮可以支持什么

1. **自动身体观测的坐标偏差是当前误差的重要来源。** 身体25点从35.51降到16.65 mm，约下降53.1%；双膝56.88→14.93 mm，双踝52.82→23.51 mm。cam3下，膝踝剩余Y误差仅约−5至−7 mm，而不是此前的数厘米。仅换髋的条件对膝踝几乎没有改善，说明只修正骨盆位置不足以解决下肢问题。证据：assessment.json、metrics.json。
2. **准确目标不等于当前优化必然准确恢复全部关节。** 已替换的12点仍有11.97 mm平均误差；双膝Z误差约−13至−15 mm，双踝约−22至−23 mm。这里保留原协方差和鲁棒损失，是软目标，不是强制插值。踝点白化残差范数仅约0.14–0.18，仍允许与姿态先验等折中；不能用loss数值大小代替梯度分析或断言某项是唯一原因。证据：target_residuals.json。
3. **还没有排除优化预算与关节限幅。** 最佳检查点是最后的3400步，3000→3400步loss为0.00067733→0.00059069，仍持续下降。故16.65 mm不是充分收敛的误差下限。Neck、Head、双侧Collar、双腕增量触限；髋、膝、肘没有触限，不能把腿部残差直接归因于腿的硬上限。证据：optimization_summary.json。
4. **体型和姿态仍存在相互补偿。** 中性体型网格误差5.68→11.15 mm，骨盆也没有达到仅真值髋的4.79 mm，而是9.31 mm。目标变准后关节总体更准，但体型没有同步改善，不能说体型估计已经解决。尚未收敛、先验、有限形状子空间和姿态限制均可能参与，当前实验没有分离这些原因。证据：shape_metrics.json、assessment.json。
5. **脚部缺少朝向观测。** 脚部六点绝对误差虽降到42.07 mm，踝对齐后仍为30.24 mm，甚至高于原来的26.73 mm。脚趾/脚跟没有进入损失；当前两个踝的旋转增量均为零，L_Foot/R_Foot也不是活跃优化关节。仅用踝位置不能决定脚尖方向，因而不能指望换准膝踝点就自动修好脚。脚部残差还可能包含体型影响，并非证明全是朝向问题。证据：optimization_summary.json、metrics.json及fit.py。
6. **腕部更准不代表手指更准。** 双腕6.14→1.57 mm，但手指40点20.97→24.33 mm，增加3.36 mm。保留的自动WiLoR腕点与真值腕点分别相差12.61/8.29 mm，掌指点也有偏差；固定手指参数、腕增量触限仍在。数据支持存在剩余目标不一致，尚不能将手指变差唯一归因于其中一项。证据：target_residuals.json、metrics.json。

### 下一步优化优先级（尚未执行）

1. **先检查同一目标的收敛，再讨论新约束。** 保留本轮3400步结果作为等预算对照，另做只延长预算的诊断，报告训练loss、参数变化及真值评测轨迹，检查是否稳定。停止与选点规则不得按真值误差挑选。若稳定后仍有明显身体残差，再分别检查已触限的颈/头/锁骨增量，以及体型补偿；一次只变一个因素。例如固定真值体型仅作诊断，不能当作部署改进。保留相同先验尺度，避免放宽硬上限时顺便削弱先验。
2. **将观测校正作为实际部署改进的重要方向，但先验证是否存在稳定偏差。** 用更多姿态、深度和非均值体型，分别统计自动2D观测与对应SMPL-X点真值投影的方向偏差，区分检测定位、遮挡、左右匹配和语义定义。按人/姿态划分独立验证集；校正的输入只能使用部署时可获得的特征。若偏差稳定再做标定式校正，若随遮挡和姿态变化较大再考虑针对性训练。不能直接把当前单帧偏移推广到真实数据。
3. **为脚和手补充局部信息。** 身体确定后，先以真值脚趾/脚跟相对踝位置做诊断，验证局部踝/足旋转自由度能否改善；再换可靠自动足部观测。手部单独检查朝向与手指姿态，保持已准确的腕肘位置，避免为匹配有偏掌指点再次扭动身体。局部阶段是否有效必须另做对照，不能由本轮直接认定。

这些结果支持优先修正身体观测，同时检查优化是否收敛；不支持“所有剩余误差都是关节定义差异”，也不支持继续只靠增加全局约束来解决未观测的脚和手自由度。\n\n## 执行与审计\n\n实际仓库基线及远端 main 为 `3d092a62a9558ed9193ae4288897fdc818a60709`。运行时含本轮新增脚本和既有无关未跟踪 point_comparison_followup.json，后者未改动或提交；上轮 review 待 Claude。实际执行代码与设置哈希、命令、退出状态见 execution.json，评测/绘图命令见 evaluation_execution.json。最多两个 CPU 子进程，各四线程，没有 GPU 推理或环境安装。\n\n输入沿用 `DownSightSmplxdata@192671271a74b99f41b9eb033d8f88b1dac77ada`、同一姿态的 cam3/cam4；原始模型和输入访问条件见[候选基线资源说明](../../experiments/synthetic_oracle_head_shape_20261108/REPRODUCE.md)。当前提取目标源文件与哈希见 body_targets.json，其他复制资源哈希见 preparation.json。原始 NPZ、图像、权重留在外部资产或 `<WORKSPACE>/DownSight/outputs/synthetic_body_target_oracle_20261109`，不随公开仓库提交。无法访问这些资产不能声称独立复现。\n\n本轮检查初始化、子空间、限幅、梯度有限、参数重建、投影、独立损失重算、相机坐标一致、执行源文件不变及两组精确回归。完整结果保留在 verification.json、optimization_summary.json、fit_log_summary.json；继承环境的非致命警告保留，不将它们隐藏成没有日志。\n\n发布前检查工作区、暂存、本次提交与 origin/main。增量仅含代码、配置、报告、精简数值证据和派生图；不含权重、数据集、视频、NPZ/NPY、完整原始输出、凭据、.env、完整 provenance/patch；单文件小于50 MB。机器路径以 `<WORKSPACE>` / `<BODY_PYTHON>` 等占位符表示。工作区/暂存检查见 evidence/prepush.json，完整收据在本轮 outputs；提交检查通过后再 push，仅覆盖当前增量。\n'''
(RUN/'report.md').write_text(text)
lines=['# 全身体目标真值：骨架与投影对比','', '[返回报告](report.md)','', '绿：真值；蓝：原条件或仅真值髋；橙：全部12个身体目标真值。每个相机的两幅对比使用相同坐标范围，侧视横纵单位比例一致。手部观测仍为自动值。','']
for side in ['left','right']:
 for before in ['control','gt_hips']:lines+=['## '+side+' / '+before+' 对比全身体真值','','![skeleton '+side+' '+before+'](assets/skeleton_'+side+'_'+before+'.png)','']
lines+=['## 原图投影','', '绿点为真值、红点为预测；半透明预测网格。部分跨相机平面或过长投影三角形会被渲染器略去。','']
for side in ['left','right']:
 for label in labels:lines+=['### '+side+' / '+label,'','!['+side+' '+label+'](assets/'+side+'_'+label+'.jpg)','']
(RUN/'projection.md').write_text('\n'.join(lines).rstrip()+'\n')
print(json.dumps(q['views'][0]['summary'],indent=2))
