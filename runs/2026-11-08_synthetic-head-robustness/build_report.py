"""Report all predeclared cases; no GT-selected showcase."""
import json,hashlib,subprocess
import numpy as np
from common import ROOT,RUN,OUT,read
m=read(RUN/'evidence/metrics.json');s=read(RUN/'evidence/shape_metrics.json');v=read(RUN/'evidence/verification.json');executions=read(RUN/'evidence/execution.json');settings=read(RUN/'settings.json');methods=read(RUN/'evidence/method_tests.json');cases=read(RUN/'cases.json');metrics=m['views'][0]['stages'];shape={r['case']:r for r in s['cases']}
keys=['body25_absolute_mean_mm','pelvis_absolute_error_mm','wrists_absolute_mean_mm','elbows_absolute_mean_mm','hands40_absolute_mean_mm','head_position_error_mm','head_orientation_error_deg']
compact=[]
for name,r in metrics.items():
 compact.append(dict(case=name,**{k:r[k] for k in keys},neutral_shape_mean_mm=shape[name]['neutral_pelvis_centered_vertex_mean_mm'],beta_rmse=shape[name]['betas_error_rmse'],pelvis_error_xyz_mm=r['pelvis_error_xyz_mm'],saturated_pose_increments=r['saturated_pose_increments']))
rows={r['case']:r for r in compact};comparison={}
for head in ['oracle','none']:
 new=rows[head+'_left'];old=rows['legacy_'+head+'_left'];comparison[head]={k:new[k]-old[k] for k in keys+['neutral_shape_mean_mm','beta_rmse']}
perturb={}
for mm in [10,20]:
 group=[r for r in compact if r['case'].startswith('perturb_'+str(mm)+'_')];perturb[str(mm)]={k:dict(mean=float(np.mean([r[k] for r in group])),min=float(min(r[k] for r in group)),max=float(max(r[k] for r in group))) for k in keys+['neutral_shape_mean_mm']}
assessment=dict(exit_status=0,primary_rows=compact,improved_minus_legacy= comparison,perturbation_summary=perturb,rank=methods['initializations']['left']['retained_rank'],all_execution_exit_zero=all(x['exit_status']==0 for x in executions),all_verification_passed=v['exit_status']==0)
(RUN/'evidence/assessment.json').write_text(json.dumps(assessment,indent=2)+'\n')
def table(names):
 lines=['| 配置 | 身体25 | 骨盆 | 腕 | 肘 | 手指40 | 中性体型网格 | 头位置 / 朝向 |','|---|---:|---:|---:|---:|---:|---:|---:|']
 for name in names:
  r=rows[name];lines.append('| '+name+' | '+' | '.join('%.2f'%r[k] for k in keys[:5]+['neutral_shape_mean_mm'])+' | %.2f mm / %.2f° |'%(r['head_position_error_mm'],r['head_orientation_error_deg']))
 return '\n'.join(lines)
main=['legacy_oracle_left','oracle_left','legacy_none_left','none_left','oracle_right','none_right'];external=['B_previous','E_previous','D_oracle_shape'];perturb_names=[c['case']+'_left' for c in cases if c['case'].startswith('perturb_')]
text='''# 共享体型优化与头部锚点鲁棒性\n\n本轮已完成全部预定拟合。使用同一对合成图：cam3/cam4 是同一姿态的两目，不是两个独立样本。\n\n[查看全部投影图](projection.md) · [方法和限制](method.md) · [数值汇总](evidence/assessment.json) · [逐关节评测](evidence/metrics.json)\n\n## 结果\n\n下表单位为 mm（朝向除外）。身体、腕、肘、手指均为相机坐标下绝对误差，不做骨盆/尺度对齐；中性体型网格为去姿态、各自骨盆居中的平均顶点误差。主初值为 left，right 为敏感性检查，不选优。\n\n'''+table(main)+'\n\n新方法减旧方法的差值（负数表示改善，均为同一新初始化）：\n\n'
for head in ['oracle','none']:
 d=comparison[head];text+='- %s：身体 %.2f mm，体型 %.2f mm，腕 %.2f mm，手指 %.2f mm。\n'%(head,d['body25_absolute_mean_mm'],d['neutral_shape_mean_mm'],d['wrists_absolute_mean_mm'],d['hands40_absolute_mean_mm'])
text+='\n本轮整套改动在两种匹配对照下均降低了身体、体型、腕和手指误差；这支持继续验证此方向，但不能区分三维目标和子空间先验各自的贡献。真值头部主组身体误差已经接近旧 D（见下表），不表示识别了非均值个体体型。\n\n'
text+='无锚点主组头部仍偏差 %.2f mm / %.2f°，另一初值为 %.2f mm / %.2f°，显示头部朝向仍缺少约束。真值头部条件换初值后，中性体型误差从 %.2f 到 %.2f mm，也说明体型/姿态耦合和初值敏感性仍未完全解决。\n'%(rows['none_left']['head_position_error_mm'],rows['none_left']['head_orientation_error_deg'],rows['none_right']['head_position_error_mm'],rows['none_right']['head_orientation_error_deg'],rows['oracle_left']['neutral_shape_mean_mm'],rows['oracle_right']['neutral_shape_mean_mm'])
text+='\n### 扰动锚点\n\n平移范数 10/20 mm；均伴随 3° 旋转，两个预定方向各取正负。全部 8 组如下，不能当作佩戴误差的统计置信区间。\n\n'+table(perturb_names)+'\n\n| 扰动范数 | 身体均值 [最小, 最大] mm | 体型均值 [最小, 最大] mm | 腕均值 [最小, 最大] mm |\n|---|---|---|---|\n'
for mm,q in perturb.items():text+='| '+mm+' mm | '+' | '.join('%.2f [%.2f, %.2f]'%(q[k]['mean'],q[k]['min'],q[k]['max']) for k in ['body25_absolute_mean_mm','neutral_shape_mean_mm','wrists_absolute_mean_mm'])+' |\n'
text+='\n锚点扰动后，头部基本继承了输入偏差，而不是由双目自动纠正；同时腕和手指误差变化较小。此结果支持检验实测安装锚点的路线，不能用真值扰动代替实测安装误差测试。\n\n### 旧结果外部参考\n\n这些结果的初始化和目标设置不同，不作新方法的单变量对照。D 使用真值体型和真值头部；E 为此前自由 10 维体型 + 0.001 均值先验。\n\n'+table(external)
text+='''\n\n## 方法究竟改变了什么\n\n完整参数见 [settings.json](settings.json) 和 [method.md](method.md)。新方法使用三角化点的完整三维协方差、稳健分组损失、腕和掌指点，并限制弱可观测体型方向。旧方法对照保留原二维目标和 10 维 beta；本轮两者均使用相同的双目观测初始化，取消 0.85 腕距下限。各组均无全网格前方 2 cm 约束、无鼻点，保留肩点。没有轮廓损失或 MANO 融合。\n\n'''
eig=methods['initializations']['left']['shape_information_eigenvalues'];text+='预设信息阈值 1，得到 **%d 个可优化体型方向**；最大三个信息特征值 %.6f、%.6f、%.6f。其余方向固定在模型均值。依据见 [method_tests.json](evidence/method_tests.json)。该信息近似忽略共享端点相关性和姿态耦合，不是完整统计可观测性证明。\n\n'%(assessment['rank'],*eig[-3:])
text+='''## 能说明什么、不能说明什么\n\n- **none 组不使用人体真值，包括初始化。** 拟合进程不读任何头部锚点文件或 GT dataset；原始网络参数、自动二维检测和已知生成相机是输入。oracle 和 perturb 分别只读获准的独立锚点。JSON/NPZ 打开记录（包括输出写入）和相同初始化检查见 [verification.json](evidence/verification.json)。这不等于已在真实设备上部署验证：相机内外参与米制基线仍假定准确。\n- **真值 beta 全为零。** 把弱信息方向拉回均值会得到天然优势；体型误差降低不能证明识别了个体体型。还需要非均值身形、多姿态、真实标定误差测试。不能根据本样本改选更强先验。\n- 对照同时改变三维加权、手点选择和体型子空间，不能把收益单独归给其中一项。肩/髋定义噪声、像素噪声均是工程假设，未做检测器噪声标定。\n- 手指姿态固定；评测保留 40 个手指关节，但改进目标只用每手腕和 4 个 MCP。表中的手部指标决定是否发生退化，不用身体改善代替手部验收。\n- 扰动组使用原真值人为扰动来模拟测量，仍不是独立实测锚点；固定 8 个方向只覆盖有限情况。\n\n## 执行、输入、验证\n\n基线与实际仓库 HEAD：`9012bbf856570e85da60261190ead1092cce9997`。运行时工作树有本轮新增脚本/报告，另外存在此前未跟踪 point_comparison_followup.json，未改动、未提交。实际执行文件 SHA-256 在 [execution.json](evidence/execution.json)；本轮源码随本报告提交。运行时不修改拟合脚本、settings、cases。\n\n输入沿用前轮 Sapiens/WiLoR 自动检测、左右网络 raw 初始化及生成相机。原始数据源 DownSightSmplxdata commit `192671271a74b99f41b9eb033d8f88b1dac77ada`，资源访问与后端说明沿用前轮报告。新目标与输入源 SHA-256 见 [triangulated.json](evidence/triangulated.json)，参数输出 SHA-256 见 optimization_summary.json 和 metrics.json。原始图、NPZ、权重和完整输出只保存在 `<WORKSPACE>/DownSight/outputs/synthetic_head_robustness_20261108`，不随仓库发布。无法访问这些资源的审核者不能声称独立复现。\n\n14 次拟合，最多两个 CPU 子进程，各使用 4 个计算线程；没有占用 GPU。逐次命令、退出码和耗时见 execution.json。几何检查包含中性关节 shape 导数、Kabsch 到 SMPL-X 参数重建和协方差正定性；运行中检查所有梯度有限、参数重建、投影一致、姿态限幅；后验验证重新计算各组协方差损失、检查子空间和无真值读取。初次导数测试漏传后端必需手/脸参数导致异常，已修正后通过，保留在 [development_failures.json](evidence/development_failures.json)。所有预定拟合结果均保留，无失败结果筛除。\n\n复现入口：依次运行 prepare.py、test_method.py、run.py、evaluate.py、evaluate_shape.py、verify.py、build_report.py；使用既有后端 Python 环境、PYTHONPATH=<REPO>、PYTHONDONTWRITEBYTECODE=1。outputs 目录需预建 runtime/cache、runtime/matplotlib，prepare 要求新目录防止覆盖旧结果。详细拟合命令见执行证据。评测命令/退出状态见 evidence/evaluation_execution.json，精简运行日志与非致命依赖/性能警告见 evidence/fit_log_summary.json。\n\n## 发布自检\n\n本轮仅发布代码、设置、报告、精简数值证据和派生投影 JPG。不含权重、原始数据集、视频、NPZ/NPY 或完整原始输出；单文件 <50 MB。开发机绝对路径以 `<WORKSPACE>` / `<BODY_PYTHON>` 表示；不发布完整 provenance/patch。自动检查工作区、暂存和提交增量、路径/凭据模式、图片 EXIF、远端 main；工作区、暂存审核记录见 evidence/prepush.json；提交审核收据留在本地，只有提交审核通过才会 push。仅审计本轮增量，不表示整个仓库历史已通过。\n'''
(RUN/'report.md').write_text(text)
projection=['# 投影对比','', '所有图均显示同一合成双目样本。绿色是真值关节，红色是预测关节，半透明为预测网格；头部着色单独显示。近相机平面相交或边长超过渲染阈值的三角形会被略去，图不替代数值评测。','', '[返回报告](report.md)','']
for name in main+perturb_names+external:
 projection+=['## '+name,'']
 for view in m['views']:
  q=view['stages'][name];projection+=['### '+view['camera'],'','!['+name+' '+view['camera']+']('+q['image']+')','']
(RUN/'projection.md').write_text('\n'.join(projection).rstrip()+'\n')
print(json.dumps(assessment,indent=2))
