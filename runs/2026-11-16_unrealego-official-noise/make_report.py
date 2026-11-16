"""Render checked numeric evidence to Markdown. No model or GT data are read here."""
import json
from pathlib import Path
p=Path(__file__).resolve().parent;read=lambda x:json.loads((p/'evidence'/x).read_text());s=read('summary.json');e=read('execution.json');verification=read('verification.json');screen=read('screening_reasons.json')
fmt=lambda x:'—' if x is None else f'{x:.2f}'
lines=['# UnrealEgo 官方二维加噪：三种损失比较','','拟合目标为官方 GT15 二维点加独立高斯噪声；四对样本，每个噪声水平使用三个固定种子。非 Head 身体三维标注在全部拟合退出后仅用于评测。Head 位置为标注、朝向为历史估计，双目相对位姿来自其他序列标注配准，因此不是无标注部署结果。','','## 每个噪声水平汇总','','表中各值为四对、三个种子的均值；GT15 为欧氏距离均值，不是 RMSE。二维投影误差相对于无噪官方二维标注，覆盖全部 GT15。','', '| 噪声标准差 px | 方法 | PA 三维 mm | 不对齐三维 mm | 左 / 右二维 px | 成功 / 应有 |','|---:|---|---:|---:|---:|---:|']
for x in s['summary']:
 v=x['values']; name={'3d':'三维门限','mixed':'混合','2d':'全二维'}[x['method']]
 if v:lines.append(f"| {x['sigma_px']} | {name} | {v['pa15_mm']:.2f} | {v['absolute15_mm']:.2f} | {v['left_clean_px']:.2f} / {v['right_clean_px']:.2f} | {v['count']} / 12 |")
 else:lines.append(f"| {x['sigma_px']} | {name} | — | — | — | 0 / 12 |")
lines+=['','## 每对样本（各三个种子的均值）','','| 噪声 px | 样本 | 方法 | PA 三维 mm | 不对齐三维 mm | 左 / 右二维 px | 成功种子 |','|---:|---|---|---:|---:|---:|---:|']
for x in s['per_pair']:
 v=x['values'];name={'3d':'三维门限','mixed':'混合','2d':'全二维'}[x['method']]
 if v:lines.append(f"| {x['sigma_px']} | {x['pair']} | {name} | {v['pa15_mm']:.2f} | {v['absolute15_mm']:.2f} | {v['left_clean_px']:.2f} / {v['right_clean_px']:.2f} | {v['count']} / 3 |")
 else:lines.append(f"| {x['sigma_px']} | {x['pair']} | {name} | — | — | — | 0 / 3 |")
counts={k:(v.get('3d_accepted',0),v.get('geometry_invalid',0),v.get('depth_unreliable',0)) for k,v in screen.items()};lines+=['','## 解释与边界','',f"每个噪声水平有 180 个输入关节；按照 1、3、5、8 px 的顺序，通过三维门限的计数分别为 {', '.join(str(counts[str(k)][0]) for k in [1,3,5,8])}。未通过的有效二维点在混合方案中退回二维，而不会同时计入两种损失。几何失败和深度不可靠的完整分布见 [screening_reasons.json](evidence/screening_reasons.json)。",'',f"所有条件使用相同阈值和同一输入条件的体型子空间；体型方向随观测和噪声改变，并不保证能够恢复准确体型。设置与噪声假设见 [protocol.json](protocol.json)。三维协方差还包含关节定义噪声，但二维分支未单独建模该项；标准化并不保证两者的真实统计分布完全相同。",'',f"共 {len(e['records'])} 条条件记录，复用了 {verification['aliases']} 条与三维分支完全相同的混合结果；实际失败数为 {len(e['failures'])}。本实验为四对同一受试者的小样本；阈值、关节定义、相机标定和 Head 估计均影响结论，不能推断真实自动检测器的性能。各组成功数如不足应有数量，汇总只包含成功拟合，不能当作完整组均值。",'', '每次在固定预算内按拟合目标选取可行检查点，不使用 GT 选点。收敛诊断、每个种子的逐关节误差和相对输入点的二维残差见 [results.json](evidence/results.json)；汇总的离散度、收敛通过数和每对记录见 [summary.json](evidence/summary.json)。标注可见性没有独立核验，脚点沿用 heel proxy。','', '拟合基线为 `5281634c1bac19a27d5bdf0130d395230521446e`，代码为本轮工作树。拟合前准备和噪声检查见 [input_hashes.json](evidence/input_hashes.json)、[noise_check.json](evidence/noise_check.json)；执行和体型子空间检查见 [execution.json](evidence/execution.json)、[verification.json](evidence/verification.json)。原始数据、模型及完整输出存于本机。','', '第一批逐点投影耗时较高，已在本轮主动中止并保留 [中止记录](evidence/vectorization_interruption.json)；改用批量投影后重新运行，数值和梯度对照见 [vectorization_smoke.json](evidence/vectorization_smoke.json)、[vectorization_gradient.json](evidence/vectorization_gradient.json)。筛选统计初次写入路径错误的失败见 [screening_initial_failure.txt](evidence/screening_initial_failure.txt)。','','本报告为自动依据证据生成；仍需审阅和推送前自检。']
(p/'report.md').write_text('\n'.join(lines)+'\n');print('report generated')
