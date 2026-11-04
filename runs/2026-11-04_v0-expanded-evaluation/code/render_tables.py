"""Render deterministic Markdown tables from evaluated numerical evidence."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--metrics',type=Path,required=True);p.add_argument('--groups',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();d=json.loads(a.metrics.read_text());g=json.loads(a.groups.read_text())
def show(x,percent=False):
 if x is None:return '—'
 mul=100 if percent else 1
 return f"{x['value']*mul:.2f} [{x['ci95'][0]*mul:.2f}, {x['ci95'][1]*mul:.2f}]"
s=['# 评测表','', '方括号为同一序列内、长度 10 帧的循环移动块 bootstrap 95% 区间（2000 次，种子 0），不是跨序列或跨人的置信区间。每侧参考有效帧为分母，缺失预测计入 >5 cm 失败率；中位数和 p90 只对有效预测计算。','','## 腕肘三维差与失败率','','| 子集 | 方法 | 关节 | 参考 / 预测数 | 中位数 cm [95%] | p90 cm [95%] | >5 cm 或缺失 % [95%] |','|---|---|---|---:|---:|---:|---:|']
for subset in ('all','new160'):
 for mode in ('v0','wilor'):
  for name in ('left_wrist','right_wrist','left_elbow','right_elbow'):
   x=d['variants'][mode]['body'][name][subset];s.append(f"| {subset} | {mode} | {name} | {x['reference_valid']} / {x['prediction_valid']} | {show(x['median_cm'])} | {show(x['p90_cm'])} | {show(x['failure_fraction_gt5_or_missing'],True)} |")
s += ['', '## 腕相对手部及掌面朝向', '', '| 子集 | 方法 | 手 | 21 点有效数 | 相对腕 21 点平均差：中位 / p90 cm | 掌面方向有效数 | 掌面旋转角：中位 / p90 度 |', '|---|---|---|---:|---:|---:|---:|']
for subset in ('all','new160'):
 for mode in ('v0','wilor'):
  for side in ('left','right'):
   x=d['variants'][mode]['hand'][side][subset];s.append(f"| {subset} | {mode} | {side} | {x['valid_21_point_count']} | {show(x['mean_relative_21_point_error_cm_median'])} / {show(x['mean_relative_21_point_error_cm_p90'])} | {x['valid_palm_count']} | {show(x['palm_rotation_error_deg_median'])} / {show(x['palm_rotation_error_deg_p90'])} |")
s += ['', '21 点误差先减去各自的腕位置，再在 21 点上取三维距离平均；掌面方向由腕、食指 MCP、小指 MCP 的局部坐标系比较得到，包含手指形状和参考定义的影响。', '', '## 失败分组', '', '| 方法 | 参考定义 | 独立帧数 / 腕帧数 | >5 cm 或缺失 | 示例 |', '|---|---|---:|---:|---|']
for mode in ('v0','wilor'):
 for name,x in g[mode].items():
  ex=' · '.join(f"[{e['frame']:05d} {e['side']}](failure_examples/{e['image']})" for e in x['examples']) or '—'; s.append(f"| {mode} | {name} | {x['count_frames']} / {x['count_wrist_frames']} | {x['gt5_or_missing']} | {ex} |")
s += ['', '手部点小于 5 cm 的分组是查看预定腕距分组后增加的探索性诊断，阈值不是接触标注；两组均只作为事后参考分析，不用于拟合。']
a.output.write_text('\n'.join(s)+'\n')
