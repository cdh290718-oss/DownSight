"""Render comparisons and apply the user's frozen wrist acceptance rule."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--metrics',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--decision',type=Path,required=True);a=p.parse_args();d=json.loads(a.metrics.read_text());old=json.loads(a.previous.read_text());decisions={}
for m in ('v0','wilor'):
 for joint in ('left_wrist','right_wrist','left_elbow','right_elbow'):
  for subset in ('all','new160'):
   for metric in ('median_cm','p90_cm','failure_fraction_gt5_or_missing'):
    x=d['variants'][m]['body'][joint][subset][metric];y=old['variants'][m]['body'][joint][subset][metric];assert abs(x['value']-y['value'])<1e-10 and all(abs(u-v)<1e-10 for u,v in zip(x['ci95'],y['ci95'])),(m,joint,subset,metric)
def stat(s):return f"{s['value']:.2f} [{s['ci95'][0]:.2f}, {s['ci95'][1]:.2f}]"
s=['# 三方评测表','','方括号是同一序列、10 帧移动块 bootstrap 95% 区间（2000 次、种子 0）；分位数仅使用有效预测，失败率把缺失预测计入。','', '## 腕和肘：DFM 三维距离','','| 子集 | 方法 | 关节 | 有效预测 / 参考 | 中位数 cm [95%] | p90 cm [95%] | >5 cm 或缺失 % |','|---|---|---|---:|---:|---:|---:|']
for subset in ('all','new160'):
 for mode in ('v0','wilor','two_stage'):
  for j in ('left_wrist','right_wrist','left_elbow','right_elbow'):
   x=d['variants'][mode]['body'][j][subset];s.append(f"| {subset} | {mode} | {j} | {x['prediction_valid']} / {x['reference_valid']} | {stat(x['median_cm'])} | {stat(x['p90_cm'])} | {100*x['failure_fraction_gt5_or_missing']['value']:.2f} |")
for subset in ('all','new160'):
 decisions[subset]={}
 for side in ('left','right'):
  x=d['variants']['v0']['body'][side+'_wrist'][subset];y=d['variants']['two_stage']['body'][side+'_wrist'][subset];decisions[subset][side]={}
  for key in ('median_cm','p90_cm'):
   v=y[key]['value'];ci=x[key]['ci95'];decisions[subset][side][key]={'value_cm':v,'v0_95ci_cm':ci,'within_v0_interval':ci[0]<=v<=ci[1]}
s += ['', '## 相对腕的 21 点和掌面朝向','','| 子集 | 方法 | 手 | 有效帧数 | 21 点平均差中位 / p90 cm | 掌面角中位 / p90 度 |','|---|---|---|---:|---:|---:|']
for subset in ('all','new160'):
 for mode in ('v0','wilor','two_stage'):
  for side in ('left','right'):
   x=d['variants'][mode]['hand'][side][subset];s.append(f"| {subset} | {mode} | {side} | {x['valid_21_point_count']} | {stat(x['mean_relative_21_point_error_cm_median'])} / {stat(x['mean_relative_21_point_error_cm_p90'])} | {stat(x['palm_rotation_error_deg_median'])} / {stat(x['palm_rotation_error_deg_p90'])} |")
s += ['', '手部各指标的有效帧与定义同前一轮，数据为相对 DFM；肘仅供参考，用户指出 DFM 肘点存在已知系统偏差，本轮未独立量化，不用于验收。']
a.output.write_text('\n'.join(s)+'\n');a.decision.write_text(json.dumps(decisions,indent=2)+'\n')
