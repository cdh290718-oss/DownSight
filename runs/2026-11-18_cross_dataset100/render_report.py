"""Insert audited metric tables into the run report without hand-entering numbers."""
import json
from pathlib import Path
RUN=Path(__file__).resolve().parent;s=json.loads((RUN/'evidence/summary.json').read_text());models=['original','rw500','rw5000','rw20000'];datasets=['egopw','sceneego','unrealego2'];labels={'egopw':'EgoPW（伪标签）','sceneego':'SceneEgo','unrealego2':'UnrealEgo2（CityPark）'}
parts=[(RUN/'results_table.md').read_text(),'\n三组各 100 张、四模型共 1200 次图像推理全部完成，无失败，无按预测结果替换样本。\n','各数据集的排序：\n']
for d in datasets:
 best=min(models,key=lambda m:s[d][m]['mean_px']);pckbest=max(models,key=lambda m:s[d][m]['correct_20px']);v=s[d][best];orig=s[d]['original'];relative=100*(1-v['mean_px']/orig['mean_px'])
 parts.append(f"- {labels[d]}：平均误差最低为 {best}，{v['mean_px']:.2f} px，相对原模型下降 {relative:.1f}%；20 px 内点数最高为 {pckbest}，{s[d][pckbest]['correct_20px']} / {v['points']}。")
parts.append('\n以上排序只对应固定子集与其标注口径，不代表完整官方 benchmark 或统计显著结论。EgoPW 的改善是伪标签一致性改善；更接近伪标签不必然更接近真实关节。各数据集分开报告，不能据其绝对像素误差直接比较数据集难度。')
parts.append('\n### 各关节结果\n\n单元格为“平均误差 px / 20 px 内点数”。每关节实际分母及出画面计数见 summary.json；分母仅排除参考点不在图内的情况。\n')
for d in datasets:
 parts.append(f'\n**{labels[d]}**\n');parts.append('| 关节 | 原模型 | 500 | 5000 | 20000 |\n|---|---:|---:|---:|---:|')
 for j in s[d]['original']['per_joint']:
  parts.append('| '+j+' | '+' | '.join(f"{s[d][m]['per_joint'][j]['mean_px']:.2f} / {s[d][m]['per_joint'][j]['correct_20px']}" for m in models)+' |')
p=RUN/'report.md';text=p.read_text();start=text.index('## 结果\n')+len('## 结果\n');end=text.index('\n## 执行与异常',start);p.write_text(text[:start]+'\n'+'\n'.join(parts)+'\n'+text[end:]);print('Report metric tables generated from summary evidence')
