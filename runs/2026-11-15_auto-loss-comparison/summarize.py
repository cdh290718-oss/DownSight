import json,statistics
from pathlib import Path
p=Path(__file__).resolve().parent; data=json.loads((p/'evidence/results.json').read_text())['records']; summary=[]
for method in ['3d','mixed','2d']:
 rows=[x for x in data if x['method']==method and x['status']=='ok']
 if rows:summary.append({'method':method,'pairs':[x['pair'] for x in rows],'pa15_mean_mm':statistics.mean(x['pa15_mm'] for x in rows),'absolute15_mean_mm':statistics.mean(x['absolute15_mm'] for x in rows),'left_clean_mean_px':statistics.mean(x['views'][0]['clean_gt15_mean_px'] for x in rows),'right_clean_mean_px':statistics.mean(x['views'][1]['clean_gt15_mean_px'] for x in rows),'convergence_pass_count':sum(x['convergence']['diagnostic_pass'] for x in rows)})
(p/'evidence/summary.json').write_text(json.dumps({'rows':summary,'warning':'3d has only three pairs; its mean is not directly comparable to four-pair means'},indent=2)+'\n')
for r in data:
 if r['status']!='ok': print(r['pair'],r['method'],r['status']);continue
 print(r['pair'],r['method'],f"{r['pa15_mm']:.2f}",f"{r['absolute15_mm']:.2f}",f"{r['views'][0]['clean_gt15_mean_px']:.2f}",f"{r['views'][1]['clean_gt15_mean_px']:.2f}",f"{r['views'][0]['input_mean_px']:.2f}" if r['views'][0]['input_mean_px'] is not None else '-',f"{r['views'][1]['input_mean_px']:.2f}" if r['views'][1]['input_mean_px'] is not None else '-',r['selected_step'],r['shape_rank'],r['convergence']['diagnostic_pass'])
