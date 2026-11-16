import json,statistics
from pathlib import Path
p=Path(__file__).resolve().parent;rows=json.loads((p/'evidence/results.json').read_text())['records']; summaries=[];sample_rows=[]
for sigma in [1,3,5,8]:
 for method in ['3d','mixed','2d']:
  chosen=[r for r in rows if r['condition'].startswith(f'gt_s{sigma}_') and r['method']==method and r['status']=='ok'];bad=[r for r in rows if r['condition'].startswith(f'gt_s{sigma}_') and r['method']==method and r['status']!='ok']
  def agg(a):
   if not a:return None
   mean=lambda v:statistics.mean(v)
   return {'count':len(a),'pa15_mm':mean(r['pa15_mm'] for r in a),'absolute15_mm':mean(r['absolute15_mm'] for r in a),'left_clean_px':mean(r['views'][0]['clean_gt15_mean_px'] for r in a),'right_clean_px':mean(r['views'][1]['clean_gt15_mean_px'] for r in a),'pa_sd_mm':statistics.stdev(r['pa15_mm'] for r in a) if len(a)>1 else 0,'convergence_pass':sum(r['convergence']['diagnostic_pass'] for r in a)}
  summaries.append({'sigma_px':sigma,'method':method,'values':agg(chosen),'failures':len(bad)})
  for pair in [f'body{i:02d}' for i in range(1,5)]:sample_rows.append({'sigma_px':sigma,'method':method,'pair':pair,'values':agg([r for r in chosen if r['pair']==pair])})
result={'expected_per_group':12,'summary':summaries,'per_pair':sample_rows,'note':'PA mean is the mean of 15 point distances after similarity alignment; std across four pairs and seeds is descriptive, not a confidence interval.'};(p/'evidence/summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
for r in summaries:print(r['sigma_px'],r['method'],r['values'],'failures',r['failures'])
