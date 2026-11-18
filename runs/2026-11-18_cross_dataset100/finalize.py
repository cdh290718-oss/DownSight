"""Recompute public metrics from local predictions, preserving all samples and masks."""
import json,math,hashlib,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];RUN=Path(__file__).resolve().parent;OUT=ROOT/'outputs/cross_dataset100_20261118';MODELS=['original','rw500','rw5000','rw20000'];DATASETS=['egopw','sceneego','unrealego2']
summary={};details={};checks={}
for dataset in DATASETS:
 manifest=json.loads((OUT/(dataset+'_manifest.json')).read_text());assert len(manifest)==100;reference={r['id']:r for r in manifest};summary[dataset]={};details[dataset]={}
 assert len({r['image_sha256'] for r in manifest})==100
 for model in MODELS:
  rows=json.loads((OUT/f'{dataset}_{model}.json').read_text());assert len(rows)==100
  assert [(r['id'],r['image_sha256']) for r in rows]==[(r['id'],r['image_sha256']) for r in manifest]
  failures=[r['id'] for r in rows if r['status']!='ok'];assert not failures,failures
  errs=[];by_joint=collections.defaultdict(list);immeans={};outside=0;pub=[]
  for r in rows:
   target=reference[r['id']];assert [p['name'] for p in r['points']]==[p['name'] for p in target['points']]
   eim=[];pr=[]
   for p,q in zip(r['points'],target['points']):
    assert p['reference_in_image']==q['reference_in_image'];valid=q['reference_in_image']
    assert all(math.isfinite(v) for v in p['prediction'])
    x,y=p['prediction'];assert p['prediction_in_image']==(0<=x<target['width'] and 0<=y<target['height'])
    if valid:
     e=math.dist(p['prediction'],q['reference']);assert abs(e-p['error_px'])<1e-6;errs.append(e);eim.append(e);by_joint[p['name']].append(e);outside+=int(not p['prediction_in_image'])
    else:assert p['error_px'] is None
    pr.append({k:p[k] for k in ['name','reference_in_image','error_px','prediction_in_image']})
   immeans[r['id']]=sum(eim)/len(eim) if eim else None
   pub.append({'id':r['id'],'status':r['status'],'points':pr})
  def metrics(e):return {'points':len(e),'mean_px':sum(e)/len(e) if e else None,'correct_20px':sum(v<=20 for v in e),'pck20_percent':100*sum(v<=20 for v in e)/len(e) if e else None}
  summary[dataset][model]={'images':len(rows),'failed_images':failures,**metrics(errs),'prediction_outside_image_on_scored_points':outside,'per_joint':{j:metrics(e) for j,e in by_joint.items()},'per_image_mean_px':immeans}
  details[dataset][model]=pub
 for model in MODELS[1:]:
  a=summary[dataset]['original']['per_image_mean_px'];b=summary[dataset][model]['per_image_mean_px'];summary[dataset][model]['images_mean_error_below_original']=sum(b[k]<a[k] for k in a if a[k] is not None and b[k] is not None)
 checks[dataset]={'images':100,'unique_image_hashes':100,'four_models_identical_samples_and_masks':True,'errors_recomputed_from_coordinates':True,'failed_images':0,'manifest_sha256':hashlib.sha256((OUT/(dataset+'_manifest.json')).read_bytes()).hexdigest()}
(RUN/'evidence/summary.json').write_text(json.dumps(summary,indent=2)+'\n')
for dataset in DATASETS:(RUN/'evidence'/f'{dataset}_point_errors.json').write_text(json.dumps(details[dataset],indent=2)+'\n')
(RUN/'evidence/validation.json').write_text(json.dumps({'command':'python runs/2026-11-18_cross_dataset100/finalize.py','exit_status':0,'checks':checks},indent=2)+'\n')
lines=['| 数据集 | 模型 | 平均误差 px | 20 px 内点数 | PCK@20 |','|---|---|---:|---:|---:|']
for dataset in DATASETS:
 for model in MODELS:
  v=summary[dataset][model];lines.append(f"| {dataset} | {model} | {v['mean_px']:.2f} | {v['correct_20px']} / {v['points']} | {v['pck20_percent']:.2f}% |")
(RUN/'results_table.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))

# Distribution explains why mean error and the 20-pixel criterion can rank differently.
bins = {}
for model, rows in details['unrealego2'].items():
 errors = [p['error_px'] for row in rows for p in row['points'] if p['reference_in_image']]
 cuts = [20, 50, 100, 200]
 counts = [sum(x <= hi and (i == 0 or x > cuts[i-1]) for x in errors) for i, hi in enumerate(cuts)] + [sum(x > 200 for x in errors)]
 assert sum(counts) == len(errors)
 bins[model] = {'bins': ['<=20', '(20,50]', '(50,100]', '(100,200]', '>200'], 'counts': counts, 'images_mean_error_below_original': summary['unrealego2'][model].get('images_mean_error_below_original')}
(RUN/'evidence/unrealego2_error_bins.json').write_text(json.dumps(bins, indent=2)+'\n')
