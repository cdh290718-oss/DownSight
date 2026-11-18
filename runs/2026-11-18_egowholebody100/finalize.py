import json,math,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent;ROOT=R.parents[1];O=ROOT/'outputs/egowholebody100_20261118';summary=json.load(open(R/'evidence/summary.json'))['summary'];manifest=json.load(open(O/'manifest.json'));records={};base=None;table='| 模型 | 2D 平均误差 px | 20 px 内点数 | PCK@20 |\n|---|---:|---:|---:|\n'
for m,v in summary.items():
 rows=json.load(open(O/(m+'.json')));assert len(rows)==100 and len({x['id'] for x in rows})==100
 keys=[(x['id'],x['image_sha256']) for x in rows];assert keys==[(x['id'],x['image_sha256']) for x in manifest]
 valid=[(x['id'],p['name'],p['reference_in_image']) for x in rows for p in x['points']]
 if base is None:base=valid
 assert valid==base
 errors=[p['error_px'] for x in rows for p in x['points'] if p['reference_in_image']];assert all(math.isfinite(e) for e in errors);assert len(errors)==v['reference_in_image_points'];assert sum(e<=20 for e in errors)==v['correct_20px'];assert abs(sum(errors)/len(errors)-v['mean_px'])<1e-9
 records[m]=[{'id':x['id'],'points':[{'name':p['name'],'error_px':p['error_px'],'prediction_in_image':p['prediction_in_image']} for p in x['points']]} for x in rows]
 table+=f"| {m} | {v['mean_px']:.2f} | {v['correct_20px']} / {len(errors)} | {v['correct_20px']/len(errors)*100:.2f}% |\n"
(R/'evidence/point_errors.json').write_text(json.dumps(records,indent=2)+'\n')
baseweights=json.load(open(ROOT/'runs/2026-11-18_gen_three_models/evidence/weights.json'))['sha256'];weights=json.load(open(R/'evidence/weights.json'));assert all(weights[k]['sha256']==h for k,h in baseweights.items())
(R/'evidence/execution.json').write_text(json.dumps({'prepare_command':'EWB_DATA_ROOT=<DATA_DISK>/EgoWholeBody/render_people_adanna <PYTHON_ENV>/bin/python runs/2026-11-18_egowholebody100/prepare.py','inference_command':'PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=2 <PYTHON_ENV>/bin/python -u runs/2026-11-18_egowholebody100/evaluate.py','prepare_exit_status':0,'inference_exit_status':0,'validation_exit_status':0,'same_image_hashes_and_masks_all_models':True,'no_nonfinite_scored_predictions':True,'prior_finetuned_checkpoint_hashes_match':True,'overlay_review':['ewb_000','ewb_001','ewb_002','ewb_003'],'input_resolution':[1280,1024],'manifest_sha256':hashlib.sha256((O/'manifest.json').read_bytes()).hexdigest(),'reference_points_total':1200,'reference_points_in_image':1194,'reference_points_out_of_image':6},indent=2)+'\n')
p=R/'report.md';s=p.read_text().replace('执行中。',table+'\n全部四组各完成 100 张，没有失败或替换图片。1200 个参考关节点中 1194 个在画内，6 个不参与 2D 评分。20000 张模型平均误差最低；5000 张模型 20 px 内点数最多，但仅比 20000 多一个点。不能从如此小的 PCK 差异判定总体优劣。汇总和逐关节统计见 [summary.json](evidence/summary.json)，逐样本逐点距离见 [point_errors.json](evidence/point_errors.json)。\n');p.write_text(s);print(table)
