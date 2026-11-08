"""Archive selected existing evidence; never run or modify fitting."""
import hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];RUN=Path(__file__).resolve().parent
SRC=ROOT/'runs/2026-11-08_synthetic-head-robustness'
EXP=ROOT/'experiments/synthetic_oracle_head_shape_20261108'
PIN='c4e2b9de76573139fa66cbb953bf973d2af88c77'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
EXP.mkdir(exist_ok=False);(EXP/'evidence').mkdir()
(EXP/'settings.json').write_bytes((SRC/'settings.json').read_bytes())
assessment=read(SRC/'evidence/assessment.json');metrics=read(SRC/'evidence/metrics.json');shapes=read(SRC/'evidence/shape_metrics.json')
opt=next(q for q in read(SRC/'evidence/optimization_summary.json') if q['case']=='oracle' and q['seed']=='left')
selected=dict(case='oracle_left',source_commit=PIN,source_run=str(SRC.relative_to(ROOT)),primary=next(q for q in assessment['primary_rows'] if q['case']=='oracle_left'),views={v['side']:v['stages']['oracle_left'] for v in metrics['views']},shape=next(q for q in shapes['cases'] if q['case']=='oracle_left'),truth_betas_squared_mean=shapes['truth_betas_squared_mean'],initialization=opt['initialization'],final_artifact=opt['final'],terms=opt['terms'],pose_increments_deg=opt['pose_increments_deg'],hard_caps_deg=opt['hard_caps_deg'],comparison_rows=assessment['primary_rows'],perturbation_summary=assessment['perturbation_summary'],interpretation='Selected predeclared left seed, not selected by GT error. Existing results, not rerun.')
(EXP/'evidence/results.json').write_text(json.dumps(selected,indent=2,allow_nan=False)+'\n')
files=['fit.py','common.py','initialization.py','prepare.py','run.py','finish.py','evaluate.py','evaluate_shape.py','verify.py','test_method.py','settings.json','cases.json','brief.md','report.md','method.md','projection.md','review.md']+['evidence/'+n for n in ['assessment.json','metrics.json','shape_metrics.json','optimization_summary.json','method_tests.json','verification.json','execution.json','executed_source_check.json','evaluation_execution.json','triangulated.json','perturbations.json','preparation.json','fit_log_summary.json','development_failures.json','visual_inspection.json','prepush.json']]+['assets/'+side+'_oracle_left.jpg' for side in ['left','right']]
resources=[]
for name in files:
 p=SRC/name;rel=str(p.relative_to(ROOT));assert subprocess.check_output(['git','show',PIN+':'+rel],cwd=ROOT)==p.read_bytes()
 resources.append(dict(path=rel,sha256=sha(p),bytes=p.stat().st_size,pinned_url='https://github.com/cdh290718-oss/DownSight/blob/'+PIN+'/'+rel))
other=['runs/2026-11-08_synthetic-original/report.md','runs/2026-11-08_synthetic-original/evidence/resources.json','runs/2026-11-08_synthetic-original/evidence/inputs.json','runs/2026-11-08_synthetic-stereo-abcd/report.md','runs/2026-11-08_synthetic-stereo-abcd/evidence/triangulation.json','runs/2026-11-08_synthetic-shape-prior/report.md']
for name in other:
 p=ROOT/name;assert subprocess.check_output(['git','show',PIN+':'+name],cwd=ROOT)==p.read_bytes();resources.append(dict(path=name,sha256=sha(p),bytes=p.stat().st_size,pinned_url='https://github.com/cdh290718-oss/DownSight/blob/'+PIN+'/'+name))
manifest=dict(experiment='synthetic_oracle_head_shape_20261108',status='Oracle diagnostic; candidate body-fitting baseline; not default pipeline or deployment validation',source_commit=PIN,fit_base_commit='9012bbf856570e85da60261190ead1092cce9997',primary_case={'case':'oracle','seed':'left','method':'improved','anchor':'oracle'},dataset={'repository':'https://github.com/cdh290718-oss/DownSightSmplxdata','commit':'192671271a74b99f41b9eb033d8f88b1dac77ada','sample':'000001','views':['cam3/left','cam4/right'],'independent_postures':1},implementation_policy='Authoritative implementation remains in the source run, pinned here by commit and SHA-256. This experiment folder is the curated entry point; it does not duplicate or silently modify the fitter.',repository_resources=resources,external_inputs=read(SRC/'evidence/preparation.json')['assets'],external_outputs=[dict(resource='outputs/synthetic_head_robustness_20261108/oracle_left/'+name+'.npz',sha256=opt[name]['sha256'],access='Development workspace or owner-provided external storage; not included in GitHub',coordinate_frame='cam3 CV, metres') for name in ['initial','final']],settings_sha256=sha(EXP/'settings.json'),archive_reruns_fitting=False)
(EXP/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
q=dict(exit_status=0,command='python runs/2026-11-08_archive-oracle-shape-baseline/package.py',base_commit=PIN,experiment=str(EXP.relative_to(ROOT)),source_files_verified_against_commit=len(resources),settings_identical=True,selected_case='oracle_left',source_final_sha256=opt['final']['sha256'],fitting_rerun=False)
(RUN/'evidence/packaging.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(q,indent=2))
