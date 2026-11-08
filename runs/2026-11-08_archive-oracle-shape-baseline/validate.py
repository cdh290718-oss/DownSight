"""Validate archive links, snapshots and reported numbers; no fitting or external asset reads."""
import hashlib,json,re,subprocess
from pathlib import Path
from urllib.parse import unquote
ROOT=Path(__file__).resolve().parents[2];RUN=Path(__file__).resolve().parent
EXP=ROOT/'experiments/synthetic_oracle_head_shape_20261108';SRC=ROOT/'runs/2026-11-08_synthetic-head-robustness'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=read(EXP/'manifest.json');result=read(EXP/'evidence/results.json');checks=[]
def check(name,value):
 checks.append(dict(check=name,passed=bool(value)))
 if not value:raise AssertionError(name)
for entry in manifest['repository_resources']:
 p=ROOT/entry['path'];check('source hash '+entry['path'],sha(p)==entry['sha256']);check('pinned source '+entry['path'],p.read_bytes()==subprocess.check_output(['git','show',manifest['source_commit']+':'+entry['path']],cwd=ROOT))
check('settings snapshot identical', (EXP/'settings.json').read_bytes()==(SRC/'settings.json').read_bytes())
assessment=read(SRC/'evidence/assessment.json');metrics=read(SRC/'evidence/metrics.json');shapes=read(SRC/'evidence/shape_metrics.json');opts=read(SRC/'evidence/optimization_summary.json')
primary=next(q for q in assessment['primary_rows'] if q['case']=='oracle_left');opt=next(q for q in opts if q['case']=='oracle' and q['seed']=='left')
check('primary row exact',result['primary']==primary);check('all comparisons retained',result['comparison_rows']==assessment['primary_rows']);check('perturbation summary exact',result['perturbation_summary']==assessment['perturbation_summary'])
check('both views exact',result['views']=={v['side']:v['stages']['oracle_left'] for v in metrics['views']})
check('shape row exact',result['shape']==next(q for q in shapes['cases'] if q['case']=='oracle_left'))
check('truth mean shape fact',result['truth_betas_squared_mean']==shapes['truth_betas_squared_mean']==0)
check('selected output hash',manifest['external_outputs'][1]['sha256']==opt['final']['sha256']==result['final_artifact']['sha256'])
check('rank exactly one',result['initialization']['retained_rank']==1)
check('head anchor and seed fixed',manifest['primary_case'] in read(SRC/'cases.json'))
readme=(EXP/'README.md').read_text();boundaries=(EXP/'BOUNDARIES.md').read_text();method=(EXP/'METHOD.md').read_text()
for key in ['body25_absolute_mean_mm','pelvis_absolute_error_mm','wrists_absolute_mean_mm','elbows_absolute_mean_mm','hands40_absolute_mean_mm','neutral_shape_mean_mm']:
 check('README value '+key,('%.2f'%primary[key])+' mm' in readme)
check('pelvis XYZ statement',', '.join('%+.2f'%x for x in primary['pelvis_error_xyz_mm']).replace('-','−') in boundaries)
check('saturated joints statement',', '.join(primary['saturated_pose_increments']) in boundaries)
for eigenvalue in result['initialization']['shape_information_eigenvalues'][-3:]:check('information spectrum '+str(eigenvalue),('%.6f'%eigenvalue) in method)
links=[]
for p in sorted(EXP.glob('*.md')):
 for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',p.read_text()):
  if target.startswith(('https://','http://','#','mailto:')):continue
  clean=unquote(target.split('#')[0]);q=(p.parent/clean).resolve();check('local link '+str(p.relative_to(ROOT))+' -> '+clean,q.exists() and (q==ROOT or ROOT in q.parents));links.append(dict(document=str(p.relative_to(ROOT)),target=target))
for p in [EXP/'README.md',EXP/'METHOD.md',EXP/'RESULTS.md',EXP/'BOUNDARIES.md',EXP/'REPRODUCE.md']:
 check('newline format '+p.name,p.read_bytes().endswith(b'\n') and not p.read_bytes().endswith(b'\n\n'))
receipt=dict(exit_status=0,command='python runs/2026-11-08_archive-oracle-shape-baseline/validate.py',scope='Archive-only verification, no rerun of fitting and no independent reproduction claim',checks=checks,local_links=links,archive_file_hashes={str(p.relative_to(ROOT)):sha(p) for p in sorted(EXP.rglob('*')) if p.is_file()})
(RUN/'evidence/validation.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n');print('PASS',len(checks),'checks;',len(links),'local links')
