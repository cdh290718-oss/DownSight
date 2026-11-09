"""Check the archive's links, immutable references, exact numbers and D identity."""
import hashlib,json,re,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];RUN=Path(__file__).resolve().parent;EXP=ROOT/'experiments/synthetic_template_head_stereo_20261109';SRC=ROOT/'runs/2026-11-09_initialization-ablation'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();m=read(EXP/'manifest.json');r=read(EXP/'evidence/results.json');source=read(SRC/'evidence/assessment.json');checks=[]
def check(name,condition):
 checks.append(dict(name=name,passed=bool(condition)))
check('settings byte exact', (EXP/'settings.json').read_bytes()==(SRC/'settings.json').read_bytes())
check('case D exact',read(EXP/'case.json')==next(c for c in read(SRC/'cases.json') if c['case']=='D'))
check('archived summary numbers exact',all(v==source['summary'][k] for k,v in r['summary'].items()))
check('A and original distinguished',r['roles']['A']!=r['roles']['original'])
check('primary budget and selected step',r['primary_budget_steps']==20000 and r['selected_training_step']==19960)
check('D stability false preserved',r['D_convergence']==source['convergence']['D'] and r['D_convergence']['diagnostic_pass'] is False)
check('D no prediction seeds read',r['D_verification']['passed'] and not any('/seeds/' in x or '/raw/' in x for x in r['D_verification']['read_files']))
check('D initial beta zero',not any(r['D_initial_beta']))
for view in read(SRC/'evidence/metrics.json')['views']:
 check('primary camera metrics exact '+view['camera'],r['primary_by_camera'][view['camera']]=={k:v for k,v in view['stages']['D_20000'].items() if k!='per_joint'})
for x in m['repository_resources']:
 p=ROOT/x['path'];raw=subprocess.check_output(['git','show',m['source_commit']+':'+x['path']],cwd=ROOT);check('pinned source '+x['path'],p.read_bytes()==raw and sha(p)==x['sha256'] and len(raw)==x['bytes'])
for x in m['external_inputs']+m['external_outputs']:
 p=ROOT/x['resource'];check('external checksum '+x['resource'],sha(p)==x['sha256'] and p.stat().st_size==x['bytes'])
links=[]
for p in list(EXP.glob('*.md'))+list(RUN.glob('*.md')):
 for target in re.findall(r'!?\[[^]]*\]\(([^)]+)\)',p.read_text()):
  if target.startswith(('http://','https://','#')):continue
  local=(p.parent/target.split('#')[0]).resolve();links.append(dict(document=str(p.relative_to(ROOT)),target=target,exists=local.exists()));check('local link '+str(p.relative_to(ROOT))+' '+target,local.exists())
readme=(EXP/'README.md').read_text()
for title,key in [('身体25点平均','body25_absolute_mean_mm'),('骨盆','pelvis_absolute_error_mm'),('双膝','knees_mean_mm'),('双踝','ankles_mean_mm'),('手指40点平均','hands40_absolute_mean_mm'),('中性体型网格','neutral_shape_mesh_mm'),('全网格绝对误差','posed_mesh_absolute_mean_mm'),('22骨骼相机朝向平均误差（°）','native22_camera_rotation_mean_deg')]:
 expected='| '+title+' | %.2f | %.2f |'%(r['summary']['A_20000'][key],r['summary']['D_20000'][key]);check('README numeric row '+title,expected in readme)
check('no fitting rerun declared',m['archive_reruns_fitting'] is False)
passed=all(c['passed'] for c in checks);result=dict(exit_status=0 if passed else 1,command='python '+str((RUN/'check_archive.py').relative_to(ROOT)),source_sha256=sha(RUN/'check_archive.py'),scope='Archive integrity, links and exact recorded metrics; no new fitting or independent experiment reproduction.',passed=passed,checks=checks,local_links=links)
(RUN/'evidence/archive_check.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS' if passed else 'FAIL',len(checks),'checks',len(links),'links')
if not passed:raise SystemExit(1)
