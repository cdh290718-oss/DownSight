"""Independent reconstruction of identical objectives, factor isolation, and read guards."""
import hashlib,json
import numpy as np
from common import ROOT,RUN,OUT,read,head_rotation_numpy
from stages import stages
from egosmplx_pipeline.body_common import BODY_POSE_NAMES
cfg=read(RUN/'settings.json');points=[p for p in read(OUT/'triangulated.json')['points'] if p['accepted']];anchor=read(OUT/'anchors/oracle.json');cases=read(RUN/'cases.json');records=[];summaries=[];basis0=None;initials={}
for c in cases:
 case=c['case'];folder=OUT/(case+'_left');q=read(folder/'optimization.json');initial=np.load(folder/'initial.npz');fail=[]
 def check(n,v):
  if not bool(v):fail.append(n)
 allowed=sorted(str((OUT/'seeds'/(n+'.json')).relative_to(ROOT)) for n in ['shape','pose'] if c[n+'_seed']=='network')
 check('reads only permitted network factors',sorted(x for x in q['read_files'] if '/seeds/' in x)==allowed)
 check('no other outputs or raw or GT dataset read',all(x.startswith(str(OUT.relative_to(ROOT))+'/') or x.startswith(str(RUN.relative_to(ROOT))+'/') for x in q['read_files']))
 check('observations identical',all(hashlib.sha256((OUT/p['destination'].split(str(OUT.relative_to(ROOT))+'/')[1]).read_bytes()).hexdigest()==p['sha256'] for p in read(RUN/'evidence/preparation.json')['copied_assets']))
 check('same settings',q['settings_sha256']==hashlib.sha256((RUN/'settings.json').read_bytes()).hexdigest())
 basis=np.array(q['initialization']['basis'])
 if basis0 is None:basis0=basis
 check('identical shape subspace',np.array_equal(basis,basis0))
 check('head aligned at start',q['initial']['head_position_error_mm']<.01 and q['initial']['head_orientation_error_deg']<.01)
 check('exact budget',q['trace'][-1]['step']==cfg['steps'] and set(q['budgets'])==set(map(str,cfg['budget_steps'])))
 expected_shape=np.zeros(10) if c['shape_seed']=='mean' else np.array(read(OUT/'seeds/shape.json')['values'],np.float32)
 expected_pose=np.zeros((21,3)) if c['pose_seed']=='template' else np.array(read(OUT/'seeds/pose.json')['values'],np.float32)
 check('correct pose factor',np.array_equal(initial['body_pose'].reshape(21,3),expected_pose))
 check('correct projected shape factor',np.allclose(initial['betas'].reshape(-1),basis@(basis.T@expected_shape),atol=1e-6,rtol=0))
 previous=read(ROOT/'outputs/synthetic_head_robustness_20261108/oracle_left/optimization.json')
 check('shape basis unchanged from original',np.array_equal(basis,np.array(previous['initialization']['basis'])))
 initials[case]={k:initial[k].copy() for k in ['body_pose','betas']}
 if c['pose_seed']=='template':check('template body zero',np.count_nonzero(initial['body_pose'])==0)
 if c['shape_seed']=='mean':check('mean beta zero',np.count_nonzero(initial['betas'])==0)
 errors={}
 for budget in cfg['budget_steps']:
  p=np.load(folder/('budget_'+str(budget)+'.npz'));snap=q['budgets'][str(budget)];terms={}
  for name in ['body','wrist','shoulder','hand_left','hand_right']:
   rows=[x for x in points if x['group']==name];d=np.array([np.linalg.norm(np.linalg.solve(np.linalg.cholesky(x['covariance']),p['smplx_joint_cam'][x['index']]-np.array(x['xyz'])))/np.sqrt(3) for x in rows]);h=cfg['metric_huber_delta'];mult=cfg['shoulder_weight'] if name=='shoulder' else cfg['wilor_weight_per_hand'] if name.startswith('hand_') else 1.;terms['metric/'+name]=cfg['metric_loss_weight']*mult*np.where(d<=h,.5*d*d,h*(d-.5*h)).mean()
  names=list(cfg['absolute_pose_caps_deg']);ids=[BODY_POSE_NAMES.index(n) for n in names];b=p['body_pose'].reshape(21,3);scales=np.deg2rad([cfg['absolute_pose_caps_deg'][n] for n in names]);terms['pose_prior']=cfg['pose_prior_weight']*np.mean((b[ids]/scales[:,None])**2);terms['shape_prior']=cfg['shape_prior_weight']*np.mean(p['betas']**2)
  R=head_rotation_numpy(p);terms['head_orientation']=cfg['head_rotation_weight']*np.sum((R-np.array(anchor['rotation']))**2)/(2*np.deg2rad(cfg['head_rotation_sigma_deg'])**2)
  metric_error={k:float(abs(v-snap['terms'][k])) for k,v in terms.items()};errors[str(budget)]=metric_error;check('independent objective terms '+str(budget),max(metric_error.values())<1e-6)
  check('sum of terms equals loss',abs(sum(snap['terms'].values())-snap['selected_loss'])<1e-6)
  check('no network centered priors',not any(k in snap['terms'] for k in ['translation_prior','rotation_prior']))
  check('fixed zero hand face feet',all(np.count_nonzero(p[k])==0 for k in q['fixed_parameters']) and np.count_nonzero(b[[9,10]])==0)
  check('absolute caps',np.all(np.linalg.norm(b[ids],axis=1)<=scales+1e-5));beta=p['betas'].reshape(-1);check('shape in subspace',np.linalg.norm(beta-basis@(basis.T@beta))<1e-5)
  check('best feasible checkpoint',snap['feasible'] and snap['step']<=budget and snap['selected_loss']<=min(t['loss'] for t in q['trace'] if t['feasible'] and t['step']<=budget)+1e-7)
  views=read(RUN/'evidence/metrics.json')['views'];label=case+'_'+str(budget);headloss=cfg['head_position_weight']*(views[0]['stages'][label]['head_position_error_mm']/1000/cfg['head_position_sigma_m'])**2;check('head position term',abs(headloss-snap['terms']['head_position'])<1e-6);check('two camera rigid invariant',abs(views[0]['stages'][label]['body25_absolute_mean_mm']-views[1]['stages'][label]['body25_absolute_mean_mm'])<.001)
 check('all fit source hashes unchanged',all(hashlib.sha256((RUN/n).read_bytes()).hexdigest()==h for rec in read(RUN/'evidence/execution.json') if rec['case']==case for n,h in rec['source_sha256'].items()))
 records.append(dict(case=case,passed=not fail,failures=fail,read_files=q['read_files'],independent_loss_errors=errors));summary={k:v for k,v in q.items() if k!='read_files'}
 # Keep scalar trajectory and meaningful geometry endpoints; full trace stays local.
 summary['trace']=[{k:v for k,v in row.items() if k!='target_joints_m'} for row in q['trace']];summary['convergence_endpoint_joints']={str(row['step']):row['target_joints_m'] for row in [q['trace'][-11],q['trace'][-1]]};summaries.append(summary)
factors=dict(network_pose_equal_AB=np.array_equal(initials['A']['body_pose'],initials['B']['body_pose']),template_pose_equal_CD=np.array_equal(initials['C']['body_pose'],initials['D']['body_pose']),network_shape_equal_AC=np.array_equal(initials['A']['betas'],initials['C']['betas']),mean_shape_equal_BD=np.array_equal(initials['B']['betas'],initials['D']['betas']))
passed=all(x['passed'] for x in records) and all(factors.values());(RUN/'evidence/optimization_summary.json').write_text(json.dumps(summaries,indent=2,allow_nan=False)+'\n');(RUN/'evidence/verification.json').write_text(json.dumps(dict(exit_status=0 if passed else 1,checks=records,factor_checks=factors),indent=2)+'\n');print('verification',passed)
if not passed:raise SystemExit(1)
