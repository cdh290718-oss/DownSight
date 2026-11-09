"""Post-fit invariants; no hyperparameter or checkpoint selection using GT."""
import hashlib,json
from body_targets import replace_targets
import numpy as np
from common import ROOT,RUN,OUT,OLD,read
cases=read(RUN/'cases.json');cfg=read(RUN/'settings.json');points=read(OUT/'triangulated.json')['points'];records=[];summaries=[];seeds={}
for c in cases:
 label=c['case']+'_'+c['seed'];folder=OUT/label;effective_points,intervention=replace_targets(points,read(OUT/'target_overrides'/(c['target_override']+'.json')) if c['target_override'] else None);q=read(folder/'optimization.json');p=np.load(folder/'final.npz');initial=np.load(folder/'initial.npz');fail=[]
 def check(name,condition):
  if not bool(condition):fail.append(name)
 check('only correct target override opened',[x for x in q['read_files'] if '/target_overrides/' in x]==([str((OUT/'target_overrides'/(c['target_override']+'.json')).relative_to(ROOT))] if c['target_override'] else []))
 check('only intended target coordinates changed',q['intervention']==intervention)
 previous=read(ROOT/'outputs/synthetic_head_robustness_20261108/oracle_left/optimization.json')
 check('initialization and subspace exact to original',q['initialization']==previous['initialization'])
 for stage in ['initial']+(['final'] if c['target_override'] is None else []):
  original=np.load(ROOT/'outputs/synthetic_head_robustness_20261108/oracle_left'/(stage+'.npz'));current=np.load(folder/(stage+'.npz'))
  check('parameters exact to original '+stage,all(np.array_equal(current[k],original[k]) for k in ['betas','body_pose','global_orient','transl','left_hand_pose','right_hand_pose','jaw_pose','leye_pose','reye_pose','expression','smplx_joint_cam','smplx_mesh_cam']))
 if c['target_override']=='hips':
  original=np.load(ROOT/'outputs/synthetic_hip_target_ablation_20261109/gt_hips_left/final.npz')
  check('GT hips exact replay',all(np.array_equal(p[k],original[k]) for k in ['betas','body_pose','global_orient','transl','smplx_joint_cam','smplx_mesh_cam']))
 check('settings exact to original',q['settings']==previous['settings'])
 check('no GT dataset reads',all('/dataset/' not in x and 'oracle_shape' not in x for x in q['read_files']))
 anchor_reads=[x for x in q['read_files'] if '/anchors/' in x]
 check('only allowed anchor read',anchor_reads==(['outputs/synthetic_body_target_oracle_20261109/anchors/'+c['anchor']+'.json'] if c['anchor'] else []))
 check('truth stripped observations',set(read(OUT/'inputs.json'))=={'views'})
 check('no head terms when none',('head_position' in q['terms'])==bool(c['anchor']) and ('head_orientation' in q['terms'])==bool(c['anchor']))
 if not c['anchor']:check('no head diagnostics in none fit',q['initial']['head_position_error_mm'] is None and q['final']['head_orientation_error_deg'] is None)
 check('no wrist floor',q['wrist_floor_m'] is None and not any('wrist_guard' in k for k in q['terms']))
 check('all iterations',q['trace'][-1]['step']==cfg['steps'])
 check('checkpoint objective',abs(q['selected_loss']-q['recorded_best_loss'])<1e-7 and q['selected_loss']<=min(x['loss'] for x in q['trace'] if x['feasible'])+1e-7)
 check('pose hard caps',all(q['pose_increments_deg'][n]<=cap+1e-4 for n,cap in q['hard_caps_deg'].items()))
 check('shape prior formula',abs(q['terms']['shape_prior']-cfg['shape_prior_weight']*np.mean(p['betas']**2))<1e-7)
 if c['seed'] not in seeds:seeds[c['seed']]={k:initial[k].copy() for k in ['betas','global_orient','transl','body_pose']}
 check('same initialization within seed',all(np.allclose(initial[k],v,atol=2e-6,rtol=0) for k,v in seeds[c['seed']].items()))
 raw=np.load(OLD/'fits'/('left' if c['seed']=='left' else 'right_cpu')/'frames'/c['seed']/'raw/params.npz')
 check('fixed hand and face',all(np.array_equal(p[k],raw[k]) for k in q['fixed_parameters']))
 subspace_error=None;metric_errors={}
 if c['method']=='improved':
  basis=np.array(q['initialization']['basis']);beta=p['betas'].reshape(-1);subspace_error=float(np.linalg.norm(beta-basis@(basis.T@beta)));check('beta stays in observed subspace',subspace_error<1e-5)
  check('2D not double counted',all(not any(k.endswith('/'+g) for g in ['body','wrist','shoulder','hand_left','hand_right']) or k.startswith('metric/') for k in q['terms']))
  for name in ['body','wrist','shoulder','hand_left','hand_right']:
   rows=[x for x in effective_points if x['accepted'] and x['group']==name]
   if not rows:continue
   residual=[]
   for x in rows:residual.append(np.linalg.norm(np.linalg.solve(np.linalg.cholesky(x['covariance']),p['smplx_joint_cam'][x['index']]-np.array(x['xyz'])))/np.sqrt(3))
   d=np.array(residual);delta=cfg['metric_huber_delta'];loss=np.where(d<=delta,.5*d*d,delta*(d-.5*delta)).mean();mult=cfg['shoulder_weight'] if name=='shoulder' else cfg['wilor_weight_per_hand'] if name.startswith('hand_') else 1.;expected=cfg['metric_loss_weight']*mult*loss;metric_errors[name]=abs(expected-q['terms']['metric/'+name]);check('independent metric reconstruction '+name,metric_errors[name]<1e-6)
 rows=read(RUN/'evidence/metrics.json')['views'];b0=rows[0]['stages'][label]['body25_absolute_mean_mm'];b1=rows[1]['stages'][label]['body25_absolute_mean_mm'];check('shared body invariance',abs(b0-b1)<.001)
 record=dict(case=label,passed=not fail,failures=fail,source_sha256=q['final']['sha256'],initialization_sha256=q['initial']['sha256'],allowed_anchor_reads=anchor_reads,subspace_residual=subspace_error,metric_loss_reconstruction_error=metric_errors,read_files=q['read_files']);records.append(record)
 summaries.append({k:v for k,v in q.items() if k not in ['settings','read_files']})
(RUN/'evidence/optimization_summary.json').write_text(json.dumps(summaries,indent=2,allow_nan=False)+'\n')
result=dict(exit_status=0 if all(x['passed'] for x in records) else 1,checks=records,scope='Full GT dataset denied during fitting. Both cases read original Head anchor; each intervention reads only its permitted extracted hip/body targets. Initialization and subspace must exactly match original. Control final parameters must exactly replay original.');(RUN/'evidence/verification.json').write_text(json.dumps(result,indent=2)+'\n');print('verified',len(records),'passed',sum(r['passed'] for r in records))
if result['exit_status']:raise SystemExit(1)
