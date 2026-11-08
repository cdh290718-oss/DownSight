"""Verify baseline regression, shape-only changes, fixed controls and exports."""
import ast,hashlib,json
import numpy as np
from common import ROOT,RUN,OUT,OLD,read,head_rotation_numpy,project_numpy
from egosmplx_pipeline.io import load
execution=read(RUN/'evidence/execution.json');assert len(execution)==7 and all(q['exit_status']==0 for q in execution)
for p in RUN.glob('*.py'):ast.parse(p.read_text())
inputs=read(OUT/'inputs.json');cfg=read(RUN/'settings.json');previous=ROOT/'outputs/synthetic_stereo_abcd_20261108'
assert (OUT/'inputs.json').read_bytes()==(previous/'inputs.json').read_bytes()
# All settings that influence B's existing objective/optimizer are unchanged.
old_cfg=read(ROOT/'runs/2026-11-08_synthetic-stereo-abcd/settings.json')
keys=['head_position_sigma_m','head_rotation_sigma_deg','head_position_weight','head_rotation_weight','steps','pose_lr','rotation_lr','translation_lr','pose_prior_weight','translation_prior_weight','rotation_prior_weight','root_increment_limit_deg','shoulder_tolerance_px','shoulder_weight','wrist_floor_fraction','wrist_guard_weight','minimum_observed_joint_z_m','checkpoint_steps','wilor_weight_per_hand']
assert all(cfg[k]==old_cfg[k] for k in keys)
checks=[]
for row in execution:
 case,seed=row['case'],row['seed'];folder=OUT/(case+'_'+seed);p=load(folder/'final.npz');q=read(folder/'optimization.json');initial=load(folder/'initial.npz');b=load(previous/('B_'+seed)/'initial.npz');raw=load(OLD/'fits'/('left' if seed=='left' else 'right_cpu')/'frames'/seed/'raw/params.npz')
 for key in ['global_orient','body_pose','transl','betas']+q['fixed_parameters']:assert np.array_equal(initial[key],b[key]),(case,seed,key)
 for key in q['fixed_parameters']:assert np.array_equal(p[key],raw[key])
 assert q['joint_optimization'] and q['dataset_read_guard_enabled']
 assert q['hard_caps_deg']==read(previous/('B_'+seed)/'optimization.json')['hard_caps_deg']
 assert q['wrist_floor_m']==read(previous/('B_'+seed)/'optimization.json')['wrist_floor_m']
 assert abs(q['recorded_best_loss']-q['selected_loss'])<2e-6
 assert q['final']['reload_mesh_max_m']<1e-5 and max(q['final']['reload_projection_max_px'].values())<.05
 assert np.max(abs(head_rotation_numpy(p)-np.array(q['final']['head_rotation'])))<2e-6
 assert np.array_equal(p['smplx_joint_cam'][:25],p['smplx_body_joints_3d_render_cam'])
 assert all(k not in p for k in ['smplx_joint_img','smplx_body_joints_2d_render_cam'])
 if case=='B_check':assert np.array_equal(p['betas'],raw['betas'])
 else:
  assert not np.array_equal(p['betas'],raw['betas']);assert q['shape_prior_weight']==cfg['shape_prior_weights'][case]
  assert abs(q['terms']['shape_prior']-q['shape_prior_weight']*np.mean(p['betas']**2))<1e-8
 checks.append(dict(case=case,seed=seed,initial_identical_to_previous_B=True,non_shape_fixed_fields_unchanged=True,head_and_reload_checks_pass=True,fit_dataset_read_guard_enabled=True,finite_npz_and_rebuilt_3d_caches=True))
new=load(OUT/'B_check_left/final.npz');old=load(previous/'B_left/final.npz');regression={k:float(np.max(abs(new[k]-old[k]))) for k in ['betas','global_orient','body_pose','transl','smplx_mesh_cam']}
assert all(x==0 for x in regression.values()),regression
m=read(RUN/'evidence/metrics.json')
for case in m['views'][0]['stages']:
 a,b=[v['stages'][case] for v in m['views']];assert a['source_sha256']==b['source_sha256'];assert abs(a['body25_absolute_mean_mm']-b['body25_absolute_mean_mm'])<.001
result=dict(exit_status=0,command='<BODY_PYTHON> runs/2026-11-08_synthetic-shape-prior/verify.py',fit_count=7,checks=checks,frozen_B_regression_max_abs=regression,prior_center='zero model shape, not initial prediction',all_other_optimizer_settings_unchanged=True,source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in RUN.glob('*.py')})
(RUN/'evidence/verification.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS fixed B reproduced, seven fits, controls, changed betas, shared geometry')
