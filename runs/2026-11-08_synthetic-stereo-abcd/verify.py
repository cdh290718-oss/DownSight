"""Verify saved runs and intended single-factor differences without refitting."""
import ast,hashlib,json
import numpy as np
from common import ROOT,RUN,OUT,OLD,read,head_rotation_numpy,angle_deg
for p in RUN.glob('*.py'):ast.parse(p.read_text())
execution=read(RUN/'evidence/execution.json');assert len(execution)==8 and all(q['exit_status']==0 for q in execution)
inputs=read(OUT/'inputs.json');checks=[]
for row in execution:
    case,seed=row['case'],row['seed'];folder=OUT/(case+'_'+seed);q=read(folder/'optimization.json');p=np.load(folder/'final.npz');source=OLD/'fits'/('left' if seed=='left' else 'right_cpu')/'frames'/seed/'raw/params.npz';raw=np.load(source)
    for key in q['fixed_parameters']:
        expected=np.array(read(OUT/'oracle_shape.json')['betas'],dtype=p[key].dtype).reshape(p[key].shape) if case=='D' and key=='betas' else raw[key]
        assert np.array_equal(p[key],expected),(case,seed,key)
    with np.load(folder/'parameters.npz') as exported:
        for key in ['global_orient','body_pose','transl']+q['fixed_parameters']:assert np.array_equal(exported[key],p[key])
        assert np.array_equal(exported['smplx_body_joints_3d_render_cam'],exported['smplx_joint_cam'][:25])
        assert np.max(abs(exported['smplx_mesh_cam']-p['smplx_mesh_cam']))<1e-5
    assert q['joint_optimization'];assert q['final']['reload_mesh_max_m']<1e-5 and max(q['final']['reload_projection_max_px'].values())<.05
    assert abs(q['recorded_best_loss']-q['selected_loss'])<2e-6
    err=float(np.max(abs(head_rotation_numpy(p)-np.array(q['final']['head_rotation']))));assert err<2e-6
    assert q['input_sha256']==hashlib.sha256((OUT/'inputs.json').read_bytes()).hexdigest()
    for name,value in q['pose_increments_deg'].items():assert value<=q['hard_caps_deg'][name]+1e-4
    checks.append(dict(case=case,seed=seed,fixed_parameters_exact=True,reload_checks_pass=True,independent_head_rotation_max_abs=err))
for seed in ['left','right']:
    b,c,d=[read(OUT/(s+'_'+seed)/'optimization.json') for s in ['B','C','D']]
    assert b['prior_caps_deg']==c['prior_caps_deg']==d['prior_caps_deg'];assert b['hard_caps_deg']==d['hard_caps_deg']
    for key in b['hard_caps_deg']:assert abs(c['hard_caps_deg'][key]-2*b['hard_caps_deg'][key])<1e-4
    initial=[np.load(OUT/(s+'_'+seed)/'initial.npz') for s in ['B','C','D']]
    for key in ['global_orient','body_pose','transl','left_hand_pose','right_hand_pose']:
        assert np.array_equal(initial[0][key],initial[1][key]) and np.array_equal(initial[0][key],initial[2][key])
    assert b['wrist_floor_m']==c['wrist_floor_m']==d['wrist_floor_m']
for side in ['left','right']:
    a=np.load(OUT/('A_'+side+'_left')/'initial.npz');b=np.load(OUT/'B_left/initial.npz')
    for key in ['global_orient','body_pose','transl','betas','left_hand_pose','right_hand_pose']:assert np.array_equal(a[key],b[key])
    ps=inputs['views'][side]['body_points'];assert all(p['name']!='nose' for p in ps);assert {p['name'] for p in ps}>={'left_shoulder','right_shoulder'}
    hands=inputs['views'][side]['hands'];assert len({q['candidate'] for q in hands.values()})==len(hands)
metrics=read(RUN/'evidence/metrics.json');assert len(metrics['views'])==2
for case in ['B','C','D','B_right_seed','C_right_seed','D_right_seed']:
    left,right=[v['stages'][case] for v in metrics['views']];assert left['source_sha256']==right['source_sha256']
    assert abs(left['body25_absolute_mean_mm']-right['body25_absolute_mean_mm'])<.001
result=dict(exit_status=0,command='<BODY_PYTHON> runs/2026-11-08_synthetic-stereo-abcd/verify.py',case_count=8,checks=checks,same_initial_parameters_for_A_B_C_D_except_D_shape=True,C_only_hard_caps_doubled_prior_unchanged=True,D_only_shape_changed=True,stereo_views_use_identical_parameters=True,body_error_rigid_transform_invariance=True,source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in RUN.glob('*.py')})
(RUN/'evidence/verification.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS 8 runs, fixed parameters, A-D differences, shared geometry and Head FK')
