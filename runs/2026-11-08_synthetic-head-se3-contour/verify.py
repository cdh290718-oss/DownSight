"""Verify saved artifacts, independent Head FK, common initialization and gates."""
import ast,hashlib,json
from pathlib import Path
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[2];RUN=Path(__file__).resolve().parent;OUT=ROOT/'outputs/synthetic_head_se3_contour_20261108';OLD=ROOT/'outputs/synthetic_original_20261108'
read=lambda p:json.loads(p.read_text())
for p in RUN.glob('*.py'):ast.parse(p.read_text())
execution=read(RUN/'evidence/execution.json');assert len(execution)==4 and all(q['exit_status']==0 for q in execution)
checks=[]
for side,dirname in [('left','left'),('right','right_cpu')]:
    inputs=read(OUT/(side+'_targets.json'));assert inputs['targets']['torso']['active'];assert not inputs['targets']['legs']['active']
    R=np.array(inputs['head_rotation_matrix']);assert np.allclose(R.T@R,np.eye(3),atol=1e-6) and np.linalg.det(R)>.999
    with np.load(OLD/'dataset/samples/000001/smplx_reconstruction.npz') as gt, np.load(OLD/'dataset/samples/000001/geometry.npz') as scene, np.load(OLD/'dataset/samples/000001'/(side+'_vertices.npz')) as view:
        # Independent numpy FK and world transform, not the differentiable fitter.
        chain_R=cv2.Rodrigues(gt['global_orient'].astype(float))[0]
        for i in [3,6,9,12,15]:chain_R=chain_R@cv2.Rodrigues(gt['body_pose'].reshape(21,3)[i-1].astype(float))[0]
        gtR=view['world_to_camera_cv'][:3,:3]@scene['object_matrix_world'][:3,:3]@chain_R
        assert np.allclose(R,gtR,atol=1e-10)
        assert np.allclose(inputs['head_position_m'],view['joints_camera_cv_m'][15],atol=1e-10)
    source=np.load(OLD/'fits'/dirname/'frames'/side/'raw/params.npz')
    initial=[]
    for case in ['se3_joint','se3_contour_joint']:
        p=np.load(OUT/side/case/'final.npz');q=read(OUT/side/case/'optimization.json')
        assert all(np.array_equal(p[k],source[k]) for k in q['fixed_parameters'])
        assert 'nose' not in q['selected_points'] and all(n in q['selected_points'] for n in ['left_shoulder','right_shoulder'])
        assert q['final']['reload_mesh_max_m']<1e-5 and q['final']['reload_selected_px_max']<.05
        assert abs(q['selected_best_loss']-q['recorded_best_loss'])<2e-6
        with np.load(OUT/side/case/'initial.npz') as z:initial.append({k:z[k].copy() for k in ['global_orient','body_pose','transl']})
        pred_R=cv2.Rodrigues(p['global_orient'].reshape(3).astype(float))[0]
        for j in [3,6,9,12,15]:pred_R=pred_R@cv2.Rodrigues(p['body_pose'].reshape(21,3)[j-1].astype(float))[0]
        err=float(np.max(np.abs(pred_R-np.array(q['final']['head_rotation']))));assert err<2e-6
        checks.append(dict(side=side,case=case,independent_head_rotation_max_abs=err,fixed_parameters_equal=True,selected_points=len(q['selected_points']),reload_checks_pass=True,all_parameters_jointly_optimized=True,contour_legs_active=False))
    assert all(np.array_equal(initial[0][k],initial[1][k]) for k in initial[0])
result=dict(exit_status=0,case_count=4,checks=checks,command='<BODY_PYTHON> runs/2026-11-08_synthetic-head-se3-contour/verify.py',common_initialization_bitwise_equal=True,gt_head_transform_independently_verified=True,source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in RUN.glob('*.py')})
(RUN/'evidence/verification.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS saved-parameter and Head SE3 verification')
