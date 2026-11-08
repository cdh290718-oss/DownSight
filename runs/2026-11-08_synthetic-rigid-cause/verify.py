"""Check control inputs, reproduction, exit statuses, and scale arithmetic."""
import ast,json,hashlib
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[2];RUN=Path(__file__).resolve().parent;O=R/'outputs/synthetic_original_20261108';D=R/'outputs/synthetic_rigid_cause_20261108';f=O/'fits/left/frames/left'
for p in RUN.glob('*.py'):ast.parse(p.read_text())
execution=[]
for n,count in [('execution.json',12),('group_execution.json',3),('no_depth_execution.json',6),('group_no_depth_execution.json',4)]:
 x=json.loads((RUN/'evidence'/n).read_text());assert len(x)==count;execution+=x
raw=np.load(f/'raw/params.npz');gt=np.load(O/'dataset/samples/000001/smplx_reconstruction.npz')
for shape in ['pred_shape','gt_shape']:
 for pose in ['pred_pose','gt_pose']:
  a=np.load(D/(shape+'_'+pose+'.npz'));assert np.array_equal(a['betas'],(gt if shape=='gt_shape' else raw)['betas'].reshape(a['betas'].shape));assert np.array_equal(a['body_pose'],(gt if pose=='gt_pose' else raw)['body_pose'].reshape(a['body_pose'].shape))
a=np.load(f/'rigid/optimized_params.npz');b=np.load(D/'pred_shape_pred_pose_detector_all/optimized_params.npz')
diff={k:float(np.max(np.abs(a[k]-b[k]))) for k in ['global_orient','body_pose','transl','smplx_mesh_cam']};assert max(diff.values())<1e-6
assert sum(v['exit_status']==0 for v in execution)==24 and [v['name'] for v in execution if v['exit_status']!=0]==['gt_shape_gt_pose_oracle_inframe_no_depth']
r=json.loads((RUN/'evidence/results.json').read_text());assert len(r['cases'])==24; assert abs(r['forearm_scale_convention']['scaled_pelvis_error_mm']-194.0786584024611)<1e-7
hashes={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in {'original_raw_params':f/'raw/params.npz','original_rigid_params':f/'rigid/optimized_params.npz','gt_smplx':O/'dataset/samples/000001/smplx_reconstruction.npz','gt_joints':O/'dataset/samples/000001/left_egosmplx65.npz','sapiens_keypoints':O/'fits/left/observations/left/sapiens/keypoints.json','calibration':O/'dataset/calibration/cam3_left_lower.egosmplx.json'}.items()}
result={'exit_status':0,'case_count':len(execution),'baseline_control_max_abs_differences':diff,'checks':['All Python scripts parse','24 stage runs succeeded, one oracle no-depth case failed its reload pixel check','four inputs use intended betas and body pose','baseline matches original rigid outputs','scale convention arithmetic verified'],'input_sha256':hashes,'stage_runners':'direct body_rigid diagnostic calls, not full pipeline; no full pipeline provenance created','environment':'original body Python environment, CPU; four threads except group controls two threads; unchanged original model and body code'}
(RUN/'evidence/verification.json').write_text(json.dumps(result,indent=2)+'\n');print('verification pass')
