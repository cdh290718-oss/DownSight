"""Recompute depth, target residual and mesh guard diagnostics from frozen artifacts."""
import json,os,hashlib
from pathlib import Path
import numpy as np
from egosmplx_pipeline.bootstrap import setup
from egosmplx_pipeline.io import read,load
from egosmplx_pipeline.body_common import JOINT_MAPPING
R=Path(__file__).resolve().parents[2]; RUN=Path(__file__).resolve().parent; O=R/'outputs/synthetic_original_20261108'; D=R/'outputs/synthetic_depth_diagnosis_20261108'; S=O/'dataset/samples/000001'
cfg=read(O/'configs/left.json');os.environ.update(EGOSMPLX_CONFIG=cfg['paths']['egosmplx_config'],EGOSMPLX_REPO_ROOT=cfg['paths']['egosmplx_repo']);model,Camera=setup(cfg['paths']['egosmplx_repo'])
from egosmplx_pipeline import fusion
fusion.initialize(model)
labels=model.layer['neutral'].lbs_weights.detach().cpu().numpy().argmax(1);head=labels==model.orig_joints_name.index('Head');neck=labels==model.orig_joints_name.index('Neck')
names=read(O/'dataset/joint_names_egosmplx65.json')['names']
rows=[]
for side,fit,cal in [('left','left','cam3_left_lower.egosmplx.json'),('right','right_cpu','cam4_right_lower.egosmplx.json')]:
    f=O/'fits'/fit/'frames'/side; obs=read((O/'fits'/fit/'observations'/side/'sapiens/keypoints.json') if side=='left' else (O/'cpu_observations/right_pose/keypoints.json'));lookup={q['name']:q for q in obs['keypoints']}; selected=read(f/'rigid/optimization_report.json')['selected']; selected_names={q['name'] for q in selected}
    with np.load(S/(side+'_egosmplx65.npz'),allow_pickle=False) as g:truth=g['joints_3d_camera_m'].astype(float);uv=g['joints_2d_px'].astype(float);inside=g['in_frame_no_occlusion_test'].astype(bool)
    with np.load(S/(side+'_vertices.npz'),allow_pickle=False) as g: gv=g['vertices_camera_cv_m'].astype(float)
    cam=Camera(str(O/'dataset/calibration'/cal));
    def meshstats(v):
        z=v[:,2];result={'minimum_z_m':float(z.min()),'below_2cm':int((z<.02).sum()),'below_2cm_head':int(((z<.02)&head).sum()),'below_2cm_neck':int(((z<.02)&neck).sum()),'below_2cm_other':int(((z<.02)&~(head|neck)).sum()),'other_minimum_z_m':float(z[~(head|neck)].min())};return result
    stages={'raw':f/'raw/params.npz','rigid':f/'rigid/optimized_params.npz','body':f/'body/params.npz','guided':f/'guided_body/params.npz','final':f/'fused/body_params.npz'}
    if side=='left':
        for sc in ['all','non_head_neck']:
            path=D/sc/'automatic/refined_params.npz'
            if path.exists():stages['ablation_'+sc]=path
        path=D/'oracle/automatic/refined_params.npz'
        if path.exists():stages['oracle_body']=path
        for sc in ['all','non_head_neck','none']:
            path=D/('rigid_'+sc)/'optimized_params.npz'
            if path.exists():stages['rigid_ablation_'+sc]=path
    out={}
    for stage,path in stages.items():
        verts,joints=fusion.geometry(load(path)); pred=joints[:65].astype(float); delta=(pred-truth)*1000; proj=cam.world2camera(pred);per=[]
        for n in JOINT_MAPPING:
            if n not in lookup:continue
            q=lookup[n];ix=names.index(JOINT_MAPPING[n]);target=np.array([q['x'],q['y']]);ok=bool(inside[ix]);
            per.append({'name':n,'selected':n in selected_names,'confidence':q['score'],'gt_in_frame':ok,'detector_error_px':float(np.linalg.norm(target-uv[ix])) if ok else None,'fit_to_detector_px':float(np.linalg.norm(proj[ix]-target)),'fit_to_gt_px':float(np.linalg.norm(proj[ix]-uv[ix])) if ok else None})
        js={names[i]:{'error_xyz_mm':delta[i].tolist(),'absolute_mm':float(np.linalg.norm(delta[i])),'pelvis_aligned_mm':float(np.linalg.norm(delta[i]-delta[0])),'truth_z_m':float(truth[i,2]),'pred_z_m':float(pred[i,2])} for i in range(25)}
        out[stage]={'mesh':meshstats(verts),'joints':js,'body25_mean_absolute_mm':float(np.linalg.norm(delta[:25],axis=1).mean()),'body25_mean_pelvis_aligned_mm':float(np.linalg.norm(delta[:25]-delta[0],axis=1).mean()),'body25_mean_signed_xyz_mm':delta[:25].mean(0).tolist(),'body25_mean_abs_xyz_mm':np.abs(delta[:25]).mean(0).tolist(),'points':per,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    rows.append({'side':side,'truth_mesh':meshstats(gv),'selected_points':sorted(selected_names),'stages':out})
result={'dataset_commit':'192671271a74b99f41b9eb033d8f88b1dac77ada','baseline_commit':'40e2eba6cd9582b0ce420ce0302410a5329e2c78','definition':'z is optical axis; errors are prediction minus GT; detector error evaluated only where GT in front and in frame, no occlusion verification; detector annotations versus internal joint conventions may differ','views':rows}
(RUN/'evidence/diagnosis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print('wrote',len(rows),'views')
