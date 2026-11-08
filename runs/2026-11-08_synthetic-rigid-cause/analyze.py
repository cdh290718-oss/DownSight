"""Evaluate rigid controls in common camera coordinates and record scale convention."""
import json,os,hashlib
from pathlib import Path
import numpy as np
from egosmplx_pipeline.io import read,load
from egosmplx_pipeline.bootstrap import setup
R=Path(__file__).resolve().parents[2];RUN=Path(__file__).resolve().parent;O=R/'outputs/synthetic_original_20261108';D=R/'outputs/synthetic_rigid_cause_20261108';S=O/'dataset/samples/000001';cfg=read(O/'configs/left.json');os.environ.update(EGOSMPLX_CONFIG=cfg['paths']['egosmplx_config'],EGOSMPLX_REPO_ROOT=cfg['paths']['egosmplx_repo']);model,Camera=setup(cfg['paths']['egosmplx_repo'])
from egosmplx_pipeline import fusion
fusion.initialize(model)
with np.load(S/'left_egosmplx65.npz',allow_pickle=False) as gt:truth=gt['joints_3d_camera_m'][:25].astype(float);gtuv=gt['joints_2d_px'][:25].copy();inside=gt['in_frame_no_occlusion_test'][:25].copy()
names=read(O/'dataset/joint_names_egosmplx65.json')['names'];cam=Camera(str(O/'dataset/calibration/cam3_left_lower.egosmplx.json'))
records=read(RUN/'evidence/execution.json'); group=RUN/'evidence/group_execution.json'; records += read(group) if group.exists() else []; nd=RUN/'evidence/no_depth_execution.json'; records += read(nd) if nd.exists() else []; gnd=RUN/'evidence/group_no_depth_execution.json'; records += read(gnd) if gnd.exists() else []; rows=[]
for item in records:
 if item['exit_status']!=0:continue
 folder=D/item['name'];p=load(folder/'optimized_params.npz');mesh,joints=fusion.geometry(p);pred=joints[:25].astype(float);delta=(pred-truth)*1000;report=read(folder/'optimization_report.json');px=cam.world2camera(pred);selected=report['selected']
 pcenter=pred-pred[:1];gcenter=truth-truth[:1]
 # Best isotropic scale in pelvis-centered 3D, separately allowing a 3D rotation.
 u,_,vh=np.linalg.svd(pcenter.T@gcenter);diag=np.eye(3);diag[-1,-1]=np.linalg.det(u@vh);rot=u@diag@vh
 scale=float(np.sum((pcenter@rot)*gcenter)/np.sum(pcenter*pcenter))
 # Similarity centered at camera origin retains image projections for a central camera.
 scaled_depth=float(pred[0,2]*scale);scale_depth_delta=float((scaled_depth-truth[0,2])*1000)
 joints_data={names[i]:{'error_xyz_mm':delta[i].tolist(),'absolute_mm':float(np.linalg.norm(delta[i])),'truth_z_m':float(truth[i,2]),'pred_z_m':float(pred[i,2])} for i in range(25)}
 points=[]
 for q in selected:
    i=q['index'];res=px[i]-np.array([q['x'],q['y']]);normed=res/np.array([1280,720]);beta=.005;ab=np.abs(normed);smooth=np.where(ab<beta,.5*normed**2/beta,ab-.5*beta).mean();weight=min(q['score'],1.)**2
    points.append({'name':q['name'],'score':q['score'],'weight':weight,'fit_to_target_px':float(np.linalg.norm(res)),'weighted_loss_numerator':float(smooth*weight),'truth_in_frame':bool(inside[i]),'truth_uv_positive_depth_px':gtuv[i].tolist() if np.isfinite(gtuv[i]).all() else None,'target_to_truth_outside_allowed_px':float(np.linalg.norm(np.array([q['x'],q['y']])-gtuv[i])) if np.isfinite(gtuv[i]).all() else None,'fit_to_truth_px':float(np.linalg.norm(px[i]-gtuv[i])) if inside[i] else None,'target_to_truth_px':float(np.linalg.norm(np.array([q['x'],q['y']])-gtuv[i])) if inside[i] else None})
 rows.append({'case':item,'joints':joints_data,'body25_absolute_mean_mm':float(np.linalg.norm(delta,axis=1).mean()),'body25_pelvis_aligned_mean_mm':float(np.linalg.norm(delta-delta[:1],axis=1).mean()),'pelvis_centered_optimal_rotation_scale_mean_mm':float(np.linalg.norm(pcenter@rot*scale-gcenter,axis=1).mean()*1000),'pelvis_centered_optimal_rotation_mean_mm':float(np.linalg.norm(pcenter@rot-gcenter,axis=1).mean()*1000),'camera_origin_scale_3d_procrustes':scale,'camera_origin_scaled_pelvis_z_m':scaled_depth,'camera_origin_scaled_pelvis_error_mm':scale_depth_delta,'selected':points,'objective_final':report['history'][-1], 'depth_min_m':float(mesh[:,2].min()),'optimized_params_sha256':hashlib.sha256((folder/'optimized_params.npz').read_bytes()).hexdigest()})
# Compare predicted and truth joint spacings, invariant to global rigid transformation.
orig=next(q for q in rows if q['case']['name']=='pred_shape_pred_pose_detector_all');raw=load(D/'pred_shape_pred_pose.npz');_,j=fusion.geometry(raw);j=j[:25]
pairs=[('hip_width','L_Hip','R_Hip'),('shoulder_width','L_Shoulder','R_Shoulder'),('left_upper_arm','L_Shoulder','L_Elbow'),('right_upper_arm','R_Shoulder','R_Elbow'),('left_forearm','L_Elbow','L_Wrist'),('right_forearm','R_Elbow','R_Wrist'),('left_thigh','L_Hip','L_Knee'),('right_thigh','R_Hip','R_Knee'),('left_shin','L_Knee','L_Ankle'),('right_shin','R_Knee','R_Ankle')]
lengths=[]
for label,a,b in pairs:
 i=names.index(a);k=names.index(b);x=float(np.linalg.norm(j[i]-j[k]));y=float(np.linalg.norm(truth[i]-truth[k]));lengths.append({'name':label,'prediction_mm':x*1000,'truth_mm':y*1000,'pred_to_truth_ratio':x/y})
forearm_ratio=np.mean([q['pred_to_truth_ratio'] for q in lengths if q['name'] in ('left_forearm','right_forearm')]); actual_z=orig['joints']['Pelvis']['pred_z_m']; truth_z=orig['joints']['Pelvis']['truth_z_m']; forearm=dict(predicted_to_truth_mean_ratio=float(forearm_ratio),scale=float(1/forearm_ratio),scaled_pelvis_z_m=float(actual_z/forearm_ratio),scaled_pelvis_error_mm=float((actual_z/forearm_ratio-truth_z)*1000))
result={'forearm_scale_convention':forearm,'scope':'cam3, original rigid optimizer and initialization; truth-shape, truth-pose, truth in-frame masks and truth target coordinates are diagnostic oracle interventions, not predictions','scale_definition':'least squares isotropic scale after pelvis-centering 25 3D joints and optimizing one proper rotation; hypothetical scaling of all predicted camera-space coordinates around camera origin, not a fit, not physical person translation, 2D preserved only for central rays','lengths':lengths,'cases':rows}
(RUN/'evidence/results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print('cases',len(rows))
