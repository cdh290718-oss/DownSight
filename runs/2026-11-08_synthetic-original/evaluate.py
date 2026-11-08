"""Compare reconstructed geometry with held-out synthetic ground truth."""
import argparse,hashlib,importlib.util,json,os
from pathlib import Path
import cv2,numpy as np
from egosmplx_pipeline.bootstrap import setup
from egosmplx_pipeline.io import read,load

ROOT=Path(__file__).resolve().parents[2];RUN=Path(__file__).resolve().parent
OUT=ROOT/'outputs/synthetic_original_20261108';DATA=OUT/'dataset';S=DATA/'samples/000001'
ap=argparse.ArgumentParser();ap.add_argument('--partial',action='store_true');args=ap.parse_args()
RECORDS=read(OUT/'inputs.json')['records'];attempts={q['side']:q for q in read(OUT/'attempts.json')}
cfg=read(Path(RECORDS[0]['config']));os.environ.update(EGOSMPLX_CONFIG=cfg['paths']['egosmplx_config'],EGOSMPLX_REPO_ROOT=cfg['paths']['egosmplx_repo'])
model,Camera=setup(cfg['paths']['egosmplx_repo'])
from egosmplx_pipeline import fusion
fusion.initialize(model)
spec=importlib.util.spec_from_file_location('review_rasterizer',ROOT/'runs/2026-11-04_mesh-projection-review/code/rasterizer.py')
raster=importlib.util.module_from_spec(spec);spec.loader.exec_module(raster)
names=read(DATA/'joint_names_egosmplx65.json')['names'];assert list(model.joints_name[:65])==names
g=load(S/'geometry.npz');topology_equal=np.array_equal(model.face,g['faces'])
head_mask=model.layer['neutral'].lbs_weights.detach().cpu().numpy().argmax(1)==model.orig_joints_name.index('Head')

def stats(values):
    v=np.asarray(values,float)
    return dict(count=int(len(v)),mean=float(v.mean()) if len(v) else None,rmse=float(np.sqrt(np.mean(v*v))) if len(v) else None,
                median=float(np.median(v)) if len(v) else None,max=float(v.max()) if len(v) else None,values=v.tolist())

def brown_project(xyz,K,dist):
    xyz=np.asarray(xyz,dtype=np.float64);valid=np.isfinite(xyz).all(1)&(xyz[:,2]>1e-6)
    out=np.full((len(xyz),2),np.nan)
    if valid.any():out[valid]=cv2.projectPoints(xyz[valid],np.zeros(3),np.zeros(3),K,dist)[0].reshape(-1,2)
    return out

def truth_arrays(path,xyz_key,uv_key):
    # Synthetic GT intentionally stores NaN 2D positions for points behind
    # the camera; strict pipeline NPZ loading is only appropriate for outputs.
    with np.load(path,allow_pickle=False) as source:
        result={k:source[k].copy() for k in source.files}
    xyz=result[xyz_key];uv=result[uv_key]
    assert np.isfinite(xyz).all()
    assert np.isfinite(uv[xyz[:,2]>0]).all()
    assert not np.isfinite(uv[xyz[:,2]<=0]).any()
    return result

def camera_rays(camera,truth_camera=None):
    yy,xx=np.indices((720,1280),dtype=np.float64);pixels=np.stack([xx.ravel(),yy.ravel()],axis=1)
    if truth_camera is None:
        return camera.camera2world_ray(pixels).reshape(720,1280,3).astype(np.float32)
    normalized=cv2.undistortPointsIter(pixels.reshape(-1,1,2),truth_camera['K_zero_based'],truth_camera['distortion'],None,None,
                                      (cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_MAX_ITER,30,1e-10)).reshape(-1,2)
    rays=np.column_stack((normalized,np.ones(len(normalized))))
    rays/=np.maximum(np.linalg.norm(rays,axis=1,keepdims=True),1e-12)
    return rays.reshape(720,1280,3).astype(np.float32)

def render(image,vertices,faces,uv,rays,truth=False):
    depth=np.full((720,1280),np.inf,np.float32);colors=np.zeros_like(image);head_pixel=np.zeros((720,1280),np.float32)
    raster.raster(vertices.astype(np.float32),faces.astype(np.int32),np.zeros(len(vertices),np.float32),uv.astype(np.float32),rays,depth,colors,head_pixel,180.)
    valid=np.isfinite(depth);overlay=image.copy()
    if truth:
        # The same geometry, shaded green rather than the prediction's cyan.
        level=colors[valid].astype(np.float32).mean(1)/146.7
        colors[valid]=np.clip(level[:,None]*np.array([55,210,70]),0,255).astype(np.uint8)
    overlay[valid]=(.45*image[valid]+.55*colors[valid]).astype(np.uint8)
    return overlay

def mark(image,uv,mask,color):
    for i in [8,9,10,11,12,13,25,29,33,37,41,45,49,53,57,61]:
        if mask[i] and np.isfinite(uv[i]).all() and 0<=uv[i,0]<1280 and 0<=uv[i,1]<720:
            xy=tuple(np.rint(uv[i]).astype(int));cv2.circle(image,xy,4,color,-1,cv2.LINE_AA)
    return image

assets=OUT/'partial_assets' if args.partial else RUN/'assets';assets.mkdir(exist_ok=True)
rows=[]
for rec in RECORDS:
    side=rec['id']
    if side not in attempts:
        if args.partial:continue
        raise ValueError('Missing execution record for '+side)
    config=read(Path(rec['config']));frame=Path(config['output'])/'frames'/side
    gt=truth_arrays(S/f'{side}_egosmplx65.npz','joints_3d_camera_m','joints_2d_px')
    gv=truth_arrays(S/f'{side}_vertices.npz','vertices_camera_cv_m','vertices_2d_px')
    cam=Camera(read(Path(config['manifest']))['frames'][0]['calibration'])
    truth=gt['joints_3d_camera_m'].astype(float);truth_uv=gt['joints_2d_px'].astype(float);inside=gt['in_frame_no_occlusion_test'].astype(bool)
    assert list(gt['joint_names'])==names
    assert np.array_equal(gt['smplx_joint_indices'],np.asarray(model.joint_idx[:65]))
    camera_truth=cam.world2camera(truth.astype(np.float64));m=inside&np.isfinite(camera_truth).all(1)
    camera_gap=stats(np.linalg.norm(camera_truth[m]-truth_uv[m],axis=1))
    image=cv2.imread(str(S/f'{side}.png'));assert image.shape==(720,1280,3)
    gt_overlay=render(image,gv['vertices_camera_cv_m'],g['faces'],gv['vertices_2d_px'],camera_rays(cam,gv),truth=True)
    mark(gt_overlay,truth_uv,inside,(10,255,10));cv2.imwrite(str(assets/f'{side}_truth.jpg'),gt_overlay,[cv2.IMWRITE_JPEG_QUALITY,94])
    row=dict(side=side,camera=rec['camera'],execution_exit=attempts[side]['exit_status'],truth_in_frame_count=int(inside.sum()),
             camera_model_difference_px=camera_gap,truth_image=f'assets/{side}_truth.jpg',stages={},ground_truth_occlusion_labels=False,
             truth_mesh_depth=dict(near_plane_threshold_m=.02,vertex_count=int(len(gv['vertices_camera_cv_m'])),behind_camera_count=int((gv['vertices_camera_cv_m'][:,2]<0).sum()),
                 below_2cm_count=int((gv['vertices_camera_cv_m'][:,2]<.02).sum()),minimum_z_m=float(gv['vertices_camera_cv_m'][:,2].min())))
    if topology_equal:
        tuv=gv['vertices_2d_px'];front=gv['vertices_camera_cv_m'][:,2]>0
        row['truth_head_vertices_in_frame']=int((head_mask&front&np.isfinite(tuv).all(1)&(tuv[:,0]>=0)&(tuv[:,0]<1280)&(tuv[:,1]>=0)&(tuv[:,1]<720)).sum())
    if attempts[side]['exit_status']!=0:
        row['failure']='Original pipeline did not complete; no model result fabricated.';rows.append(row);continue
    validation=read(frame/'validation.json');row['validation']={k:validation.get(k) for k in ['passed','pipeline_profile','matched_hands','matches','local_surface_accepted','body_reload_max_abs_m','fusion_reload_max_abs_m','obj_reload_max_abs_m']}
    final=frame/'fused/body_params.npz'
    if not final.exists():final=frame/'body/params.npz'
    stages={'raw':frame/'raw/params.npz','rigid':frame/'rigid/optimized_params.npz','body':frame/'body/params.npz','final':final}
    for stage,path in stages.items():
        p=load(path);vertices,joints=fusion.geometry(p);prediction=joints[:65].astype(float)
        if stage=='final':
            mesh=load(frame/'fused/mesh.npz');vertices=mesh['vertices_cam'].astype(float)
            for si,hand in enumerate(['left','right']):prediction[25+si*20:45+si*20]=(model.orig_hand_regressor[hand]@vertices)[1:]
        absolute=np.linalg.norm(prediction-truth,axis=1)*1000
        aligned=np.linalg.norm((prediction-prediction[:1])-(truth-truth[:1]),axis=1)*1000
        uv_poly=cam.world2camera(prediction);uv_brown=brown_project(prediction,gv['K_zero_based'],gv['distortion'])
        valid_poly=inside&(prediction[:,2]>1e-6)&np.isfinite(uv_poly).all(1)
        valid_brown=inside&np.isfinite(uv_brown).all(1)
        groups={}
        for name,indices in [('body25',np.arange(25)),('upper6',np.arange(8,14)),('left_fingers20',np.arange(25,45)),('right_fingers20',np.arange(45,65))]:
            mask=np.zeros(65,bool);mask[indices]=True
            groups[name]=dict(absolute_mm=stats(absolute[mask]),pelvis_aligned_mm=stats(aligned[mask]),
                ground_truth_in_frame=int((mask&inside).sum()),absolute_in_frame_mm=stats(absolute[mask&inside]),
                projection_original_camera_px=stats(np.linalg.norm(uv_poly[mask&valid_poly]-truth_uv[mask&valid_poly],axis=1)),
                projection_brown_camera_px=stats(np.linalg.norm(uv_brown[mask&valid_brown]-truth_uv[mask&valid_brown],axis=1)))
            if name in ('left_fingers20','right_fingers20'):
                wrist=12 if name.startswith('left') else 13
                wrist_aligned=np.linalg.norm((prediction-prediction[wrist])-(truth-truth[wrist]),axis=1)*1000
                groups[name]['wrist_aligned_mm']=stats(wrist_aligned[mask])
                groups[name]['wrist_depth_error_mm']=float((prediction[wrist,2]-truth[wrist,2])*1000)
        result=dict(groups=groups,per_joint=[dict(name=names[i],truth_in_frame=bool(inside[i]),prediction_positive_depth=bool(prediction[i,2]>1e-6),
            absolute_mm=float(absolute[i]),pelvis_aligned_mm=float(aligned[i]),
            projection_original_px=float(np.linalg.norm(uv_poly[i]-truth_uv[i])) if valid_poly[i] else None,
            projection_brown_px=float(np.linalg.norm(uv_brown[i]-truth_uv[i])) if valid_brown[i] else None) for i in range(65)])
        if stage=='final' and topology_equal and vertices.shape==gv['vertices_camera_cv_m'].shape:
            result['vertex_all_absolute_mm']=stats(np.linalg.norm(vertices-gv['vertices_camera_cv_m'],axis=1)*1000)
            result['vertex_all_pelvis_aligned_mm']=stats(np.linalg.norm((vertices-prediction[:1])-(gv['vertices_camera_cv_m']-truth[:1]),axis=1)*1000)
            for key in ['vertex_all_absolute_mm','vertex_all_pelvis_aligned_mm']:result[key].pop('values')
        if stage=='final':
            vertex_uv=cam.world2camera(vertices)
            if topology_equal:
                result['head_vertices_in_frame']=int((head_mask&(vertices[:,2]>0)&np.isfinite(vertex_uv).all(1)&(vertex_uv[:,0]>=0)&(vertex_uv[:,0]<1280)&(vertex_uv[:,1]>=0)&(vertex_uv[:,1]<720)).sum())
            overlay=render(image,vertices,mesh['faces'],vertex_uv,camera_rays(cam))
            mark(overlay,truth_uv,inside,(20,255,20));mark(overlay,uv_poly,np.isfinite(uv_poly).all(1)&(prediction[:,2]>1e-6),(20,50,255))
            cv2.imwrite(str(assets/f'{side}_prediction.jpg'),overlay,[cv2.IMWRITE_JPEG_QUALITY,94]);row['prediction_image']=f'assets/{side}_prediction.jpg'
        row['stages'][stage]=result
    rows.append(row)
result=dict(dataset_commit='192671271a74b99f41b9eb033d8f88b1dac77ada',profile='mano_guided_original_seam_20260929_v1',
    scope='One synthetic posture, two independent camera-view fits; GT used only after inference',body_joint_count=25,hand_joint_count=40,
    topology_exactly_matches_backend=bool(topology_equal),units={'joint_3d':'millimeters','projection':'pixels'},
    visibility='Ground-truth in_frame_no_occlusion_test means in front and inside image only; actual occlusion is not assessed',views=rows)
(OUT/'partial_metrics.json' if args.partial else RUN/'evidence/metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({'topology_equal':topology_equal,'results':[{k:r.get(k) for k in ['side','execution_exit','failure']} for r in rows]},indent=2))
