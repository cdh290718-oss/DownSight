"""Held-out GT evaluation; render each shared mesh through both true cameras."""
import hashlib,importlib.util,json,os
from pathlib import Path
import cv2
import numpy as np
import torch
from common import ROOT,RUN,OUT,OLD,read,head_rotation_numpy,angle_deg,project_numpy
from stages import stages
from egosmplx_pipeline.bootstrap import setup
from egosmplx_pipeline.io import load
cfg=read(OLD/'configs/left.json');paths=cfg['paths'];os.environ.update(EGOSMPLX_CONFIG=paths['egosmplx_config'],EGOSMPLX_REPO_ROOT=paths['egosmplx_repo']);model,_=setup(paths['egosmplx_repo'])
from egosmplx_pipeline import fusion
fusion.initialize(model);inputs=read(OUT/'inputs.json');oracle=read(OUT/'anchors/oracle.json');D=OLD/'dataset/samples/000001';PREVIOUS=ROOT/'outputs/synthetic_stereo_abcd_20261108';names=read(OLD/'dataset/joint_names_egosmplx65.json')['names'];assert list(model.joints_name[:65])==names
spec=importlib.util.spec_from_file_location('raster',ROOT/'runs/2026-11-04_mesh-projection-review/code/rasterizer.py');raster=importlib.util.module_from_spec(spec);spec.loader.exec_module(raster)
head_ids=model.layer['neutral'].lbs_weights.detach().numpy().argmax(1)==15

def draw(side,label,mesh,uv,image,rays,truth_uv,prediction_uv):
    depth=np.full((720,1280),np.inf,np.float32);colors=np.zeros_like(image);hp=np.zeros((720,1280),np.float32)
    drawn,rejected=raster.raster(mesh.astype(np.float32),np.asarray(model.face,np.int32),head_ids.astype(np.float32),uv.astype(np.float32),rays,depth,colors,hp,180.)
    valid=np.isfinite(depth);result=image.copy();result[valid]=(.45*result[valid]+.55*colors[valid]).astype(np.uint8)
    for points,color,radius in [(truth_uv,(0,230,0),5),(prediction_uv,(0,40,255),3)]:
        for i in [1,2,3,4,5,6,8,9,10,11,12,13]:
            p=points[i]
            if np.isfinite(p).all() and 0<=p[0]<1280 and 0<=p[1]<720:cv2.circle(result,tuple(np.rint(p).astype(int)),radius,color,-1,cv2.LINE_AA)
    cv2.rectangle(result,(0,0),(1280,40),(25,25,25),-1);cv2.putText(result,side+' / '+label+' / green: GT, red: prediction',(12,27),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1,cv2.LINE_AA)
    filename=side+'_'+label+'.jpg';cv2.imwrite(str(RUN/'assets'/filename),result,[cv2.IMWRITE_JPEG_QUALITY,94])
    return dict(image='assets/'+filename,drawn_faces=int(drawn),rejected_faces=int(rejected),visible_pixels=int(valid.sum()))

def stats(x):
    a=np.asarray(x);valid=a[np.isfinite(a)];return dict(count=len(valid),total_count=len(a),invalid_count=int(len(a)-len(valid)),mean=float(np.mean(valid)) if len(valid) else None,max=float(np.max(valid)) if len(valid) else None)

views=[]
for side in ['left','right']:
    camera=inputs['views'][side]['camera'];T=np.array(camera['left_to_view']);R=T[:3,:3];tr=T[:3,3];K=np.array(camera['K']);dist=np.array(camera['distortion'])
    with np.load(D/(side+'_egosmplx65.npz')) as z:truth=z['joints_3d_camera_m'];truth_uv=z['joints_2d_px'];inside=z['in_frame_no_occlusion_test']
    with np.load(D/(side+'_vertices.npz')) as z:headtruth=z['joints_camera_cv_m'][15]
    yy,xx=np.indices((720,1280),dtype=float);pixels=np.stack([xx.ravel(),yy.ravel()],1);normal=cv2.undistortPointsIter(pixels[:,None,:],K,dist,None,None,(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_MAX_ITER,30,1e-10)).reshape(-1,2);rays=np.column_stack([normal,np.ones(len(normal))]);rays/=np.linalg.norm(rays,axis=1,keepdims=True);rays=rays.reshape(720,1280,3).astype(np.float32)
    image=cv2.imread(str(D/(side+'.png')));row=dict(side=side,camera='cam3' if side=='left' else 'cam4',stages={},truth_head_position_m=headtruth.tolist(),truth_pelvis_position_m=truth[0].tolist())
    for stage in stages():
        label=stage['label'];folder=stage['folder'];path=folder/(stage['file']+'.npz')
        assert path.exists(), str(path)
        p=load(path);mesh,joints=fusion.geometry(p);meshview=mesh@R.T+tr;prediction=joints[:65]@R.T+tr
        with torch.no_grad():native=model.layer['neutral'](**{k:torch.as_tensor(p[k],dtype=torch.float32).reshape(1,-1) for k in fusion.PARAMS}).joints[0].numpy()+p['transl'].reshape(3)
        head=native[15]@R.T+tr;delta=(prediction-truth)*1000;uv=project_numpy(joints[:65],camera);vuv=project_numpy(mesh,camera);valid=inside&np.isfinite(uv).all(1);body=np.arange(65)<25
        headuv=vuv[head_ids];hin=np.isfinite(headuv).all(1)&(headuv[:,0]>=0)&(headuv[:,0]<1280)&(headuv[:,1]>=0)&(headuv[:,1]<720)
        automatic={}
        for group in ['body','wrist','shoulder']:
            ps=[q for q in inputs['views'][side]['body_points'] if q['group']==group];ii=[q['index'] for q in ps];target=np.array([q['xy'] for q in ps]);automatic[group]=stats(np.linalg.norm(uv[ii]-target,axis=1))
        for i,h in enumerate(['left','right']):
            q=inputs['views'][side]['hands'].get(h)
            if q:
                ii=np.array([12+i]+list(range(25+i*20,45+i*20)))[q['included_indices']];target=np.array(q['xy'])[q['included_indices']];automatic['hand_'+h]=stats(np.linalg.norm(uv[ii]-target,axis=1))
        target_ids=[1,2,3,4,5,6,8,9,10,11,12,13];other_ids=[i for i in range(25) if i not in target_ids];feet=np.array([14,15,16,17,18,19]);ankles=np.array([5,5,5,6,6,6])
        opt=read(folder/'optimization.json');caps=opt['hard_caps_deg'];snapshot=opt if stage['case']=='original' else opt['initial'] if stage['budget']==0 else opt['budgets'][str(stage['budget'])];angles=opt['pose_increments_deg'] if stage['case']=='original' else snapshot['pose_absolute_deg'];saturation=[n for n,x in angles.items() if x>=caps[n]-.01]
        result=dict(body_targets12_mean_mm=float(np.linalg.norm(delta[target_ids],axis=1).mean()),other_body13_mean_mm=float(np.linalg.norm(delta[other_ids],axis=1).mean()),feet6_mean_mm=float(np.linalg.norm(delta[feet],axis=1).mean()),feet6_ankle_aligned_mean_mm=float(np.linalg.norm(delta[feet]-delta[ankles],axis=1).mean()),body25_absolute_mean_mm=float(np.linalg.norm(delta[:25],axis=1).mean()),body25_excluding_hips_mean_mm=float(np.linalg.norm(delta[[i for i in range(25) if i not in [1,2]]],axis=1).mean()),pelvis_knees_ankles_mean_mm=float(np.linalg.norm(delta[[0,3,4,5,6]],axis=1).mean()),knees_mean_mm=float(np.linalg.norm(delta[[3,4]],axis=1).mean()),ankles_mean_mm=float(np.linalg.norm(delta[[5,6]],axis=1).mean()),body25_pelvis_aligned_mean_mm=float(np.linalg.norm(delta[:25]-delta[:1],axis=1).mean()),pelvis_absolute_error_mm=float(np.linalg.norm(delta[0])),pelvis_error_xyz_mm=delta[0].tolist(),pelvis_predicted_m=prediction[0].tolist(),wrists_absolute_mean_mm=float(np.linalg.norm(delta[[12,13]],axis=1).mean()),elbows_absolute_mean_mm=float(np.linalg.norm(delta[[10,11]],axis=1).mean()),hands40_absolute_mean_mm=float(np.linalg.norm(delta[25:65],axis=1).mean()),body25_inframe_projection_px=stats(np.linalg.norm(uv[valid&body]-truth_uv[valid&body],axis=1)),body_invalid_prediction_count=int((inside&body&~np.isfinite(uv).all(1)).sum()),body_truth_inframe_count=int((inside&body).sum()),head_position_error_mm=float(np.linalg.norm(head-headtruth)*1000),head_orientation_error_deg=angle_deg(R@head_rotation_numpy(p),R@np.array(oracle['rotation'])),head_vertices_inframe_no_occlusion_count=int(hin.sum()),minimum_mesh_z_m=float(meshview[:,2].min()),shoulder_projected_width_px=float(np.linalg.norm(uv[8]-uv[9])) if np.isfinite(uv[[8,9]]).all() else None,automatic_target_residuals_px=automatic,saturated_pose_increments=saturation,best_step=opt['best_step'] if stage['case']=='original' else snapshot['step'],objective=snapshot['selected_loss'],source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),hand_model='fixed network hand/face parameters' if stage['case']=='original' else 'common zero hand/face inputs; layer mean preserved; no MANO fusion',per_joint=[dict(name=names[i],absolute_mm=float(np.linalg.norm(delta[i])),error_xyz_mm=delta[i].tolist(),pelvis_aligned_mm=float(np.linalg.norm(delta[i]-delta[0])),pelvis_relative_error_xyz_mm=(delta[i]-delta[0]).tolist(),truth_inframe=bool(inside[i]),projection_px=float(np.linalg.norm(uv[i]-truth_uv[i])) if valid[i] else None) for i in range(65)])
        if stage['case']=='original' or stage['budget'] in [3400,20000]:result.update(draw(side,label,meshview,vuv,image,rays,truth_uv,uv))
        row['stages'][label]=result
    views.append(row)
result=dict(scope='One synthetic posture, four shape/pose initialization factors; original procedure as separate reference. Every new case uses the same seed-independent objective and fixed hand/face/foot parameters.',gt_usage='True Head and generating cameras only in fitting. All body and WiLoR observations automatic. GT body/hand/shape used only for evaluation. Mean shape equals GT in this sample.',evaluation='Absolute camera-space errors except explicitly aligned fields. Same shared body in two cameras, not independent samples. Original saturation is relative to network; new-case saturation is absolute local angle.',views=views)
(RUN/'evidence/metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print('evaluated',len(views),'views',len(views[0]['stages']),'stages')
