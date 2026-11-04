"""Same-target evaluation and same-renderer images for the fixed demo pairs."""
import argparse,json,os,sys,hashlib,importlib.util,time,copy
from pathlib import Path
import cv2,numpy as np
from egosmplx_pipeline.io import read,load,write
from egosmplx_pipeline.bootstrap import setup

def stats(d):
 d=np.asarray(d,float);return dict(count=len(d),rmse=float(np.sqrt(np.mean(d*d))) if len(d) else None,mean=float(d.mean()) if len(d) else None,median=float(np.median(d)) if len(d) else None,p90=float(np.percentile(d,90)) if len(d) else None,values=d.tolist())

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--inputs',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--wait-seconds',type=int,default=0);ap.add_argument('--previous-metrics',type=Path,required=True);ap.add_argument('--previous-inputs',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 data=read(a.inputs);cfg=read(data['records'][0]['config']);os.environ['EGOSMPLX_CONFIG']=cfg['paths']['egosmplx_config'];os.environ['EGOSMPLX_REPO_ROOT']=cfg['paths']['egosmplx_repo'];os.environ['EGOSMPLX_REFERENCE_ROOT']=str(a.out/'unused')
 model,Camera=setup(cfg['paths']['egosmplx_repo'])
 from egosmplx_pipeline import fusion,v2_policy
 from egosmplx_pipeline.body_common import JOINT_MAPPING
 from egosmplx_pipeline.reference.surface_audit import local_intersections
 from egosmplx_pipeline.reference.contour_seam import adjacency,distance
 fusion.initialize(model)
 spec=importlib.util.spec_from_file_location('rasterizer',Path('runs/2026-11-04_mesh-projection-review/code/rasterizer.py'));raster=importlib.util.module_from_spec(spec);spec.loader.exec_module(raster)
 dominant=model.layer['neutral'].lbs_weights.detach().numpy().argmax(1);head=(dominant==model.orig_joints_name.index('Head'));neck=(dominant==model.orig_joints_name.index('Neck'));bodymask=~(head|neck)
 previous={r['frame']:r for r in read(a.previous_metrics)['frames']};previous_inputs={r['id']:r for r in read(a.previous_inputs)['records']};records=[]
 for rec in data['records']:
  cfg=read(rec['config']);root=Path(cfg['output'])/'frames'/rec['id'];original=rec['original'];cam=Camera(rec['row']['calibration']);im=cv2.imread(rec['row']['image']);lookup={q['name']:q for q in read(rec['row']['predictions']['sapiens'])['keypoints']};w=load(original['wilor']);matches=read(original['recipe'])['matches'];ctx=read(rec['context']);pairid=rec['pair'];name=pairid+'-'+rec['camera']
  names=[n for n in JOINT_MAPPING if n in lookup and lookup[n]['score']>=.3 and np.isfinite([lookup[n]['x'],lookup[n]['y']]).all() and 0<=lookup[n]['x']<1280 and 0<=lookup[n]['y']<720]
  ids=[model.joints_name.index(JOINT_MAPPING[n]) for n in names];goal=np.array([[lookup[n]['x'],lookup[n]['y']] for n in names]);active=np.array([n in v2_policy.EXEMPT or v2_policy.interior(goal[i]) for i,n in enumerate(names)])
  Y,X=np.indices((720,1280));rays=cam.camera2world_ray(np.stack([X.ravel(),Y.ravel()],axis=1)).reshape(720,1280,3).astype(np.float32)
  result=dict(pair=pairid,camera=rec['camera'],frame=rec['id'],body_names=names,retained_body_mask=active.tolist(),time_difference_seconds=ctx['timestamp_difference_seconds'],methods={})
  for mode in ('original','v2'):
   if mode=='v2' and a.wait_seconds:
    deadline=time.monotonic()+a.wait_seconds;statefile=Path(cfg['output'])/'run_state.json'
    while time.monotonic()<deadline:
     state=read(statefile).get('state') if statefile.exists() else None
     if state in ('complete','failed'):break
     time.sleep(10)
   if mode=='v2' and (not (Path(cfg['output'])/'run_state.json').exists() or read(Path(cfg['output'])/'run_state.json').get('state')!='complete'):
    result['methods'][mode]={'failed':True};continue
   mesh=load(original['mesh'] if mode=='original' else root/'fused/mesh.npz');p=load(original['body'] if mode=='original' else root/'fused/body_params.npz')
   vertices=mesh['vertices_cam'].astype(float);faces=mesh['faces'];_,j=fusion.geometry(p);uv=cam.world2camera(j.astype(float));errs=np.linalg.norm(uv[ids]-goal,axis=1)
   m=dict(body_all=stats(errs),body_retained=stats(errs[active]),hands={},body_min_z_m=float(vertices[bodymask,2].min()),head_visible=int((head&(vertices[:,2]>0)&np.isfinite(cam.world2camera(vertices)).all(1)&(cam.world2camera(vertices)[:,0]>=0)&(cam.world2camera(vertices)[:,0]<1280)&(cam.world2camera(vertices)[:,1]>=0)&(cam.world2camera(vertices)[:,1]<720)).sum()),groups={})
   for g,mask in [('nose',[n=='nose' for n in names]),('lower',[n.endswith(('_hip','_knee','_ankle')) for n in names]),('wrist',[n.endswith('_wrist') for n in names]),('shoulder_elbow',[n.endswith(('_shoulder','_elbow')) for n in names])]:m['groups'][g]=stats(errs[mask])
   m['body_shoulders']=stats(errs[[n.endswith('_shoulder') for n in names]]);m['pelvis_z_m']=float(j[0,2]);m['projected_shoulder_distance_px']=float(np.linalg.norm(uv[8]-uv[9]))
   handj=np.stack([model.orig_hand_regressor[s]@vertices for s in ('left','right')]);kin=handj.copy();kin[:,0]=j[[12,13]]
   for si,side in enumerate(('left','right')):
    if side not in matches:m['hands'][side]=None;continue
    g=w['joints_2d'][matches[side]['candidate']];pred=cam.world2camera(handj[si]);ok=np.isfinite(g).all(1)&(g[:,0]>=0)&(g[:,0]<1280)&(g[:,1]>=0)&(g[:,1]<720);interior=ok&v2_policy.interior(g);error=np.linalg.norm(pred-g,axis=1)
    tm=np.array(ctx['tri_weights'])[si]>0;target=np.array(ctx['tri'])[si];delta=kin[si]-target;dist=np.linalg.norm(delta,axis=1);ray=target/np.maximum(np.linalg.norm(target,axis=1,keepdims=True),1e-10);longitudinal=np.sum(delta*ray,axis=1);transverse=np.linalg.norm(delta-longitudinal[:,None]*ray,axis=1)
    adj=adjacency(faces,len(vertices));seeds=mesh[side+'_local_to_mesh'][mesh[side+'_wrist_loop_local']];hops=distance(adj,seeds);local=np.flatnonzero((hops>0)&(hops<10));cross=local_intersections(vertices,faces,local)
    m['hands'][side]=dict(all=stats(error[ok]),retained=stats(error[interior]),wrist_2d_px=float(error[0]) if ok[0] else None,triangulated_3d_m=stats(dist[tm]),triangulated_depth_m=stats(np.abs(longitudinal[tm])),triangulated_transverse_m=stats(transverse[tm]),triangulated_wrist_m=float(dist[0]) if tm[0] else None,proper_crossings=cross['proper_crossings'],intersection_face_pairs=cross['face_pairs'],local_faces_tested=cross['local_faces_tested'])
   colors=np.zeros_like(im);depth=np.full((720,1280),np.inf,np.float32);hp=np.zeros((720,1280),np.float32);projected=cam.world2camera(vertices).astype(np.float32)
   drawn,rejected=raster.raster(vertices.astype(np.float32),faces.astype(np.int32),head.astype(np.float32),projected,rays,depth,colors,hp,180.)
   visible=np.isfinite(depth);overlay=im.copy();overlay[visible]=(.45*im[visible]+.55*colors[visible]).astype(np.uint8)
   file=name+'-'+mode+'.jpg';assert cv2.imwrite(str(a.out/file),overlay,[cv2.IMWRITE_JPEG_QUALITY,93])
   m['image']=file;m['render']=dict(visible_pixels=int(visible.sum()),head_pixels=int((hp>.5).sum()),triangles_drawn=int(drawn),triangles_rejected=int(rejected),scene_occlusion=False)
   if mode=='v2':m['validation']=read(root/'validation.json')
   result['methods'][mode]=m
  prior=previous[rec['id']]
  assert prior['body_names']==names and prior['retained_body_mask']==active.tolist()
  assert np.allclose(prior['methods']['original']['body_all']['values'],result['methods']['original']['body_all']['values'],atol=1e-6,rtol=0)
  for side in ('left','right'):
   oldhand=prior['methods']['original']['hands'][side];newhand=result['methods']['original']['hands'][side]
   assert (oldhand is None)==(newhand is None)
   if oldhand:assert np.allclose(oldhand['all']['values'],newhand['all']['values'],atol=1e-6,rtol=0)
  earlier=copy.deepcopy(prior['methods']['v2']);oldcfg=read(previous_inputs[rec['id']]['config']);oldparams=load(Path(oldcfg['output'])/'frames'/rec['id']/'fused/body_params.npz');_,oldj=fusion.geometry(oldparams);olduv=cam.world2camera(oldj.astype(float))
  earlier['body_shoulders']=stats(np.asarray(earlier['body_all']['values'])[[n.endswith('_shoulder') for n in names]]);earlier['pelvis_z_m']=float(oldj[0,2]);earlier['projected_shoulder_distance_px']=float(np.linalg.norm(olduv[8]-olduv[9]))
  result['methods']={'original':result['methods']['original'],'edge40':earlier,'no_margin':result['methods']['v2']}
  records.append(result);write(a.out/'metrics.local.json',dict(frames=records));print('EVALUATED',name,flush=True)
 # No machine paths or full per-worker provenance in public metrics.
 for r in records:
  if 'validation' in r['methods'].get('no_margin',{}):
   v=r['methods']['no_margin'].pop('validation');r['methods']['no_margin']['validation']={k:v.get(k) for k in ['passed','pipeline_profile','matched_hands','body_reload_max_abs_m','fusion_reload_max_abs_m','obj_reload_max_abs_m','local_surface_accepted']}
 write(a.out/'metrics.json',dict(body_retained_definition='Historical 40 px subset held fixed across all methods; no-margin fitting uses body_all',previous_original_metrics_verified=True,reference='automatic 2D and derived stereo 3D, not ground truth',frames=records))
if __name__=='__main__':main()
