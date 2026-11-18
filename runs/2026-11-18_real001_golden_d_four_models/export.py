"""Export compact SMPL-X coefficients, verify reload, and render calibrated mesh overlays."""
import importlib.util,zipfile,shutil,json
from pathlib import Path
import cv2,numpy as np,torch
from common import ROOT,RUN,OUT,read,write,sha,setup_model
from geometry import BodyKinematics
model,Camera=setup_model();torch.set_num_threads(2)
from egosmplx_pipeline import fusion
fusion.initialize(model);layer=model.layer['neutral'].cpu().eval();layer.requires_grad_(False)
records=read(OUT/'records.json');done=read(OUT/'fit_complete.json');assert done['exit_status']==0 and done['steps']==20000
z=np.load(OUT/'budget_20000.npz');assert (z['selected_step']>=0).all();prepared=read(OUT/'prepared.json')['items'];fk=BodyKinematics(layer)
bundle=OUT/'download_bundle';bundle.mkdir(exist_ok=True);assets=RUN/'assets';assets.mkdir(exist_ok=True);entries=[];checks=[];renders=[];transform=np.array(prepared[0]['inputs']['views']['right']['camera']['left_to_view'])
PARAMS=fusion.PARAMS+['transl'];lens={'betas':10,'jaw_pose':3,'leye_pose':3,'reye_pose':3,'expression':10,'body_pose':63,'global_orient':3,'left_hand_pose':45,'right_hand_pose':45,'transl':3}
flat=bool(getattr(layer,'flat_hand_mean',False));pca=bool(getattr(layer,'use_pca',False));assert not pca

def export_params(p,stem,group,coordinate,source_sha,description):
 folder=bundle/group;folder.mkdir(exist_ok=True);arrays={k:np.asarray(p[k]).reshape(1,lens[k]) for k in PARAMS}
 assert all(np.isfinite(v).all() for v in arrays.values())
 arrays.update(coordinate_frame=np.array(coordinate),gender=np.array('neutral'),units=np.array('meters'),pose_representation=np.array('axis_angle_radians'),flat_hand_mean=np.array(flat),use_pca=np.array(False))
 npz=folder/(stem+'.npz');np.savez_compressed(npz,**arrays)
 write(folder/(stem+'.json'),dict(description=description,coordinate_frame=coordinate,gender='neutral',units='meters',pose_representation='axis_angle_radians',flat_hand_mean=flat,use_pca=False,parameters={k:arrays[k].tolist() for k in PARAMS}))
 reloaded=dict(np.load(npz,allow_pickle=False));v,j=fusion.geometry(reloaded);v0,j0=fusion.geometry(p);err=float(np.max(np.abs(v-v0)));assert err<1e-5
 entries.append(dict(file=npz.name,relative_path=str(npz.relative_to(bundle)),group=group,description=description,coordinate_frame=coordinate,sha256=sha(npz),bytes=npz.stat().st_size,source_sha256=source_sha,reload_mesh_max_abs_m=err))
 return v,j

spec=importlib.util.spec_from_file_location('real001_rasterizer',ROOT/'runs/2026-11-04_mesh-projection-review/code/rasterizer.py');raster=importlib.util.module_from_spec(spec);spec.loader.exec_module(raster)
faces=np.asarray(model.face,np.int32);dominant=layer.lbs_weights.detach().numpy().argmax(1);head=(dominant==model.orig_joints_name.index('Head')).astype(np.float32)
views=[]
for rec in records:
 cam=Camera(rec['row']['calibration']);im=cv2.imread(rec['row']['image']);assert im is not None;yy,xx=np.indices(im.shape[:2]);rays=cam.camera2world_ray(np.stack([xx.ravel(),yy.ravel()],axis=1)).reshape(*im.shape[:2],3).astype(np.float32);views.append((rec,cam,im,rays))

def render_new(vertices,name):
 for rec,cam,im,rays in views:
  v=vertices.astype(float)
  if rec['camera']=='cam4':v=v@transform[:3,:3].T+transform[:3,3]
  uv=cam.world2camera(v).astype(np.float32);depth=np.full(im.shape[:2],np.inf,np.float32);colors=np.zeros_like(im);hp=np.zeros(im.shape[:2],np.float32)
  drawn,rejected=raster.raster(v.astype(np.float32),faces,head,uv,rays,depth,colors,hp,180.)
  valid=np.isfinite(depth);overlay=im.copy();overlay[valid]=(.45*im[valid]+.55*colors[valid]).astype(np.uint8)
  path=assets/(name+'__'+rec['camera']+'__mesh_overlay.jpg');assert cv2.imwrite(str(path),overlay,[cv2.IMWRITE_JPEG_QUALITY,94]);renders.append(dict(detector=name,camera=rec['camera'],file=path.name,sha256=sha(path),visible_pixels=int(valid.sum()),drawn_triangles=int(drawn),rejected_near_or_long=int(rejected),mesh_source='parameter-reloaded SMPL-X body, no hand replacement'))

for k,(pair,condition,method) in enumerate(done['ids']):
 assert method=='2d' and condition=='baseline';name=pair.replace('real001_','');t=lambda x:torch.tensor(z[x][k:k+1],dtype=torch.float32)
 with torch.no_grad():native,_=fk(t('body_pose'),t('betas'),t('global0'),t('delta'),t('translation'))
 R=cv2.Rodrigues(z['delta'][k].astype(float))[0]@cv2.Rodrigues(z['global0'][k].astype(float))[0]
 p={key:np.zeros((1,lens[key]),np.float32) for key in PARAMS};p.update(betas=z['betas'][k:k+1],body_pose=z['body_pose'][k].reshape(1,63),global_orient=cv2.Rodrigues(R)[0].reshape(1,3).astype(np.float32),transl=z['translation'][k:k+1])
 with torch.no_grad():full_native=layer(**{key:torch.tensor(p[key]).reshape(1,-1) for key in fusion.PARAMS}).joints[0,:22].numpy()+p['transl'].reshape(3)
 delta=float(np.max(np.abs(full_native-native[0].numpy())));assert delta<1e-5,delta
 stem='real001_t029p0__goldenD2d__'+name+'__shared_cam3_cv';v,j=export_params(p,stem,'01_new_four_models','cam3 CV',sha(OUT/'budget_20000.npz'),'Current full-2D golden D, detector '+name+', shared body for cam3/cam4')
 checks.append(dict(detector=name,selected_step=int(z['selected_step'][k]),full_smplx_vs_fk_max_abs_m=delta,shared_frame='cam3 CV'))
 render_new(v,name);print('Exported/rendered',name,flush=True)

for rec in records:
 for method in ['original','v2']:
  source=Path(rec['original']['body']) if method=='original' else Path(read(rec['config'])['output'])/'frames'/rec['id']/'fused/body_params.npz'
  p=dict(np.load(source,allow_pickle=False));stem='real001_t029p0__legacy_'+method+'__'+rec['camera']+'_cv'
  v,j=export_params(p,stem,'02_legacy_three_methods',rec['camera']+' CV',sha(source),'Archived '+method+' per-view SMPL-X body coefficients; not final MANO seam mesh')
  err=float(np.max(np.abs(v-p['smplx_mesh_cam'])));assert err<1e-5,err
  checks.append(dict(legacy=method,camera=rec['camera'],source_body_mesh_reload_max_abs_m=err))
  src=ROOT/'runs/2026-11-15_real-d-stereo/assets'/(rec['camera']+'-'+method+'.jpg');dest=assets/('legacy_'+method+'__'+rec['camera']+'__mesh_overlay.jpg');shutil.copyfile(src,dest)
  renders.append(dict(legacy=method,camera=rec['camera'],file=dest.name,sha256=sha(dest),historical_image_sha256=sha(src),mesh_source='Historical published image; contains historical hand-fusion geometry beyond SMPL-X coefficients'))
old=read(ROOT/'runs/2026-11-15_real-d-stereo/evidence/results.json')
for case,label in [('D','baseline'),('plus','head_tx_plus20mm_ry_plus3deg'),('minus','head_tx_minus20mm_ry_minus3deg')]:
 source=ROOT/'outputs/real_d_stereo_20261115'/(case+'_left')/'final.npz';assert sha(source)==old['cases'][case]['final_sha256'];p=dict(np.load(source,allow_pickle=False));group='02_legacy_three_methods' if case=='D' else '03_legacy_head_perturbations';stem='real001_t029p0__legacy_D3d__'+label+'__shared_cam3_cv'
 v,j=export_params(p,stem,group,'cam3 CV',sha(source),'Archived old 3D D: '+label+', shared cam3/cam4 body');err=float(np.max(np.abs(v-p['smplx_mesh_cam'])));assert err<1e-5
 checks.append(dict(legacy='D3d_'+label,source_hash_matches_old_report=True,source_body_mesh_reload_max_abs_m=err))
 for rec in records:
  src=ROOT/'runs/2026-11-15_real-d-stereo/assets'/(rec['camera']+'-'+case+'.jpg');dest=assets/('legacy_D3d_'+label+'__'+rec['camera']+'__mesh_overlay.jpg');shutil.copyfile(src,dest);renders.append(dict(legacy='D3d_'+label,camera=rec['camera'],file=dest.name,sha256=sha(dest),historical_image_sha256=sha(src),mesh_source='Historical published image, archived shared-body D fit'))

write(bundle/'camera_transform.json',dict(source_frame='cam3 CV',target_frame='cam4 CV',units='meters',formula='x_cam4 = R @ x_cam3 + t',cam4_from_cam3=transform.tolist()))
for rec in records:write(bundle/(rec['camera']+'_calibration.json'),read(rec['row']['calibration']))
readme='''# real001, t=29.0s: SMPL-X parameter downloads

01_new_four_models: current full-2D golden D, four Sapiens2-1B detectors.
02_legacy_three_methods: historical original and v2 (one body per camera), plus historical D3d baseline (one shared body).
03_legacy_head_perturbations: historical D3d with positive/negative Head-anchor perturbations.

All NPZ files contain only fitted parameters and small metadata; no SMPL-X licensed model weights, images or raw meshes. JSON companions contain the same coefficients. Body pose is (1,63), each hand pose (1,45), global orientation (1,3), translation (1,3); rotations are axis-angle in radians and translation is meters. Gender neutral, use_pca=False; flat_hand_mean is stored per file and must be respected. Zero local hand input does not imply a flat hand when flat_hand_mean=False.

Load NPZ with allow_pickle=False. Pass betas, body_pose, global_orient, left_hand_pose, right_hand_pose, jaw_pose, leye_pose, reye_pose, expression and transl to a matching SMPL-X layer. You must obtain the licensed SMPL-X model separately. Metadata keys are not model arguments.

shared_cam3_cv means one physical body is fitted to both images and saved in cam3 coordinates. To render in cam4, transform the regenerated mesh using camera_transform.json. Do NOT fit or add the two camera bodies together. Legacy original/v2 files are separate per-view fits and already use their named camera's coordinates.

Historical original/v2 overlays include MANO hand replacement/seam processing. SMPL-X coefficients reconstruct the SMPL-X body surface, not those extra hand mesh edits. Current golden D fixes local hand pose and does not use WiLoR observations. No MPJPE, PA-MPJPE or GT-based RMSE is claimed; this recording has no body GT. Camera and Head anchors retain the historical estimated/calibrated provenance and have not been independently validated against true body geometry.
'''
(bundle/'README.md').write_text(readme);write(bundle/'manifest.json',dict(files=entries,hand_convention=dict(use_pca=pca,flat_hand_mean=flat),parameter_only=True))
# Overview: same image size, transparency, color and camera projection across new models.
rows=[]
for name in ['original','rw500','rw5000','rw20000']:
 cells=[]
 for camera in ['cam3','cam4']:
  im=cv2.imread(str(assets/(name+'__'+camera+'__mesh_overlay.jpg')));im=cv2.resize(im,(640,360));canvas=np.full((400,640,3),245,np.uint8);canvas[40:]=im;cv2.putText(canvas,'golden D 2D | '+name+' | '+camera,(12,27),cv2.FONT_HERSHEY_SIMPLEX,.65,(30,30,30),1,cv2.LINE_AA);cells.append(canvas)
 rows.append(np.concatenate(cells,axis=1))
cv2.imwrite(str(assets/'new_four_models_overview.jpg'),np.concatenate(rows,axis=0),[cv2.IMWRITE_JPEG_QUALITY,94])
archive=OUT/'real001_t029p0_smplx_new4_and_legacy_all.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as dest:
 for file in sorted(bundle.rglob('*')):
  if file.is_file():dest.write(file,str(file.relative_to(bundle)))
write(RUN/'evidence/exports.json',dict(command='PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. <EGOSMPLX_ENV>/bin/python export.py',exit_status=0,files=entries,checks=checks,renders=renders,archive=dict(file=archive.name,sha256=sha(archive),bytes=archive.stat().st_size),hand_convention=dict(use_pca=pca,flat_hand_mean=flat),metrics='No ground-truth accuracy metrics computed; reload checks only'))
shutil.copyfile(bundle/'README.md',RUN/'DOWNLOADS_README.md');print('DONE:',len(entries),'parameter files',flush=True)
