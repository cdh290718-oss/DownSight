"""Export unchanged fitted meshes and measure radial pixel RMSE to each detector."""
import sys,json,zipfile,hashlib,math
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];RUN=Path(__file__).resolve().parent;OUT=ROOT/'outputs/real001_obj_rmse_20261118';PRIOR=ROOT/'runs/2026-11-18_real001_golden_d_four_models';OLD=ROOT/'outputs/real001_golden_d_four_models_20261118'
sys.path.insert(0,str(PRIOR))
from common import setup_model,sha,read,write
model,Camera=setup_model();torch.set_num_threads(2)
from egosmplx_pipeline import fusion
fusion.initialize(model);layer=model.layer['neutral'].cpu().eval();layer.requires_grad_(False);faces=np.asarray(model.face,dtype=np.int64)
records=read(OLD/'records.json');prepared=read(OLD/'prepared.json')['items'];pred=read(OLD/'detections.json');exports=read(PRIOR/'evidence/exports.json');T=np.asarray(prepared[0]['inputs']['views']['right']['camera']['left_to_view']);bundle=OUT/'bundle';bundle.mkdir(exist_ok=True);entries=[];allpoints={};summary={};checks=[]
assert [r['camera'] for r in records]==['cam3','cam4'];cameras=[Camera(r['row']['calibration']) for r in records];mask0=np.asarray(prepared[0]['points'][0]['valid'])
def write_obj(p,v):
 with p.open('w') as f:
  f.write('# real001 t=29.0s; neutral SMPL-X fitted mesh; units meters; CV camera coordinates\n')
  np.savetxt(f,v,fmt='v %.9f %.9f %.9f');np.savetxt(f,faces+1,fmt='f %d %d %d')
 rv=[];rf=[]
 for line in p.read_text().splitlines():
  x=line.split()
  if x and x[0]=='v':rv.append([float(q) for q in x[1:]])
  if x and x[0]=='f':rf.append([int(q)-1 for q in x[1:]])
 rv=np.asarray(rv);rf=np.asarray(rf);assert np.array_equal(rf,faces);delta=float(np.max(np.abs(rv-v)));assert delta<1e-8;assert rf.min()>=0 and rf.max()<len(rv)
 return rv,delta
for name in ['original','rw500','rw5000','rw20000']:
 entry=next(e for e in exports['files'] if e['group']=='01_new_four_models' and '__'+name+'__' in e['file']);path=OLD/'download_bundle'/entry['relative_path'];assert sha(path)==entry['sha256'];p=dict(np.load(path,allow_pickle=False))
 v,j=fusion.geometry(p)
 with torch.no_grad():native=layer(**{k:torch.tensor(p[k],dtype=torch.float32).reshape(1,-1) for k in fusion.PARAMS}).joints[0].numpy()+p['transl'].reshape(3)
 assert np.isfinite(v).all() and np.isfinite(native).all();item=next(i for i in prepared if i['detector']==name);assert [q['valid'] for q in item['points']]==[q['valid'] for q in prepared[0]['points']]
 points=[]
 for view,(rec,cam) in enumerate(zip(records,cameras)):
  camera=rec['camera'];vv=v.astype(np.float64);jj=native.astype(np.float64)
  if view==1:vv=vv@T[:3,:3].T+T[:3,3];jj=jj@T[:3,:3].T+T[:3,3]
  folder=bundle/(camera+'_cv');folder.mkdir(exist_ok=True);filename='real001_t029p0__goldenD2d__'+name+'__'+camera+'_cv.obj';file=folder/filename;loaded,err=write_obj(file,vv)
  entries.append(dict(file=filename,relative_path=str(file.relative_to(bundle)),model=name,coordinate_frame=camera+' CV',units='meters',vertices=len(vv),faces=len(faces),sha256=sha(file),bytes=file.stat().st_size,parameter_source_sha256=sha(path),obj_reload_max_abs_m=err))
  checks.append(dict(model=name,camera=camera,obj_reload_max_abs_m=err,topology_exact=True))
  for k,q in enumerate(item['points']):
   target=np.asarray(q['xy'][view],dtype=np.float64);det=pred[name][view]['points'][k];assert q['name']==det['name'] and np.array_equal(target,np.asarray(det['xy']))
   projected=cam.world2camera(jj[q['native_index']][None])[0];assert np.isfinite(projected).all() and np.isfinite(target).all();res=projected-target
   points.append(dict(camera=camera,joint=q['name'],fit_mask=bool(q['valid'][view]),automatic_target_in_image=bool(det['in_image']),dx_px=float(res[0]),dy_px=float(res[1]),squared_distance_px2=float(res@res),distance_px=float(np.linalg.norm(res))))
 def metric(rows):
  n=len(rows);sq=sum(q['squared_distance_px2'] for q in rows)
  return dict(points=n,sum_squared_distance_px2=sq,radial_rmse_px=math.sqrt(sq/n),coordinate_rmse_px=math.sqrt(sq/(2*n)))
 summary[name]={}
 for scope in ['common_fit_mask','all_12_joints_per_view']:
  selected=[q for q in points if scope=='all_12_joints_per_view' or q['fit_mask']];summary[name][scope]={camera:metric([q for q in selected if q['camera']==camera]) for camera in ['cam3','cam4']};summary[name][scope]['stereo']=metric(selected)
  a=summary[name][scope];assert abs(a['stereo']['sum_squared_distance_px2']-sum(a[c]['sum_squared_distance_px2'] for c in ['cam3','cam4']))<1e-7
 allpoints[name]=points
write(RUN/'evidence/point_residuals.json',allpoints);write(RUN/'evidence/rmse.json',dict(command='PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. <EGOSMPLX_ENV>/bin/python export_evaluate.py',exit_status=0,definition='radial_rmse_px = sqrt(sum_i((u_projected-u_auto)^2 + (v_projected-v_auto)^2)/N)',coordinate_definition='coordinate_rmse_px = radial_rmse_px/sqrt(2), provided to disambiguate convention',primary_scope='all 12 named finite automatic targets in each camera, including omitted/out-of-image shoulder predictions; 24 pooled',supplemental_scope='common_fit_mask, 11 points in each camera, 22 joint-view observations pooled',targets='Each model compared with its own detector predictions, not ground truth or a common independent target',source_detections_sha256=sha(OLD/'detections.json'),source_prepared_sha256=sha(OLD/'prepared.json'),summary=summary))
write(bundle/'camera_transform.json',dict(source_frame='cam3 CV',target_frame='cam4 CV',units='meters',cam4_from_cam3=T.tolist(),formula='x_cam4=R @ x_cam3+t'))
for rec in records:write(bundle/(rec['camera']+'_calibration.json'),read(rec['row']['calibration']))
(bundle/'README.md').write_text('''# Four fitted SMPL-X meshes: real001, t=29.0s

cam3_cv contains the four primary meshes (original, rw500, rw5000, rw20000 detectors).
cam4_cv contains the SAME four fitted bodies transformed to cam4; these are not additional fits.

OBJ files contain vertices and 1-based triangular faces, meters, camera CV convention (+X right, +Y down, +Z forward). No texture, skinning rig or SMPL-X pose/shape coefficients are stored in OBJ. Use the preceding SMPL-X parameter Release to edit pose or shape. Do not overlay both camera-coordinate versions in one scene without converting frames. Do not center, rescale or flip axes if using supplied camera calibration. Viewers may use different up-axis conventions.

These meshes preserve the preceding full-2D golden D fits without re-optimization. Fixed local hand pose, no MANO hand replacement or seam fusion. Four 20000-step fits did not pass the preceding stability diagnostic; no 3D ground truth is available. RMSE in the report compares projected model joints to each detector's automatic body targets, not a mesh nearest-vertex distance or true pose error.

Original parameter bundle:
https://github.com/cdh290718-oss/DownSight/releases/tag/real001-goldend2d-four-models-20261118
''')
write(bundle/'manifest.json',{'files':entries,'checks':checks});archive=OUT/'real001_t029p0_four_models_OBJ_cam3_cam4.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
 for p in sorted(bundle.rglob('*')):
  if p.is_file():z.write(p,str(p.relative_to(bundle)))
write(RUN/'evidence/exports.json',dict(files=entries,checks=checks,archive=dict(file=archive.name,bytes=archive.stat().st_size,sha256=sha(archive)),fit_unchanged=True,mano_fusion=False,command='export_evaluate.py',exit_status=0))
for name,v in summary.items():print(name,'common',[(k,round(x['radial_rmse_px'],3)) for k,x in v['common_fit_mask'].items()],'all',[(k,round(x['radial_rmse_px'],3)) for k,x in v['all_12_joints_per_view'].items()],flush=True)
print('DONE: eight OBJ coordinate versions of four fitted bodies',flush=True)
