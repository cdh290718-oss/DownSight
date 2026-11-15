import json,os,sys,importlib.util,hashlib
from pathlib import Path
import numpy as np,cv2
ROOT=Path(__file__).resolve().parents[2];RUN=Path(__file__).resolve().parent;OUT=ROOT/'outputs/gt2d_vs_sapiens_20261115';ARCH=ROOT.parent/'EgoFishPose/experiments/benchmark_body_hand10_head01_20261014_v1';read=lambda p:json.loads(Path(p).read_text());local=read(OUT/'observation_local.json');gt=np.load(ARCH/'body_gt_official.npz');cal=[read(ARCH/'body_calibration_fisheye'/('camera_'+s+'.json'))['input_adapter'] for s in ['left','right']];ids={'left_shoulder':4,'right_shoulder':1,'left_elbow':5,'right_elbow':2,'left_wrist':6,'right_wrist':3,'left_hip':11,'right_hip':7,'left_knee':12,'right_knee':8,'left_ankle':13,'right_ankle':9};(RUN/'assets').mkdir(exist_ok=True);results=[]
for p in local['pairs']:
 pair=p['pair'];idx=(int(pair[-2:])-1)*2; target=gt['joints_m'][idx].astype(float);ref_tri={z['index']:z for z in read(ROOT/'outputs/unrealego_gt2d_ideal_20261115'/pair/'triangulated.json')['points']};items=[]
 for q in p['points']:
  e={k:v for k,v in q.items() if k not in ['automatic_geometry']};geo=q.get('automatic_geometry',{});gtxyz=target[ids[q['name']]]
  e['automatic_geometry_reason']=geo.get('reason'); e['automatic_ray_gap_m']=geo.get('ray_gap_m');e['automatic_ray_distances_m']=geo.get('ray_distances_m');e['automatic_parallax_deg']=geo.get('parallax_deg');e['automatic_std_principal_m']=geo.get('std_principal_m');e['automatic_reprojection_residual_px']=geo.get('reprojection_residual_px')
  if 'point_cam3_m' in geo:
   distance=float(np.linalg.norm(np.array(geo['point_cam3_m'])-gtxyz)*1000);e['auto_tri_vs_official_3d_mm']=distance;e['automatic_ray_gap_m']=geo['ray_gap_m'];e['automatic_parallax_deg']=geo['parallax_deg']
  z=ref_tri[q['index']]
  if z['accepted']:e['official_tri_vs_official_3d_mm']=float(np.linalg.norm(np.array(z['xyz'])-gtxyz)*1000)
  items.append(e)
 for i,side in enumerate(['left','right']):
  im=cv2.imread(str(ARCH/'body_input'/(pair+'_'+side+'.png')));dx=read(ARCH/'body_work_full_image'/(pair+'_'+side)/'observations'/(pair+'_'+side)/'sapiens/keypoints.json');lookup={z['name']:z for z in dx['keypoints']};xy=gt['joints2d'][idx+i].astype(float)*cal[i]['scale']+np.array(cal[i]['offset'])
  for q in p['points']:
   n=q['name'];g=xy[ids[n]];a=lookup.get(n)
   if a and np.isfinite(g).all() and np.isfinite([a['x'],a['y']]).all():
    av=(int(round(a['x'])),int(round(a['y'])));gv=tuple(np.rint(g).astype(int));cv2.line(im,gv,av,(0,180,255),1,cv2.LINE_AA);cv2.circle(im,gv,4,(0,255,0),-1);cv2.circle(im,av,3,(0,0,255),-1);cv2.putText(im,n.replace('left_','L_').replace('right_','R_'),(gv[0]+4,gv[1]-4),cv2.FONT_HERSHEY_SIMPLEX,.28,(0,255,255),1,cv2.LINE_AA)
  cv2.imwrite(str(RUN/'assets'/(pair+'-'+side+'-points.jpg')),im,[cv2.IMWRITE_JPEG_QUALITY,90])
 results.append({'pair':pair,'matched_count':p['shared_target_count'],'points':items})
out={'command':'<BODY_PYTHON> runs/2026-11-15_gt2d-vs-sapiens/analyze_observations.py','exit_status':0,'coordinate_units':'2D distance in 1280x720 adapted pixels; 3D distance in mm in left camera; no 3D alignment','visibility':'in-frame is geometric only; occlusion unknown, no manual labels; model/GT landmark definition can differ','automatic_scope':'each independently detected point whether or not confidence threshold met; matched_accepted means fit eligible','pairs':results};(RUN/'evidence/observations.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print('saved',len(results))
