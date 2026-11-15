import json,sys,os,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];RUN=Path(__file__).resolve().parent;OUT=ROOT/'outputs/gt2d_vs_sapiens_20261115';BASE=ROOT.parent
sys.path.insert(0,str(ROOT/'runs/2026-11-04_session-hand6-v2/code'))
from prepare import triangulate,Camera
read=lambda p:json.loads(Path(p).read_text());write=lambda p,x:Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
arch=BASE/'EgoFishPose/experiments/benchmark_body_hand10_head01_20261014_v1';samples=read(arch/'body_samples.json')['samples'];ds=BASE/'EgoSMPLX/data/UnrealEgo_RW_Test/prepared/samples.npz';data=np.load(ds);rig=read(ROOT/'runs/2026-11-15_unrealego-gt2d/evidence/preparation_ideal.json');R=np.array(rig['rig_rotation']);t=np.array(rig['rig_translation_m']);paths=read(ROOT/'outputs/initialization_ablation_20261109/model_paths.json');cams=[Camera(arch/'body_calibration_fisheye'/('camera_'+s+'.json')) for s in ['left','right']];cal=[read(arch/'body_calibration_fisheye'/('camera_'+s+'.json'))['input_adapter'] for s in ['left','right']];names=['left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist','left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle'];ids=[8,9,10,11,12,13,1,2,3,4,5,6];gtidx=[4,1,5,2,6,3,11,7,12,8,13,9];views={'left':{'camera':{'left_to_view':np.eye(4).tolist()}},'right':{'camera':{'left_to_view':np.block([[R,t[:,None]],[np.zeros((1,3)),np.ones((1,1))]]).tolist()}}};allrows=[]
def ideal(q):
 if 'covariance_cam3_m2' not in q:return None,'geometry_gate'
 C=(np.array(q['covariance_cam3_m2'])-np.eye(3)*.008**2)/64+np.eye(3)*.008**2; std=np.sqrt(np.maximum(np.linalg.eigvalsh(C),0));E=np.linalg.norm(np.array(q['reprojection_residual_px']),axis=1);xyz=np.array(q['point_cam3_m']);ok=bool(np.isfinite(C).all() and std.max()<=.5 and E.max()<=20 and xyz[2]>=.02 and (R@xyz+t)[2]>=.02)
 return (C if ok else None),('accepted' if ok else 'reprojection_depth_or_uncertainty')
for pair in range(1,5):
 name='body%02d'%pair;rec=samples[2*(pair-1):2*pair];uv=[];det=[]
 for i,r in enumerate(rec):
  xy=data['keypoints2d'][r['sample_index']].astype(float);xy[:,:2]=xy[:,:2]*cal[i]['scale']+np.array(cal[i]['offset']);uv.append(xy);f=arch/'body_work_full_image'/r['id']/'observations'/r['id']/'sapiens/keypoints.json';det.append({q['name']:q for q in read(f)['keypoints']})
 old=ROOT/'outputs/unrealego_gt2d_ideal_20261115'/name;oldgt={p['index']:p for p in read(old/'triangulated.json')['points']};allpoints=[]; auto=[];control=[]
 for n,idx,gi in zip(names,ids,gtidx):
  d=[det[i].get(n) for i in range(2)];g=np.array([uv[i][gi,:2] for i in range(2)]);item={'name':n,'index':idx,'views':[]}
  for i in range(2):
   z=d[i];ginside=bool(280<=g[i,0]<1000 and 0<=g[i,1]<720);entry={'official_in_image':ginside,'visibility':'unknown: no verified visibility/occlusion label','definition':'official joint versus Sapiens surface landmark; semantic equivalence not established','detected':z is not None}
   if z:
    xy=np.array([z['x'],z['y']]);entry.update(score=float(z['score']),automatic_in_image=bool(280<=xy[0]<1000 and 0<=xy[1]<720),distance_px=float(np.linalg.norm(xy-g[i])))
   item['views'].append(entry)
  if all(d):
   xy=np.array([[z['x'],z['y']] for z in d]);auto_in=all(z['automatic_in_image'] for z in item['views']);score_ok=all(z['score']>=.3 for z in item['views']);q=triangulate(cams,R,t,xy) if auto_in else {'accepted':False,'reason':'outside_image'};C,why=ideal(q); reason=('outside_image' if not auto_in else 'score_below_0.3' if not score_ok else why);accepted=bool(auto_in and score_ok and C is not None and oldgt[idx]['accepted']);item.update(automatic_geometry=q,automatic_gate=reason,official_gate='accepted' if oldgt[idx]['accepted'] else 'rejected',matched_accepted=accepted)
   if accepted:
    kind=n.split('_')[-1];sig={'shoulder':.03,'hip':.03,'elbow':.02,'knee':.02,'ankle':.015,'wrist':.008}[kind];grp='shoulder' if kind=='shoulder' else 'wrist' if kind=='wrist' else 'body';common={'name':n,'index':idx,'source':'sapiens','group':grp,'accepted':True,'covariance':(C+np.eye(3)*sig**2).tolist()};auto.append(dict(common,xyz=q['point_cam3_m']));control.append(dict(common,xyz=oldgt[idx]['xyz']))
  else:item.update(automatic_gate='missing_detection',matched_accepted=False)
  allpoints.append(item)
 for mode,points in [('auto',auto),('official',control)]:
  dest=OUT/(name+'_'+mode);dest.mkdir(exist_ok=False);(dest/'anchors').mkdir();(dest/'runtime/cache').mkdir(parents=True);(dest/'runtime/matplotlib').mkdir();write(dest/'inputs.json',{'views':views});write(dest/'model_paths.json',paths);write(dest/'triangulated.json',{'points':points});write(dest/'anchors/oracle.json',read(old/'anchors/oracle.json'))
 allrows.append({'pair':name,'sample_indices':[r['sample_index'] for r in rec],'shared_point_names':[a['name'] for a in auto],'shared_target_count':len(auto),'points':allpoints});print(name,'matched',len(auto),flush=True)
# Raw 2D GT and 3D triangulated coordinates remain local, not public.
write(OUT/'observation_local.json',{'pairs':allrows});write(RUN/'evidence/preparation.json',{'command':'<BODY_PYTHON> runs/2026-11-15_gt2d-vs-sapiens/prepare.py','exit_status':0,'dataset_sha256':hashlib.sha256(ds.read_bytes()).hexdigest(),'calibration':'other S9 sequence official annotations; not independent device calibration','two_conditions':'same matched indices and identical covariance from auto triangulation; coordinates differ','sigma_assumption':'1px isotropic 2D + 8mm metric floor plus same definition noise; optimistic for Sapiens','eligibility':'Sapiens both views >=0.3 and in image; triangulation gates; official matched points pass','visibility':'unknown, no independent verified visibility annotation','pairs':[{'pair':x['pair'],'matched':x['shared_target_count'],'names':x['shared_point_names']} for x in allrows]})
