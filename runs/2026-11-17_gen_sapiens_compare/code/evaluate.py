import json,sys,hashlib,time
from pathlib import Path
import cv2,numpy as np,torch
from safetensors.torch import load_file
W=Path('..').resolve();root=W/'gen_egosmplx_run';back=W/'EgoSMPLX';repo=back/'official_baseline/sapiens2';sys.path[:0]=[str(back/'.runtime/sapiens_deps'),str(repo)]
from sapiens.engine.config import Config
from sapiens.engine.datasets import Compose
from sapiens.registry import MODELS
from sapiens.pose.datasets import parse_pose_metainfo
from sapiens.pose.datasets.codecs.udp_heatmap import UDPHeatmap
c=Config.fromfile(repo/'sapiens/pose/configs/keypoints308/shutterstock_goliath_3po/sapiens2_1b_keypoints308_shutterstock_goliath_3po-1024x768.py');c.model['backbone'].pop('init_cfg',None);model=MODELS.build(c.model);base=back/'pretrained_models/external_pose/sapiens2_1b_pose.safetensors';model.load_state_dict(load_file(str(base)),strict=True);model.cuda().eval();pp=MODELS.build(c.data_preprocessor).cuda();pipeline=Compose(c.test_pipeline);meta=parse_pose_metainfo(dict(from_file=str(repo/'sapiens/pose/configs/_base_/keypoints308.py')));codec=UDPHeatmap(**{k:v for k,v in c.codec.items() if k!='type'});dfm=np.load(root/'dfm_reference_virtual.npz');frames=json.loads((root/'frames.json').read_text())['frames'];assert len(frames)==40
mapping=[('left_shoulder',5,16),('right_shoulder',6,17),('left_elbow',7,18),('right_elbow',8,19),('left_wrist',62,20),('right_wrist',41,21),('left_hip',9,1),('right_hip',10,2),('left_knee',11,4),('right_knee',12,5),('left_ankle',13,7),('right_ankle',14,8)];out=Path('outputs/gen_sapiens_compare_20261117');res=[]
for version in ['original','full_epoch1']:
 if version!='original':model.load_state_dict(load_file('outputs/fullmodel_training_20261117/results/adapter_01.safetensors'),strict=False)
 for row in frames:
  image=root/row['image'];im=cv2.imread(str(image));assert im is not None;h,w=im.shape[:2];d=pp(pipeline(dict(img=im,bbox=np.array([[0,0,w,h]],np.float32),bbox_score=np.ones(1,np.float32))))
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
   hm=model(d['inputs']);hm=(hm+model(d['inputs'].flip(-1)).flip(-1)[:,meta['flip_indices']])/2 if c.val_cfg.get('flip_test',False) else hm
  xy,score=codec.decode(hm.float().cpu().numpy()[0]);m=d['data_samples']['meta'];conv=lambda v:v.detach().cpu().numpy() if torch.is_tensor(v) else np.array(v);xy=(xy/conv(m['input_size'])*conv(m['bbox_scale'])+conv(m['bbox_center'])-.5*conv(m['bbox_scale']))[0]
  fr=int(row['id'].split('_')[1]);idx=int(np.where(dfm['frames']==fr)[0][0]);gt=dfm[row['camera']+'_body_uv'][idx];vals=[]
  for name,j,k in mapping:
   p=xy[j];q=gt[k];valid=bool(np.isfinite(q).all() and 0<=q[0]<w and 0<=q[1]<h);vals.append(dict(name=name,prediction=p.tolist(),reference=q.tolist(),reference_in_image=valid,error_px=float(np.linalg.norm(p-q)) if valid else None,score=float(score[0,j]),prediction_in_image=bool(0<=p[0]<w and 0<=p[1]<h)))
  res.append(dict(version=version,id=row['id'],image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),points=vals));print(version,row['id'],flush=True)
 (out/(version+'.json')).write_text(json.dumps([r for r in res if r['version']==version],indent=2))
summary={}
for version in ['original','full_epoch1']:
 v=[r for r in res if r['version']==version];errs=[x['error_px'] for r in v for x in r['points'] if x['error_px'] is not None];summary[version]=dict(images=len(v),reference_in_image_points=len(errs),mean_px=float(np.mean(errs)),correct_20px=sum(x<=20 for x in errs),per_joint={name:dict(count=len(e),mean_px=float(np.mean(e)) if e else None,correct_20px=sum(x<=20 for x in e)) for name,_,_ in mapping for e in [[x['error_px'] for r in v for x in r['points'] if x['name']==name and x['error_px'] is not None]]})
Path('runs/2026-11-17_gen_sapiens_compare/evidence/summary.json').write_text(json.dumps(dict(command='evaluate.py; original and epoch 1, official full image bbox/flip',exit_status=0,requested=100,available=40,reference='DFM projected body is unvalidated model reference, not independent GT',summary=summary),indent=2)+'\n');print(json.dumps(summary),flush=True)
