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
c=Config.fromfile(repo/'sapiens/pose/configs/keypoints308/shutterstock_goliath_3po/sapiens2_1b_keypoints308_shutterstock_goliath_3po-1024x768.py');c.model['backbone'].pop('init_cfg',None);model=MODELS.build(c.model);base=back/'pretrained_models/external_pose/sapiens2_1b_pose.safetensors';model.load_state_dict(load_file(str(base)),strict=True);model.cuda().eval();pp=MODELS.build(c.data_preprocessor).cuda();pipeline=Compose(c.test_pipeline);meta=parse_pose_metainfo(dict(from_file=str(repo/'sapiens/pose/configs/_base_/keypoints308.py')));codec=UDPHeatmap(**{k:v for k,v in c.codec.items() if k!='type'});frames=json.loads(Path('outputs/egowholebody100_20261118/manifest.json').read_text());assert len(frames)==100
mapping=[('left_shoulder',5,16),('right_shoulder',6,17),('left_elbow',7,18),('right_elbow',8,19),('left_wrist',62,20),('right_wrist',41,21),('left_hip',9,1),('right_hip',10,2),('left_knee',11,4),('right_knee',12,5),('left_ankle',13,7),('right_ankle',14,8)];out=Path('outputs/egowholebody100_20261118');res=[]
for version in ['original','rw500','rw5000','rw20000']:
 model.load_state_dict(load_file(str(base)),strict=True)
 weights={'rw500':'outputs/fullmodel_training_20261117/results/adapter_01.safetensors','rw5000':'outputs/rw_a5000_training_20261117T122117126742Z/results/adapter_01.safetensors','rw20000':'outputs/rw_a20000_training_20261117T152700444455Z/results/adapter_01.safetensors'}
 if version!='original':
  state=load_file(weights[version]);assert all(n in state for n,_ in model.named_parameters());check=model.load_state_dict(state,strict=False);assert not check.unexpected_keys;del state
 for row in frames:
  image=Path(row['image']);im=cv2.imread(str(image));assert im is not None;h,w=im.shape[:2];d=pp(pipeline(dict(img=im,bbox=np.array([[0,0,w,h]],np.float32),bbox_score=np.ones(1,np.float32))))
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
   hm=model(d['inputs']);hm=(hm+model(d['inputs'].flip(-1)).flip(-1)[:,meta['flip_indices']])/2 if c.val_cfg.get('flip_test',False) else hm
  xy,score=codec.decode(hm.float().cpu().numpy()[0]);m=d['data_samples']['meta'];conv=lambda v:v.detach().cpu().numpy() if torch.is_tensor(v) else np.array(v);xy=(xy/conv(m['input_size'])*conv(m['bbox_scale'])+conv(m['bbox_center'])-.5*conv(m['bbox_scale']))[0]
  vals=[]
  for point in row['points']:
   name=point['name'];j=point['sapiens_index'];p=xy[j];q=np.array(point['reference']);valid=point['reference_in_image'];vals.append(dict(name=name,error_px=float(np.linalg.norm(p-q)) if valid else None,reference_in_image=valid,score=float(score[0,j]),prediction_in_image=bool(0<=p[0]<w and 0<=p[1]<h)))
  res.append(dict(version=version,id=row['id'],image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),points=vals));print(version,row['id'],flush=True)
 (out/(version+'.json')).write_text(json.dumps([r for r in res if r['version']==version],indent=2))
summary={}
for version in ['original','rw500','rw5000','rw20000']:
 v=[r for r in res if r['version']==version];errs=[x['error_px'] for r in v for x in r['points'] if x['error_px'] is not None];summary[version]=dict(images=len(v),reference_in_image_points=len(errs),mean_px=float(np.mean(errs)),correct_20px=sum(x<=20 for x in errs),per_joint={name:dict(count=len(e),mean_px=float(np.mean(e)) if e else None,correct_20px=sum(x<=20 for x in e)) for name,_,_ in mapping for e in [[x['error_px'] for r in v for x in r['points'] if x['name']==name and x['error_px'] is not None]]})
Path('runs/2026-11-18_egowholebody100/evidence/summary.json').write_text(json.dumps(dict(command='evaluate.py; official full-image bbox and configured flip inference',exit_status=0,images=100,reference='native render skeleton projected with 01_12; single subject convenience sample',summary=summary),indent=2)+'\n');print(json.dumps(summary),flush=True)
