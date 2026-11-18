"""Same full-image Sapiens2 inference as the cross-dataset experiment; no labels."""
import sys,json,time
from pathlib import Path
import cv2,numpy as np,torch
from safetensors.torch import load_file
from common import ROOT,RUN,OUT,read,write,sha
back=ROOT.parent/'EgoSMPLX';repo=back/'official_baseline/sapiens2';sys.path[:0]=[str(back/'.runtime/sapiens_deps'),str(repo)]
from sapiens.engine.config import Config
from sapiens.engine.datasets import Compose
from sapiens.registry import MODELS
from sapiens.pose.datasets import parse_pose_metainfo
from sapiens.pose.datasets.codecs.udp_heatmap import UDPHeatmap
records=[r for r in read(ROOT/'outputs/session_hand6_v2_no_margin_rerun_20261104/prepared/inputs.json')['records'] if r['pair']=='new_001'];assert [r['camera'] for r in records]==['cam3','cam4'];write(OUT/'records.json',records)
config=repo/'sapiens/pose/configs/keypoints308/shutterstock_goliath_3po/sapiens2_1b_keypoints308_shutterstock_goliath_3po-1024x768.py';c=Config.fromfile(config);c.model['backbone'].pop('init_cfg',None);model=MODELS.build(c.model).cuda().eval();pp=MODELS.build(c.data_preprocessor).cuda();pipeline=Compose(c.test_pipeline);meta=parse_pose_metainfo(dict(from_file=str(repo/'sapiens/pose/configs/_base_/keypoints308.py')));codec=UDPHeatmap(**{k:v for k,v in c.codec.items() if k!='type'})
base=back/'pretrained_models/external_pose/sapiens2_1b_pose.safetensors';weights={'original':base,'rw500':ROOT/'outputs/fullmodel_training_20261117/results/adapter_01.safetensors','rw5000':ROOT/'outputs/rw_a5000_training_20261117T122117126742Z/results/adapter_01.safetensors','rw20000':ROOT/'outputs/rw_a20000_training_20261117T152700444455Z/results/adapter_01.safetensors'};expected=read(ROOT/'runs/2026-11-18_egowholebody100/evidence/weights.json');mapping=[('left_shoulder',5,16,8),('right_shoulder',6,17,9),('left_elbow',7,18,10),('right_elbow',8,19,11),('left_wrist',62,20,12),('right_wrist',41,21,13),('left_hip',9,1,1),('right_hip',10,2,2),('left_knee',11,4,3),('right_knee',12,5,4),('left_ankle',13,7,5),('right_ankle',14,8,6)];results={};hashes={};start=time.time()
for name,path in weights.items():
 hashes[name]=sha(path);assert hashes[name]==expected[name]['sha256'];model.load_state_dict(load_file(str(base)),strict=True)
 if name!='original':
  state=load_file(str(path));assert all(n in state for n,_ in model.named_parameters());ret=model.load_state_dict(state,strict=False);assert not ret.unexpected_keys;del state
 results[name]=[]
 for rec in records:
  im=cv2.imread(rec['row']['image']);assert im is not None;h,w=im.shape[:2];d=pp(pipeline(dict(img=im,bbox=np.array([[0,0,w,h]],np.float32),bbox_score=np.ones(1,np.float32))))
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
   hm=model(d['inputs']);hm=(hm+model(d['inputs'].flip(-1)).flip(-1)[:,meta['flip_indices']])/2 if c.val_cfg.get('flip_test',False) else hm
  xy,scores=codec.decode(hm.float().cpu().numpy()[0]);m=d['data_samples']['meta'];conv=lambda v:v.detach().cpu().numpy() if torch.is_tensor(v) else np.array(v);xy=(xy/conv(m['input_size'])*conv(m['bbox_scale'])+conv(m['bbox_center'])-.5*conv(m['bbox_scale']))[0];assert np.isfinite(xy).all()
  points=[dict(name=n,sapiens_index=j,native_index=nid,index=mid,xy=xy[j].tolist(),score=float(scores[0,j]),in_image=bool(0<=xy[j,0]<w and 0<=xy[j,1]<h)) for n,j,nid,mid in mapping]
  results[name].append(dict(camera=rec['camera'],image_sha256=sha(rec['row']['image']),width=w,height=h,points=points));print(name,rec['camera'],'ok',flush=True)
write(OUT/'detections.json',results);write(RUN/'evidence/detection.json',dict(command='PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 <PYTHON_ENV>/bin/python detect.py',exit_status=0,elapsed_seconds=time.time()-start,weights_sha256=hashes,config_sha256=sha(config),images=[{'camera':r['camera'],'sha256':sha(r['row']['image'])} for r in records],mapping=mapping,failed_images=0,precision='BF16 autocast',flip_test=bool(c.val_cfg.get('flip_test',False))))
