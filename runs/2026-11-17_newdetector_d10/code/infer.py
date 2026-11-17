import sys,json,hashlib
from pathlib import Path
import cv2,numpy as np,torch
from safetensors.torch import load_file
root=Path.cwd();base=root.parent/'EgoSMPLX';archive=root.parent/'EgoFishPose/experiments/benchmark_body_hand10_head01_20261014_v1';repo=base/'official_baseline/sapiens2';sys.path[:0]=[str(base/'.runtime/sapiens_deps'),str(repo)]
from sapiens.engine.config import Config
from sapiens.engine.datasets import Compose
from sapiens.registry import MODELS
from sapiens.pose.datasets import parse_pose_metainfo
from sapiens.pose.datasets.codecs.udp_heatmap import UDPHeatmap
cfg=Config.fromfile(repo/'sapiens/pose/configs/keypoints308/shutterstock_goliath_3po/sapiens2_1b_keypoints308_shutterstock_goliath_3po-1024x768.py');cfg.model['backbone'].pop('init_cfg',None);model=MODELS.build(cfg.model);model.load_state_dict(load_file(str(base/'pretrained_models/external_pose/sapiens2_1b_pose.safetensors')),strict=True);model.load_state_dict(load_file(str(root/'outputs/fullmodel_training_20261117/results/adapter_01.safetensors')),strict=False);model.cuda().eval();pp=MODELS.build(cfg.data_preprocessor).cuda();pipeline=Compose(cfg.test_pipeline);meta=parse_pose_metainfo(dict(from_file=str(repo/'sapiens/pose/configs/_base_/keypoints308.py')));codec=UDPHeatmap(**{k:v for k,v in cfg.codec.items() if k!='type'});samples=json.loads((archive/'body_samples.json').read_text())['samples'];records=[]
for r in samples:
 path=archive/'body_original'/(r['id']+'.png');raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest()==r['image_sha256'];im=cv2.imdecode(np.frombuffer(raw,np.uint8),1);h,w=im.shape[:2];d=pp(pipeline(dict(img=im,bbox=np.array([[0,0,w,h]],np.float32),bbox_score=np.ones(1,np.float32))))
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
  hm=model(d['inputs']);hm=(hm+model(d['inputs'].flip(-1)).flip(-1)[:,meta['flip_indices']])/2 if cfg.val_cfg.get('flip_test',False) else hm
 xy,score=codec.decode(hm.float().cpu().numpy()[0]);m=d['data_samples']['meta'];cv=lambda v:v.detach().cpu().numpy() if torch.is_tensor(v) else np.array(v);xy=(xy/cv(m['input_size'])*cv(m['bbox_scale'])+cv(m['bbox_center'])-.5*cv(m['bbox_scale']))[0];records.append(dict(id=r['id'],xy=xy.tolist(),score=score[0].tolist(),image_sha256=r['image_sha256']));print(r['id'],flush=True)
Path('outputs/newdetector_d10_20261117/predictions.json').write_text(json.dumps(records,indent=2));print('inference completed',len(records),flush=True)
