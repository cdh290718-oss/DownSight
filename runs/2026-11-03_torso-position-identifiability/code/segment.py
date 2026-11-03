"""Compute Sapiens labels only for preselected stereo frames; no annotation input."""
import os,sys,argparse,json,hashlib,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--deps',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--images',type=Path,required=True);p.add_argument('--frames',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
sys.path.insert(0,str(a.repo));sys.path.insert(0,str(a.deps))
import cv2,numpy as np,torch
import torch.nn.functional as F
from safetensors.torch import load_file
from sapiens.engine.config import Config
from sapiens.engine.datasets import Compose
from sapiens.registry import MODELS
from sapiens.dense.datasets import DOME_CLASSES_29
sha=lambda x:hashlib.sha256(x.read_bytes()).hexdigest()
expected='4b73c44963b377e93fcb4c4053f72a189836a22d05e12c30383046b9cd3c5bd4';assert sha(a.checkpoint)==expected
cfg=Config.fromfile(a.repo/'sapiens/dense/configs/seg/shutterstock_goliath/sapiens2_1b_seg_shutterstock_goliath-1024x768.py');cfg.model['backbone'].pop('init_cfg',None)
model=MODELS.build(cfg.model);model.load_state_dict(load_file(str(a.checkpoint),device='cpu'),strict=True);model=model.cuda().eval();prep=MODELS.build(cfg.data_preprocessor).cuda();pipe=Compose(cfg.test_pipeline)
a.output.mkdir(parents=True,exist_ok=False);obs=np.load(a.frames,allow_pickle=False); frames=obs['frames']; records=[]
for frame in frames:
 for cam in (2,3):
  path=a.images/f'cam{cam}'/f'{int(frame):05d}.jpg';im=cv2.imread(str(path));assert im is not None
  data=prep(pipe(dict(img=im)));inputs=data['inputs']
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):logits=model(inputs)
  logits=F.interpolate(logits.float(),size=im.shape[:2],mode='bilinear',align_corners=False);labels=logits.argmax(1)[0].to(torch.uint8).cpu().numpy();dest=a.output/f'{int(frame):05d}_cam{cam}.npy';np.save(dest,labels)
  counts={str(k):int(np.sum(labels==k)) for k in (11,20,22,23)}; records.append({'frame':int(frame),'camera':cam,'counts':counts,'image_sha256':sha(path),'labels_sha256':sha(dest)});print('DONE',frame,cam,counts,flush=True)
with (a.output/'metadata.json').open('x') as f:json.dump({'checkpoint_sha256':expected,'class_names':{str(k):DOME_CLASSES_29[k]['name'] for k in (11,20,22,23)},'records':records,'annotation_input':False},f,indent=2)
