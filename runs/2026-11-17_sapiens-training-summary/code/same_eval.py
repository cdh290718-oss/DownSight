import sys,json,time
from pathlib import Path
import numpy as np, torch,cv2
base=Path('../EgoSMPLX').resolve();repo=base/'official_baseline/sapiens2';sys.path[:0]=[str(base/'.runtime/sapiens_deps'),str(repo)]
from sapiens.engine.config import Config
from sapiens.engine.datasets import Compose
from sapiens.pose.datasets import parse_pose_metainfo
from sapiens.pose.datasets.codecs.udp_heatmap import UDPHeatmap
from sapiens.registry import MODELS
from safetensors.torch import load_file
cfg=Config.fromfile(repo/'sapiens/pose/configs/keypoints308/shutterstock_goliath_3po/sapiens2_1b_keypoints308_shutterstock_goliath_3po-1024x768.py');cfg.model['backbone'].pop('init_cfg',None)
model=MODELS.build(cfg.model);model.load_state_dict(load_file(str(base/'pretrained_models/external_pose/sapiens2_1b_pose.safetensors')),strict=True);model.cuda().eval();pp=MODELS.build(cfg.data_preprocessor).cuda();pipeline=Compose(cfg.test_pipeline);meta=parse_pose_metainfo(dict(from_file=str(repo/'sapiens/pose/configs/_base_/keypoints308.py')));codec=UDPHeatmap(**{k:v for k,v in cfg.codec.items() if k!='type'})
out=Path('outputs/sapiens_s6_heldout_20261117');rows=json.loads((out/'manifest.json').read_text())['records'];job=Path('outputs/fullmodel_training_20261117/results'); results=[]
for ep in [0,1]:
 if ep:
  model.load_state_dict(load_file(str(job/f'adapter_{ep:02d}.safetensors')),strict=False)
 errs={str(j):[] for j in [5,6,7,8,62,41,9,10,11,12,13,14]};correct=[];failure=0
 for r in rows:
  im=cv2.imread(str(out/r['image']));w,h=r['size'];d=pp(pipeline(dict(img=im,bbox=np.array([[0,0,w,h]],np.float32),bbox_score=np.ones(1,np.float32))))
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
   hm=model(d['inputs']);hm=(hm+model(d['inputs'].flip(-1)).flip(-1)[:,meta['flip_indices']])/2 if cfg.val_cfg.get('flip_test',False) else hm
  xy,score=codec.decode(hm.float().cpu().numpy()[0]); mt=d['data_samples']['meta']; cv=lambda v:v.detach().cpu().numpy() if torch.is_tensor(v) else np.array(v)
  xy=(xy/cv(mt['input_size'])*cv(mt['bbox_scale'])+cv(mt['bbox_center'])-.5*cv(mt['bbox_scale']))[0];e=np.linalg.norm(xy-np.array(r['keypoints308_xy']),axis=-1);mask=np.array(r['keypoints308_supervision_mask'],bool);ok=np.isfinite(e);failure+=int(not ok[mask].all());correct.append(int(((e<=20)&mask&ok).sum()))
  for j in errs:
   if mask[int(j)] and ok[int(j)]:errs[j].append(float(e[int(j)]))
 allerr=[v for vs in errs.values() for v in vs];result=dict(epoch=ep,mean_px=float(np.mean(allerr)),correct_per_image=float(np.mean(correct)),per_joint_px={j:float(np.mean(v)) if v else None for j,v in errs.items()},valid_points=len(allerr),images=len(rows),failed_images=failure);results.append(result);print(json.dumps(result),flush=True)
Path('runs/2026-11-17_fullmodel_sameperson/evidence/results.json').write_text(json.dumps(dict(command='evaluate.py; original, epoch 1, epoch 10; official full bbox and flip test',exit_status=0,results=results),indent=2)+'\n')
