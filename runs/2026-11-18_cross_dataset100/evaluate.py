"""Four frozen Sapiens2 checkpoints on fixed local manifests, full image inference."""
import os,json,sys,hashlib,time,argparse
from pathlib import Path
import cv2,numpy as np,torch
from safetensors.torch import load_file
ROOT=Path(__file__).resolve().parents[2];RUN=Path(__file__).resolve().parent;OUT=ROOT/'outputs/cross_dataset100_20261118';back=ROOT.parent/'EgoSMPLX';repo=back/'official_baseline/sapiens2'
sys.path[:0]=[str(back/'.runtime/sapiens_deps'),str(repo)]
from sapiens.engine.config import Config
from sapiens.engine.datasets import Compose
from sapiens.registry import MODELS
from sapiens.pose.datasets import parse_pose_metainfo
from sapiens.pose.datasets.codecs.udp_heatmap import UDPHeatmap
parser=argparse.ArgumentParser();parser.add_argument('--datasets',nargs='+',default=['egopw','sceneego','unrealego2']);args=parser.parse_args()
frames=[]
for dataset in args.datasets:
 f=json.loads((OUT/(dataset+'_manifest.json')).read_text());assert len(f)==100 and len({r['id'] for r in f})==100;frames.extend(f)
cpath=repo/'sapiens/pose/configs/keypoints308/shutterstock_goliath_3po/sapiens2_1b_keypoints308_shutterstock_goliath_3po-1024x768.py';c=Config.fromfile(cpath);c.model['backbone'].pop('init_cfg',None);model=MODELS.build(c.model).cuda().eval();pp=MODELS.build(c.data_preprocessor).cuda();pipeline=Compose(c.test_pipeline);meta=parse_pose_metainfo(dict(from_file=str(repo/'sapiens/pose/configs/_base_/keypoints308.py')));codec=UDPHeatmap(**{k:v for k,v in c.codec.items() if k!='type'})
base=back/'pretrained_models/external_pose/sapiens2_1b_pose.safetensors'
weights={'original':base,'rw500':ROOT/'outputs/fullmodel_training_20261117/results/adapter_01.safetensors','rw5000':ROOT/'outputs/rw_a5000_training_20261117T122117126742Z/results/adapter_01.safetensors','rw20000':ROOT/'outputs/rw_a20000_training_20261117T152700444455Z/results/adapter_01.safetensors'}
expected=json.loads((ROOT/'runs/2026-11-18_egowholebody100/evidence/weights.json').read_text());weight_records={};start=time.time()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
for version,path in weights.items():
 weight_records[version]={'sha256':sha(path)};assert weight_records[version]['sha256']==expected[version]['sha256']
 model.load_state_dict(load_file(str(base)),strict=True)
 if version!='original':
  state=load_file(str(path));assert all(n in state for n,_ in model.named_parameters());check=model.load_state_dict(state,strict=False);assert not check.unexpected_keys;del state
 results=[]
 for row in frames:
  tick=time.time();record={k:row[k] for k in ['dataset','sequence','id','image_sha256']}
  try:
   image=Path(row['image']);assert sha(image)==row['image_sha256'];im=cv2.imread(str(image));assert im is not None;h,w=im.shape[:2];assert (w,h)==(row['width'],row['height'])
   d=pp(pipeline(dict(img=im,bbox=np.array([[0,0,w,h]],np.float32),bbox_score=np.ones(1,np.float32))))
   with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
    hm=model(d['inputs'])
    if c.val_cfg.get('flip_test',False):hm=(hm+model(d['inputs'].flip(-1)).flip(-1)[:,meta['flip_indices']])/2
   xy,score=codec.decode(hm.float().cpu().numpy()[0]);m=d['data_samples']['meta'];conv=lambda v:v.detach().cpu().numpy() if torch.is_tensor(v) else np.array(v);xy=(xy/conv(m['input_size'])*conv(m['bbox_scale'])+conv(m['bbox_center'])-.5*conv(m['bbox_scale']))[0]
   assert np.isfinite(xy).all()
   record['points']=[]
   for point in row['points']:
    j=point['sapiens_index'];p=xy[j];valid=point['reference_in_image'];record['points'].append({'name':point['name'],'reference_in_image':valid,'error_px':float(np.linalg.norm(p-np.array(point['reference']))) if valid else None,'prediction':p.tolist(),'score':float(score[0,j]),'prediction_in_image':bool(0<=p[0]<w and 0<=p[1]<h)})
   record['status']='ok'
  except Exception as e:
   record.update(status='failed',error=type(e).__name__+': '+str(e))
  record['elapsed_seconds']=time.time()-tick;results.append(record)
  print(version,row['id'],record['status'],flush=True)
  if len(results)%25==0:(OUT/'progress.json').write_text(json.dumps({'model':version,'completed':len(results),'total':len(frames),'elapsed_seconds':time.time()-start}))
 for dataset in args.datasets:(OUT/f'{dataset}_{version}.json').write_text(json.dumps([r for r in results if r['dataset']==dataset],indent=2))
(RUN/'evidence'/('inference_'+ '_'.join(args.datasets)+'.json')).write_text(json.dumps({'datasets':args.datasets,'command':'PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 <PYTHON_ENV>/bin/python runs/2026-11-18_cross_dataset100/evaluate.py --datasets '+' '.join(args.datasets),'elapsed_seconds':time.time()-start,'weights':weight_records,'config_sha256':sha(cpath),'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'flip_test':bool(c.val_cfg.get('flip_test',False)),'precision':'bfloat16 autocast','exit_status':0},indent=2)+'\n')
print('DONE',flush=True)
