"""Paired detector comparison; raw coordinates/checkpoints stay in local outputs."""
import collections
import hashlib
import json
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from safetensors.torch import load_file

RUN=Path(__file__).resolve().parents[1];ROOT=RUN.parents[1];BACK=ROOT.parent/'EgoSMPLX'
OUT=ROOT/'outputs/rw20000_frozen200_20261118';REPO=BACK/'official_baseline/sapiens2'
sys.path[:0]=[str(BACK/'.runtime/sapiens_deps'),str(REPO)]
from sapiens.engine.config import Config
from sapiens.engine.datasets import Compose
from sapiens.registry import MODELS
from sapiens.pose.datasets import parse_pose_metainfo
from sapiens.pose.datasets.codecs.udp_heatmap import UDPHeatmap

def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    start=time.time();torch.set_num_threads(8)
    training=ROOT/'outputs/rw_a20000_training_20261117T152700444455Z'
    assert read(training/'exit.json')['exit_code']==0
    assert read(training/'results/status.json')['state']=='completed'
    weights={'original':BACK/'pretrained_models/external_pose/sapiens2_1b_pose.safetensors',
             'rw500':ROOT/'outputs/fullmodel_training_20261117/results/adapter_01.safetensors',
             'rw20000':training/'results/adapter_01.safetensors'}
    hashes={v:sha(p) for v,p in weights.items()}
    assert hashes['rw20000']=='57be3cd654227a7a3ecb3a7a29bdff544f7288fc63d14f41ef71226e7d971c98'
    assert hashes['original']=='2dab7014a17e99e460c18817325a71dd7a81ce48d87027f01c2ee7d7b3af969f'
    write(RUN/'evidence/checkpoints.json',dict(sha256=hashes,training_exit=read(training/'exit.json'),
        training_status=read(training/'results/status.json'),best=read(training/'results/best.json'),
        old_training=read(ROOT/'runs/2026-11-17_sapiens-training-summary/evidence/full_training.json')['status'],
        command='hash checkpoints and verify completed training state',exit_status=0))
    manifests={'test_frozen':ROOT/'outputs/rw_test_completion_20261117/test_manifest.json'}
    datasets={k:read(p)['records'] for k,p in manifests.items()}
    training_rows=read(ROOT/'outputs/rw_a20000_dataset_20261117/train.json')['records']
    train_sequences={r['sequence'] for r in training_rows};train_hashes={r['image_sha256'] for r in training_rows}
    for dataset,rows in datasets.items():
        assert not train_sequences&{r['sequence'] for r in rows}
        assert not train_hashes&{r['image_sha256'] for r in rows}
        for r in rows:assert sha(manifests[dataset].parent/r['image'])==r['image_sha256']
    mapping=[m for m in read(ROOT/'runs/2026-11-16_sapiens-unrealego-audit/evidence/audit.json')['joint_mapping'] if m['rw_index'] is not None]
    joints=[m['sapiens308_id'] for m in mapping]
    cfg=Config.fromfile(REPO/'sapiens/pose/configs/keypoints308/shutterstock_goliath_3po/sapiens2_1b_keypoints308_shutterstock_goliath_3po-1024x768.py')
    cfg.model['backbone'].pop('init_cfg',None)
    model=MODELS.build(cfg.model);model.cuda().eval();pp=MODELS.build(cfg.data_preprocessor).cuda()
    pipeline=Compose(cfg.test_pipeline)
    meta=parse_pose_metainfo(dict(from_file=str(REPO/'sapiens/pose/configs/_base_/keypoints308.py')))
    codec=UDPHeatmap(**{k:v for k,v in cfg.codec.items() if k!='type'})
    def infer(image):
        im=cv2.imread(str(image));assert im is not None;h,w=im.shape[:2]
        d=pp(pipeline(dict(img=im,bbox=np.array([[0,0,w,h]],np.float32),bbox_score=np.ones(1,np.float32))))
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            hm=model(d['inputs'])
            if cfg.val_cfg.get('flip_test',False):hm=(hm+model(d['inputs'].flip(-1)).flip(-1)[:,meta['flip_indices']])/2
        xy,score=codec.decode(hm.float().cpu().numpy()[0]);m=d['data_samples']['meta']
        cv=lambda x:x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
        xy=(xy/cv(m['input_size'])*cv(m['bbox_scale'])+cv(m['bbox_center'])-.5*cv(m['bbox_scale']))[0]
        assert np.isfinite(xy).all()
        return xy,score[0]
    summary={};details=[]
    for version in ['rw20000']:
        model.load_state_dict(load_file(str(weights['original'])),strict=True)
        if version!='original':
            adapter=load_file(str(weights[version]));assert all(n in adapter for n,_ in model.named_parameters())
            state=model.load_state_dict(adapter,strict=False);assert not state.unexpected_keys;del adapter
        model.eval();summary[version]={}
        for dataset,rows in datasets.items():
            raw=[];errs=collections.defaultdict(list);subject=collections.defaultdict(list);sequence=collections.defaultdict(list);correct=[];count=0
            reuse=dataset=='validation' and version in ['original','rw5000']
            reused={}
            if reuse:
                epoch='00' if version=='original' else '01'
                reused={r['id']:r for r in read(training/('results/predictions_'+epoch+'.json'))}
                assert set(reused)=={r['id'] for r in rows}
            for n,r in enumerate(rows):
                if reuse:
                    cached=reused[r['id']];xy=np.array(cached['xy']);score=np.array(cached['scores'])[0]
                    assert cached['mask']==[bool(x) for x in r['keypoints308_supervision_mask']]
                else:xy,score=infer(manifests[dataset].parent/r['image'])
                gt=np.asarray(r['keypoints308_xy']);mask=np.asarray(r['keypoints308_supervision_mask'],bool)
                assert not mask[0] and all(not v or j in joints for j,v in enumerate(mask))
                error=np.linalg.norm(xy-gt,axis=-1);assert np.isfinite(error[mask]).all()
                count+=int(mask.sum());correct.append(int(((error<=20)&mask).sum()))
                for j in joints:
                    if mask[j]:errs[str(j)].append(float(error[j]));subject[r['subject']].append(float(error[j]));sequence[r['sequence']].append(float(error[j]))
                details.append(dict(version=version,dataset=dataset,id=r['id'],subject=r['subject'],sequence=r['sequence'],pair_id=r['pair_id'],
                                    per_joint_error_px=[float(error[j]) if mask[j] else None for j in joints]))
                raw.append(dict(id=r['id'],xy=xy.tolist(),score=score.tolist(),image_sha256=r['image_sha256']))
                if not reuse and (n+1)%100==0:print(version,dataset,n+1,'/',len(rows),flush=True)
            values=[v for arr in errs.values() for v in arr]
            result=dict(images=len(rows),valid_points=count,mean_px=float(np.mean(values)) if values else None,
                        correct_20px=sum(correct),pck20_percent=100*sum(correct)/count if count else None,
                        correct_points_per_image=float(np.mean(correct)) if correct else None,failed_images=0,
                        per_joint_px={str(j):float(np.mean(errs[str(j)])) if errs[str(j)] else None for j in joints},
                        per_subject={s:dict(points=len(v),mean_px=float(np.mean(v))) for s,v in subject.items()},
                        per_sequence={s:dict(points=len(v),mean_px=float(np.mean(v))) for s,v in sequence.items()},
                        reused_frozen_training_predictions=reuse)
            if reuse:
                prior=read(training/('results/eval_'+epoch+'.json'))
                assert abs(result['mean_px']-prior['mean_px'])<1e-6
            summary[version][dataset]=result
            write(OUT/(version+'_'+dataset+'_predictions.json'),raw)
            print(version,dataset,'mean_px',result['mean_px'],flush=True)
            write(RUN/'evidence/2d_progress.json',summary)
    write(RUN/'evidence/2d_errors.json',dict(joint_names=[m['name'] for m in mapping],joint_ids=joints,records=details))
    write(RUN/'evidence/2d_summary.json',dict(command='<TRAIN_PYTHON> code/evaluate_2d.py',exit_status=0,
        summary=summary,checkpoint_sha256=hashes,manifest_sha256={k:sha(p) for k,p in manifests.items()},
        elapsed_seconds=time.time()-start,scope='frozen test selection; 200 images across S1 S9 S11; existing label semantics and visibility limits'))
    print('ALL DETECTOR EVALUATIONS COMPLETE',flush=True)
if __name__=='__main__':main()
