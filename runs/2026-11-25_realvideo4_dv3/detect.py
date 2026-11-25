"""Same single-image GPU inference and decoder as the frozen RW20000 benchmark."""
import sys,time,hashlib
from fileset import ROOT,RUN,OUT,read,write,sha
BACK=ROOT.parent/"EgoSMPLX"
sys.path.insert(0,str(ROOT/'runs/2026-11-18_rw20000_frozen200/code'))
import evaluate_2d as d
import numpy as np
import torch
import cv2

def main():
    expected=read(ROOT/'runs/2026-11-18_rw20000_frozen200/evidence/checkpoints.json')['sha256']
    base=BACK/'pretrained_models/external_pose/sapiens2_1b_pose.safetensors'
    trained=ROOT/'outputs/rw_a20000_training_20261117T152700444455Z/results/adapter_01.safetensors'
    assert sha(base)==expected['original'] and sha(trained)==expected['rw20000']
    cfg=d.Config.fromfile(d.REPO/'sapiens/pose/configs/keypoints308/shutterstock_goliath_3po/sapiens2_1b_keypoints308_shutterstock_goliath_3po-1024x768.py')
    cfg.model['backbone'].pop('init_cfg',None)
    model=d.MODELS.build(cfg.model).cuda().eval()
    model.load_state_dict(d.load_file(str(base)),strict=True)
    weights=d.load_file(str(trained))
    assert all(k in weights for k,_ in model.named_parameters())
    assert not model.load_state_dict(weights,strict=False).unexpected_keys
    del weights
    pp=d.MODELS.build(cfg.data_preprocessor).cuda()
    pipeline=d.Compose(cfg.test_pipeline)
    meta=d.parse_pose_metainfo(dict(from_file=str(d.REPO/'sapiens/pose/configs/_base_/keypoints308.py')))
    codec=d.UDPHeatmap(**{k:v for k,v in cfg.codec.items() if k!='type'})
    pending={r['pair']+'_'+s+'.png' for r in read(RUN/'evidence/frames.json') for s in ('left','right')}
    finished={}
    while pending:
        ready=[p for p in (OUT/'images').glob('*.png') if p.name in pending and time.time()-p.stat().st_mtime>1]
        if not ready:
            time.sleep(1);continue
        for path in sorted(ready):
            data=path.read_bytes()
            im=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
            h,w=im.shape[:2]
            assert (h,w)==(720,1280)
            sample=pp(pipeline(dict(img=im,bbox=np.array([[0,0,w,h]],np.float32),bbox_score=np.ones(1,np.float32))))
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                hm=model(sample['inputs'])
                if cfg.val_cfg.get('flip_test',False):
                    hm=(hm+model(sample['inputs'].flip(-1)).flip(-1)[:,meta['flip_indices']])/2
            xy,_=codec.decode(hm.float().cpu().numpy()[0]);m=sample['data_samples']['meta']
            def array(x):return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
            xy=(xy/array(m['input_size'])*array(m['bbox_scale'])+array(m['bbox_center'])-.5*array(m['bbox_scale']))[0]
            assert xy.shape==(308,2) and np.isfinite(xy).all()
            finished[path.name]=dict(image_sha256=hashlib.sha256(data).hexdigest(),xy=xy.tolist())
            pending.remove(path.name)
            if len(finished)%100==0:print('detected',len(finished),flush=True)
    write(OUT/'predictions.json',finished)
    write(RUN/'evidence/detection.json',dict(images=len(finished),predictions_sha256=sha(OUT/'predictions.json'),
        weights_sha256=expected['rw20000'],base_weights_sha256=expected['original'],flip=bool(cfg.val_cfg.get('flip_test',False)),exit_status=0))
if __name__=='__main__':main()
