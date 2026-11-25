"""Produce exactly the fp16 probability samples consumed by the prior stereo contour term."""
import hashlib,json,sys
from pathlib import Path
import cv2,numpy as np,torch
from torch.nn.functional import interpolate
from safetensors.torch import load_file

ROOT=Path.cwd()
RUN=Path(__file__).resolve().parent
OUT=ROOT/'outputs/validation200_fingers_20261125'
read=lambda p: json.loads(p.read_text())
def sha(p):
    digest=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<24),b''): digest.update(chunk)
    return digest.hexdigest()

def main():
    conf=read(ROOT/'configs/pipeline.head01.developer.json')
    paths=conf['paths']
    sys.path[:0]=[paths['sapiens_repo'],paths['sapiens_dependencies']]
    from sapiens.engine.config import Config
    from sapiens.engine.datasets import Compose
    from sapiens.registry import MODELS
    config=Path(paths['sapiens_repo'])/'sapiens/dense/configs/seg/shutterstock_goliath/sapiens2_1b_seg_shutterstock_goliath-1024x768.py'
    ck=Path(paths['seg_checkpoint'])
    assert sha(ck)==conf['checkpoint_sha256']['seg']
    cfg=Config.fromfile(config);cfg.model['backbone'].pop('init_cfg',None)
    net=MODELS.build(cfg.model)
    net.load_state_dict(load_file(str(ck)),strict=True)
    net=net.cuda().eval()
    prep=MODELS.build(cfg.data_preprocessor).cuda()
    pipe=Compose(cfg.test_pipeline)
    m=ROOT/'outputs/rw_a20000_dataset_20261117/validation.json'
    archive=read(ROOT/'runs/2026-11-23_stereo_wrist_candidates/evidence/splits.json')
    assert sha(m)==archive['hashes']['validation_manifest']
    pairs={}
    for r in read(m)['records']:
        views=pairs.setdefault(r['pair_id'],{})
        assert r['view'] not in views
        views[r['view']]=r
    assert len(pairs)==200
    results={}
    for i,(_,views) in enumerate(sorted(pairs.items()),1):
        pair=f'val{i:03d}'
        for side in ('left','right'):
            r=views[side]
            path=m.parent/r['image']
            assert r['view']==side and sha(path)==r['image_sha256']
            im=cv2.imread(str(path)); assert im.shape==(872,872,3)
            x=prep(pipe(dict(img=im)))['inputs']
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16): logits=net(x)
            prob=interpolate(logits.float(),size=im.shape[:2],mode='bilinear',align_corners=False).softmax(1)[0,:,2::4,2::4].cpu().numpy().astype(np.float16)
            assert prob.shape==(29,218,218) and np.isfinite(prob).all()
            name=pair+'_'+side
            output=OUT/'seg'/(name+'.npy')
            temp=output.with_suffix('.partial.npy')
            np.save(temp,prob)
            temp.replace(output)
            results[name]=dict(image_sha256=r['image_sha256'],sampled_sha256=sha(output))
        if len(results)%20==0: print('eyes',len(results),flush=True)
    assert len(results)==400
    (RUN/'evidence/segmentation.json').write_text(json.dumps(dict(eyes=results,weight_sha256=sha(ck),config_sha256=sha(config),manifest_sha256=sha(m),exit_status=0),indent=2)+'\n')

if __name__=='__main__':main()
