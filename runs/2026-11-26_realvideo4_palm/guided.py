"""Apply the same WiLoR regressor with an RW20000 wrist-centered crop for each hand."""
from pathlib import Path
import sys
import os
import subprocess
import json
import hashlib
RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[1]
OUT=ROOT/'outputs/realvideo4_palm_20261126'
OLD=ROOT/'outputs/realvideo4_dv3_20261125'
FRAMES=[f'clip{i:03}' for i in range(1,5)]
EYES=['left','right']
read=lambda p:json.loads(p.read_text())
write=lambda p,d:p.write_text(json.dumps(d,indent=2)+'\n')
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''): h.update(chunk)
    return h.hexdigest()


def infer(config):
    import cv2
    import numpy as np
    import torch
    from egosmplx_pipeline import wilor_predict as w
    paths=config['paths']
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model,cfg=w.load_wilor(paths['wilor_checkpoint'], paths['wilor_config'])
    model=model.to(device).eval()
    items={r['pair']:r for r in read(OLD/'fit/0/prepared.json')['items']}
    frame_log=read(ROOT/'runs/2026-11-25_realvideo4_dv3/evidence/frames.json')
    assert digest(Path(paths['wilor_checkpoint']))==config['checkpoint_sha256']['wilor']
    records={}
    for n,f in enumerate(FRAMES):
        for v,eye in enumerate(EYES):
            path=OLD/'images'/(f+'_'+eye+'.png')
            assert digest(path)==frame_log[n]['views'][eye]['frame']
            img=cv2.imread(str(path))
            points=[next(pt for pt in items[f]['points'] if pt['name']==s+'_Wrist') for s in ('L','R')]
            assert all(pt['valid'][v] for pt in points)
            centers=np.array([np.array(pt['xy'][v]) for pt in points])
            radius=img.shape[1]/16
            boxes=np.column_stack([centers-radius,centers+radius])
            data=w.ViTDetDataset(cfg,img,boxes,np.array([0,1]),rescale_factor=2.)
            batch=w.recursive_to(next(iter(torch.utils.data.DataLoader(data,batch_size=2))),device)
            with torch.no_grad():
                out=model(batch)
                pred=out['pred_cam'].clone()
                pred[:,1]*=2*batch['right']-1
                sizes=batch['img_size'].float()
                focal=cfg.EXTRA.FOCAL_LENGTH/cfg.MODEL.IMAGE_SIZE*sizes.max()
                tr=w.cam_crop_to_full(pred,batch['box_center'].float(),batch['box_size'].float(),sizes,focal)
                joints=out['pred_keypoints_3d'].detach().clone()
                joints[:,:,0]*=(2*batch['right']-1)[:,None]
            records[path.stem]={}
            for i,side in enumerate(('L','R')):
                uv=w.project(joints[i].cpu().numpy(),tr[i].cpu().numpy(),float(focal),sizes[i].cpu().numpy())
                records[path.stem][side]=dict(uv=uv.tolist(),local=joints[i].cpu().numpy().tolist(),crop_center=centers[i].tolist(),crop_size=float(batch['box_size'][i]))
    write(OUT/'guided.json',dict(selected=records,box_fraction=1/8,rescale=2,exit_status=0))


if __name__=='__main__':
    config=read(ROOT/'configs/pipeline.head01.developer.json')
    if sys.argv[1:]==['infer']:
        infer(config)
    else:
        paths=config['paths']
        mano=OUT/'mano_data'
        if not mano.exists(): mano.symlink_to(Path(paths['wilor_repo'])/'mano_data',target_is_directory=True)
        env=dict(os.environ,PYTHONPATH=os.pathsep.join((str(ROOT),paths['wilor_repo'])),PYTHONDONTWRITEBYTECODE='1',
                 YOLO_CONFIG_DIR=str(OUT/'yolo'),XDG_CACHE_HOME=str(OUT/'cache'),MPLCONFIGDIR=str(OUT/'mpl'),
                 TORCH_HOME=str(OUT/'torch'),TMPDIR=str(OUT/'tmp'),CUDA_CACHE_PATH=str(OUT/'cuda'),
                 TORCHINDUCTOR_CACHE_DIR=str(OUT/'inductor'),TRITON_CACHE_DIR=str(OUT/'triton'))
        subprocess.run([paths['wilor_python'],'-Bu',str(Path(__file__).resolve()),'infer'],cwd=OUT,env=env,check=True)
