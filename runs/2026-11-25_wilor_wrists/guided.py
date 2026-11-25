"""Apply the same WiLoR regressor with an RW20000 wrist-centered crop for each hand."""
from pathlib import Path
import sys
import os
import subprocess
from detect import OUT, ROOT, RUN, FRAMES, EYES, digest, read, write


def infer(config):
    import cv2
    import numpy as np
    import torch
    from egosmplx_pipeline import wilor_predict as w
    paths=config['paths']
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model,cfg=w.load_wilor(paths['wilor_checkpoint'], paths['wilor_config'])
    model=model.to(device).eval()
    items={r['pair']:r for r in read(ROOT/'outputs/stereo_wrist_candidates_20261123/validation_fit/prepared.json')['items']}
    images=read(OUT/'images.json')['frames']
    adapters=[read(ROOT/'outputs/rw_frozen200_golden_d_20261118'/('camera_'+eye+'.json'))['input_adapter'] for eye in EYES]
    records={}
    for n,f in enumerate(FRAMES):
        for v,adapter in enumerate(adapters):
            path=Path(images[2*n+v]['image'])
            assert digest(path)==read(RUN/'evidence/detections.json')['input_sha256'][path.stem]
            img=cv2.imread(str(path))
            points=[next(pt for pt in items[f]['points'] if pt['name']==s+'_Wrist') for s in ('L','R')]
            assert all(pt['valid'][v] for pt in points)
            centers=np.array([(np.array(pt['xy'][v])-adapter['offset'])/adapter['scale'] for pt in points])
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
                uv=w.project(joints[i,:1].cpu().numpy(),tr[i].cpu().numpy(),float(focal),sizes[i].cpu().numpy())[0]
                records[path.stem][side]=dict(uv=uv.tolist(),crop_center=centers[i].tolist(),crop_size=float(batch['box_size'][i]))
    write(RUN/'evidence/guided.json',dict(selected=records,box_fraction=1/8,rescale=2,exit_status=0))


if __name__=='__main__':
    config=read(ROOT/'configs/pipeline.head01.developer.json')
    if sys.argv[1:]==['infer']:
        infer(config)
    else:
        paths=config['paths']
        env=dict(os.environ,PYTHONPATH=os.pathsep.join((str(ROOT),paths['wilor_repo'])),PYTHONDONTWRITEBYTECODE='1',
                 YOLO_CONFIG_DIR=str(OUT/'yolo'),XDG_CACHE_HOME=str(OUT/'cache'),MPLCONFIGDIR=str(OUT/'mpl'),
                 TORCH_HOME=str(OUT/'torch'),TMPDIR=str(OUT/'tmp'),CUDA_CACHE_PATH=str(OUT/'cuda'),
                 TORCHINDUCTOR_CACHE_DIR=str(OUT/'inductor'),TRITON_CACHE_DIR=str(OUT/'triton'))
        subprocess.run([paths['wilor_python'],'-Bu',str(Path(__file__).resolve()),'infer'],cwd=OUT,env=env,check=True)
