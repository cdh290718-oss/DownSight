from pathlib import Path
import json,sys
import numpy as np
import cv2
RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[1]
OUT=ROOT/'outputs/native_mano_audit_20261127'
sys.path.insert(0,str(RUN.parent/'2026-11-27_vidihand2_comparison'))
import predict
sys.path.insert(0,str(RUN.parent/'2026-11-25_realvideo4_dv3'))
from calibrate import projection

def inputs():
    folder=ROOT/'outputs/rw_a20000_dataset_20261117'
    pairs={}
    for row in predict.read(folder/'validation.json')['records']:
        pairs.setdefault(row['pair_id'],{})[row['view']]=row
    maps={}
    K=predict.videos['clip003_left']['K']
    y,x=np.indices((480,640))
    rays=np.stack([x,y,np.ones_like(x)],-1).reshape(-1,3)@np.linalg.inv(K).T
    for eye in ('left','right'):
        cam=predict.read(ROOT/'outputs/rw_frozen200_golden_d_20261118'/('camera_'+eye+'.json'))
        a=cam['input_adapter']
        maps[eye]=((projection(rays,cam)-a['offset'])/a['scale']).reshape(480,640,2).astype('f')
    assert len(pairs)==200
    for n,k in enumerate(sorted(pairs),1):
        for eye in ('left','right'):
            row=pairs[k][eye]
            path=folder/row['image']
            assert predict.digest(path)==row['image_sha256']
            img=cv2.remap(cv2.imread(str(path)),maps[eye],None,cv2.INTER_LINEAR)
            yield f'val{n:03d}_{eye}',dict(K=K,center_index=0),img[None,:,:,::-1]


import torch
from egosmplx_pipeline import wilor_predict as w
read=predict.read
digest=predict.digest
ARCH=RUN.parent/'2026-11-27_validation200_v5/evidence'
choices=read(ARCH/'selection.json')
previous=read(ARCH/'predictions.json')

def main():
    from egosmplx_pipeline import wilor_predict as w
    cfg=read(ROOT/'configs/pipeline.head01.developer.json')
    paths=cfg['paths']
    for field,name in [('wilor','wilor_checkpoint'),('detector','wilor_detector')]:
        assert digest(Path(paths[name]))==cfg['checkpoint_sha256'][field]
    model,conf=w.load_wilor(paths['wilor_checkpoint'],paths['wilor_config'])
    model=model.cuda().eval()
    detector=w.load_yolo_detector(paths['wilor_detector']).to('cuda')
    stats={}
    for key,info,video in inputs():
        frame_id,view=key.split('_')
        if not any(h['source']=={'left':'cam3','right':'cam4'}[view] for h in choices[frame_id]['hands'].values()):
            continue
        frame=cv2.cvtColor(video[info['center_index']],cv2.COLOR_RGB2BGR)
        det=detector(frame,conf=.20,verbose=False)[0].boxes
        cls=det.cls.detach().cpu().numpy()
        scores=det.conf.detach().cpu().numpy()
        boxes=det.xyxy.detach().cpu().numpy()
        keep=[int(idx[np.argmax(scores[idx])]) for side in (0,1) for idx in [np.flatnonzero(cls==side)] if len(idx)]
        saved={}
        if keep:
            ds=w.ViTDetDataset(conf,frame,boxes[keep],cls[keep],rescale_factor=2.)
            batch=w.recursive_to(next(iter(torch.utils.data.DataLoader(ds,batch_size=len(keep)))),'cuda')
            with torch.no_grad():
                pred=model(batch)
                param=pred['pred_mano_params']
                camera=pred['pred_cam'].clone()
                camera[:,1]*=2*batch['right']-1
                sizes=batch['img_size'].float()
                focal=conf.EXTRA.FOCAL_LENGTH/conf.MODEL.IMAGE_SIZE*sizes.max()
                tr=w.cam_crop_to_full(camera,batch['box_center'].float(),batch['box_size'].float(),sizes,focal).cpu().numpy()
                eye=torch.eye(3,device='cuda').reshape(1,1,3,3)
                neutral=model.mano(global_orient=eye.expand(len(keep),1,3,3),hand_pose=eye.expand(len(keep),15,3,3),betas=param['betas'],pose2rot=False)
                rest=neutral.joints[:,:21].cpu().numpy()
            for b,idx in enumerate(keep):
                side=('left','right')[int(cls[idx])]
                sign=np.array([1 if cls[idx] else -1,1,1])
                local=pred['pred_keypoints_3d'][b].detach().cpu().numpy()*sign
                uv=w.project(local,tr[b],float(focal),sizes[b].cpu().numpy())
                frame_id,view=key.split('_')
                hand=choices[frame_id]['hands'][side]
                source={'left':'cam3','right':'cam4'}[view]
                if hand['source']!=source:
                    continue
                orig=ROOT/'outputs/validation200_v5_20261127/poses'/('wilor_'+key+'_'+side+'.npz')
                assert digest(orig)==previous[key]['sha256'][side]==hand['candidates'][source]['source_sha256']
                archive=np.load(orig)
                delta={k:float(np.max(abs(x-archive[k]))) for k,x in [('pose',param['hand_pose'][b].cpu().numpy()),('rest',rest[b]*sign),('uv',uv)]}
                assert delta['pose']<1e-5 and delta['rest']<1e-6 and delta['uv']<.001,delta
                vertices=pred['pred_vertices'][b].cpu().numpy()*sign
                path=OUT/'poses'/(frame_id+'_'+side+'.npz')
                np.savez_compressed(path,vertices=vertices,faces=model.mano.faces,neutral=neutral.vertices[b].cpu().numpy()*sign,joints=local,weights=model.mano.lbs_weights.cpu().numpy())
                stats[frame_id+'_'+side]=dict(view=view,errors=delta,sha256=digest(path))
    assert len(stats)==280
    (RUN/'evidence/replay.json').write_text(json.dumps(stats))

if __name__=="__main__": main()
