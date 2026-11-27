import json
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
from fetch import RUN,ROOT,OUT,digest
read=lambda p:json.loads(p.read_text())
PRIOR=RUN.parent/'2026-11-27_vidihand2_wrist'
videos=read(PRIOR/'evidence/video.json')

def inputs():
    for key,info in videos.items():
        f=ROOT/'outputs/vidihand2_wrist_20261127/video'/(key+'.npy')
        assert digest(f)==info['file_sha256']
        yield key,info,np.load(f,mmap_mode='r')

def vidi():
    from vidihand_v2.inference import load_model
    from utils.mano import ManoModel
    assert (RUN/'evidence/download.json').is_file()
    model=load_model()
    order=list(ManoModel.openpose)
    stats={}
    for key,info,video in inputs():
        center=info['center_index']
        result=model(dict(video=video,K=np.array(info['K'])[None],source_frame_ids=info['source_one_based']-1+(np.arange(len(video))-center)*2,fps=info['fps']))
        joint,mano=result['joint'],result['mano']
        valid=mano['valid']
        mj=model.mano.mano2joint(mano)[center][:,order]
        prefix=OUT/'poses'/('vidi_'+key)
        saved={}
        for side in range(2):
            if not valid[center,side]: continue
            beta=mano['betas'][center,side]
            rest=model.mano.joints(torch.eye(3,device='cuda').expand(1,16,3,3),torch.tensor(beta,device='cuda')[None]).detach().cpu().numpy()[0,order]
            if side==0: rest[:,0]*=-1
            name=('left','right')[side]
            path=prefix.with_name(prefix.name+'_'+name+'.npz')
            np.savez_compressed(path,rest=rest,pose=mano['hand_pose'][center,side],joint=joint['joints3d'][center,side,order],mano_joint=mj[side],K=np.array(info['K']))
            saved[name]=digest(path)
        stats[key]=dict(center_valid=valid[center].tolist(),valid_count=valid.sum(0).tolist(),valid_by_frame=valid.tolist(),frames=len(video),sha256=saved)
        print(key,stats[key],flush=True)
    return stats

def wilor():
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
                rest=model.mano(global_orient=eye.expand(len(keep),1,3,3),hand_pose=eye.expand(len(keep),15,3,3),betas=param['betas'],pose2rot=False).joints[:,:21].cpu().numpy()
            for b,idx in enumerate(keep):
                side=('left','right')[int(cls[idx])]
                sign=np.array([1 if cls[idx] else -1,1,1])
                local=pred['pred_keypoints_3d'][b].detach().cpu().numpy()*sign
                uv=w.project(local,tr[b],float(focal),sizes[b].cpu().numpy())
                path=OUT/'poses'/('wilor_'+key+'_'+side+'.npz')
                np.savez_compressed(path,rest=rest[b]*sign,pose=param['hand_pose'][b].detach().cpu().numpy(),uv=uv,K=np.array(info['K']))
                saved[side]=digest(path)
        stats[key]=dict(raw_classes=cls.tolist(),raw_scores=scores.tolist(),sha256=saved)
        print(key,stats[key],flush=True)
    return stats

def temporal():
    from egosmplx_pipeline.wilor_predict import load_yolo_detector
    cfg=read(ROOT/'configs/pipeline.head01.developer.json')
    path=Path(cfg['paths']['wilor_detector'])
    assert digest(path)==cfg['checkpoint_sha256']['detector']
    detector=load_yolo_detector(str(path)).to('cuda')
    baseline=read(RUN/'evidence/wilor.json')['views']
    stats={}
    for key,info,video in inputs():
        counts=[]
        for n,rgb in enumerate(video):
            detections=detector(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),conf=.20,verbose=False)[0].boxes
            labels=detections.cls.detach().cpu().numpy()
            if n==info['center_index']:
                assert np.array_equal(labels,baseline[key]['raw_classes'])
            counts.append([int((labels==side).sum()) for side in (0,1)])
        stats[key]=dict(boxes_by_frame=counts)
    return stats

if __name__=='__main__':
    m=sys.argv[1]
    stats={'vidi':vidi,'wilor':wilor,'temporal':temporal}[m]()
    (RUN/'evidence'/(m+'.json')).write_text(json.dumps(dict(views=stats,torch=torch.__version__,gpu=torch.cuda.get_device_name(),exit_status=0),indent=2)+'\n')
