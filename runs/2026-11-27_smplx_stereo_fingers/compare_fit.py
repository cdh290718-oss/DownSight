"""Keep the arm-corrected SMPL-X scaffold and test the pre-existing MANO observation mask."""
from pathlib import Path
import sys
import cv2
import numpy as np
import torch

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
OUT = ROOT/'outputs/smplx_stereo_fingers_20261127'
ARM = RUN.parent/'2026-11-26_purple_arm_hand_a'
ARM_OUT = ROOT/'outputs/purple_arm_hand_a_20261126'
MANO = RUN.parent/'2026-11-26_wilor_stereo_fingers'
sys.path.insert(0,str(RUN.parent/'2026-11-26_realvideo4_wilor_outline'))
import paired
from paired import read,write,sha,ref,SIDES,JOINT_NAMES
DIGITS = ['thumb','index','middle','ring','pinky']
TIPS = [4,8,12,16,20]
ROOTS = [5,9,13,17]
FINGER_JOINTS = [i for i in range(1,21) if i not in TIPS]


def fit(i):
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    key = f'clip{i+1:03}'
    ref.prev.ref.camera_files = ref.prev.base.fit.camera_files = ref.cameras
    ref.compare.OUT = ref.OUT
    prior = read(ARM/'evidence'/(key+'.json'))
    seed = ARM_OUT/'palm/poses'/(key+'.npz')
    assert sha(seed)==prior['start_sha256']
    pack = dict(np.load(seed))
    x0 = torch.from_numpy(pack['locked'])
    start = dict(body_pose=x0[:21].numpy(),delta=x0[21].numpy(),translation=x0[22].numpy(),global0=pack['global0'],betas=pack['betas'])
    folder = ref.OLD/'fit/0'
    input_hashes = read(ref.PRIOR/'evidence/inputs.json')
    assert sha(folder/'prepared.json')==input_hashes['shards']['0'] and sha(folder/'protocol.json')==input_hashes['protocol']
    assert [sha(p) for p in ref.cameras()]==read(ref.PRIOR/'evidence/calibration.json')['converted']
    model,cams = ref.prev.ref.common.setup_model()
    row = read(folder/'prepared.json')['items'][i]
    assert row['pair']==key
    frame = ref.FixedFeet(key,row,start,model,cams,read(folder/'protocol.json'))
    for eye in SIDES:
        name=key+'_'+eye
        assert sha(ref.OUT/'seg'/(name+'.npy'))==read(paired.palm.V4/'evidence/segmentation.json')['eyes'][name]['sampled_sha256']
    det_path=ref.OLD/'predictions.json'
    assert sha(det_path)==read(ref.PRIOR/'evidence/detection.json')['predictions_sha256']
    det=read(det_path)
    ch=np.array([41]+[n for a in range(21,41,4) for n in range(a+3,a-1,-1)])
    target=np.array([[np.asarray(det[key+'_'+eye+'.png']['xy'])[ch+shift] for shift in (21,0)] for eye in SIDES])
    inside=((target>=0)&(target<[1280,720])).all(-1)
    size=np.median(np.linalg.norm(target[:,:,ROOTS]-target[:,:,:1],axis=-1),axis=2)
    usable=inside & (np.linalg.norm(target-target[:,:,:1],axis=-1)<3.2*size[:,:,None])
    mano=read(MANO/'evidence/metrics.json')[key]
    meta=read(ARM_OUT/'palm/evidence'/(key+'.json'))
    qualities=torch.tensor([meta['metrics']['after'][s]['quality'] for s in SIDES])
    names=['wrist']+[d+suffix for d in DIGITS for suffix in ['1','2','3','']]
    ids=torch.tensor([[JOINT_NAMES.index(s+'_'+n) for n in names] for s in SIDES])
    for n,s in enumerate(SIDES):
        assert np.array_equal(target[:,n],mano[s]['point_check']['target'])
        assert np.array_equal(usable[:,n],mano[s]['point_check']['usable'])
        assert np.array_equal(target[:,n][:,[0]+ROOTS+TIPS].astype(np.float32),prior['results']['control']['hands'][s]['target'])
    obs=x0.new_tensor(target)
    with torch.no_grad():
        seed_j=paired.body(frame,x0)[1]
    def projected(j):
        pts=j[ids].reshape(-1,3)
        return torch.stack([paired.project(pts@eye.transform_t[:3,:3].T+eye.transform_t[:3,3],eye.cam).reshape(2,21,2) for eye in frame.eyes])
    results={}
    for mode in ('A','B'):
        points=ROOTS if mode=='A' else FINGER_JOINTS
        mask=torch.tensor(usable[:,:,points])
        movable=torch.tensor(paired.WRISTS+list(range(23,53)))
        free=x0[movable].clone().requires_grad_(True)
        opt=torch.optim.Adam([free],lr=.018)
        best_loss=float('inf')
        for step in range(paired.STEPS+1):
            x=x0.index_copy(0,movable,free)
            v,j=paired.body(frame,x)
            err=(projected(j)[:,:,points]-obs[:,:,points]).norm(dim=-1)
            robust=2*(torch.sqrt(1+(err/6)**2)-1)
            if mode=='A':
                point_term=robust.mean((0,2))
            else:
                point_term=(robust*mask).sum((0,2))/mask.sum((0,2)).clamp_min(1)
            edge=frame.observed(v,True)[0]
            loss=.01*(qualities*point_term).mean()+ref.WEIGHT*edge+.003*(x[paired.WRISTS]-x0[paired.WRISTS]).square().mean()+.0008*(x[23:]/.52).square().mean()
            assert torch.isfinite(loss)
            if loss.item()<best_loss:
                best_loss=loss.item()
                best=x.detach().clone()
                best_step=step
            if step<paired.STEPS:
                opt.zero_grad()
                loss.backward()
                opt.step()
                with torch.no_grad():
                    cap=free.new_tensor(np.deg2rad([100]*2+[45]*30))[:,None]
                    free.mul_((cap/free.norm(dim=1,keepdim=True).clamp_min(1e-8)).clamp_max(1.))
        with torch.no_grad():
            v,j=paired.body(frame,best)
            uv=projected(j).numpy()
            edge,per_eye=frame.observed(v,False)
        drift=float((j[:22]-seed_j[:22]).norm(dim=1).max()*1000)
        assert drift<.001
        dest=OUT/'poses'/(key+'_'+mode+'.npz')
        np.savez_compressed(dest,**dict(pack,locked=best.numpy()))
        rec=dict(pose_sha256=sha(dest),step=best_step,objective=best_loss,body_max_mm=drift,hands={})
        for n,side in enumerate(SIDES):
            h=prior['results']['control']['hands'][side]
            replay=float(np.max(abs(uv[:,n][:,[0]+ROOTS+TIPS]-h['uv'])))
            if mode=='A':
                assert replay<.02,(key,side,replay)
            rec['hands'][side]=dict(uv=uv[:,n].tolist(),observation=target[:,n].tolist(),inside=inside[:,n].tolist(),
                point_indices=points,used=(np.ones_like(usable[:,n,points]) if mode=='A' else usable[:,n,points]).tolist(),
                hand_edge_px=[d[side[0].upper()+'_hand'] for d in per_eye],delta_from_archived_A_px=replay)
        results[mode]=rec
        print(key,mode,best_step,flush=True)
    write(RUN/'evidence'/(key+'.json'),dict(results=results,seed_sha256=sha(seed),quality=qualities.tolist(),exit_status=0))

if __name__=='__main__':
    fit(int(sys.argv[1]))
