"""Matched wrist/finger refinement with a common start and optional MANO silhouettes."""
from copy import copy
from pathlib import Path
import sys
import cv2
import numpy as np
import torch

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
OUT = ROOT/'outputs/realvideo4_wilor_outline_20261126'
PREV = RUN.parent/'2026-11-26_realvideo4_palm'
sys.path.insert(0,str(PREV))
import palm
from common_run import rotation, project
from outline import boundary
from smplx.joint_names import JOINT_NAMES
ref, read, write, sha = palm.ref, palm.read, palm.write, palm.sha
SIDES = ['left','right']
WRISTS = [19,20]
MODES = ['control','wilor']
STEPS = 70
WILOR_WEIGHT = .012


def visible_targets(frame,key):
    result = []
    evidence = {}
    masks = read(RUN/'evidence/masks.json')['views']
    archived = read(palm.V4/'evidence/segmentation.json')['eyes']
    for eye,base in zip(SIDES,frame.eyes):
        name = key+'_'+eye
        seg_path = ref.OUT/'seg'/(name+'.npy')
        assert sha(seg_path) == archived[name]['sampled_sha256']
        labels = np.load(seg_path).argmax(0)
        for ident,side in zip([6,15],['L','R']):
            target_path = OUT/'masks'/(name+'_'+side+'.png')
            assert sha(target_path) == masks[name+'_'+side]['mask_sha256']
            mask = cv2.imread(str(target_path),0)[2::4,2::4]>0
            outline = boundary(mask)
            # Allow 12 pixels of semantic-boundary uncertainty. Hidden portions
            # unsupported by the image segmentation cannot attract the model.
            nearby = cv2.dilate((labels==ident).astype('uint8'),np.ones((7,7),np.uint8))>0
            used = outline & nearby
            fraction = used.sum()/max(1,outline.sum())
            label = side+'_hand'
            weight = float(fraction) if used.sum()>=6 and label in base.targets else 0.
            evidence[name+'_'+side]=dict(used=int(used.sum()),total=int(outline.sum()),weight=weight)
            if weight:
                view = copy(base)
                view.targets = {label:base.pix[used].astype(float)}
                result.append((view,weight))
    return result,evidence


def body(frame,x):
    flat = x[23:].reshape(2,45)
    m = frame.layer(global_orient=frame.g,body_pose=x[:21].reshape(1,63),betas=frame.beta,
                    left_hand_pose=flat[:1],right_hand_pose=flat[1:],jaw_pose=frame.beta[:,:3],
                    leye_pose=frame.beta[:,:3],reye_pose=frame.beta[:,:3],expression=frame.beta)
    root = m.joints[0,:1]
    def convert(points):
        return (points-root)@rotation(x[21]).T+root+x[22]
    return convert(m.vertices[0]),convert(m.joints[0])


def run(i):
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    key = f'clip{i+1:03}'
    old = read(PREV/'evidence'/(key+'.json'))
    ref.prev.ref.camera_files=ref.prev.base.fit.camera_files=ref.cameras
    ref.compare.OUT=ref.OUT
    base = palm.OUT/'poses'/(key+'.npz')
    assert sha(base)==old['pose_hash']
    pack = dict(np.load(base))
    x0 = torch.from_numpy(pack['locked'])
    assert not x0[23:].any()
    start = dict(body_pose=x0[:21].numpy(),delta=x0[21].numpy(),translation=x0[22].numpy(),betas=pack['betas'],global0=pack['global0'])
    cfg=read(ref.OLD/'fit/0/protocol.json')
    row=read(ref.OLD/'fit/0/prepared.json')['items'][i]
    input_audit=read(ref.PRIOR/'evidence/inputs.json')
    assert sha(ref.OLD/'fit/0/protocol.json')==input_audit['protocol'] and sha(ref.OLD/'fit/0/prepared.json')==input_audit['shards']['0']
    assert [sha(p) for p in ref.cameras()]==read(ref.PRIOR/'evidence/calibration.json')['converted']
    model,cam=ref.prev.ref.common.setup_model()
    frame=ref.FixedFeet(key,row,start,model,cam,cfg)
    target_views,visibility=visible_targets(frame,key)
    det_path=ref.OLD/'predictions.json'
    assert sha(det_path)==read(ref.PRIOR/'evidence/detection.json')['predictions_sha256']
    det=read(det_path)
    channels = [[62,49,53,57,61],[41,28,32,36,40]]
    digits = ['thumb','index','middle','ring','pinky']
    ids = [[JOINT_NAMES.index(side+'_'+k) for k in ['wrist']+[d+'1' for d in digits[1:]]+digits] for side in SIDES]
    obs=torch.tensor(np.array([[np.asarray(det[key+'_'+eye+'.png']['xy'])[ch+list(np.array([21,25,29,33,37])+shift)] for ch,shift in zip(channels,[21,0])] for eye in SIDES]),dtype=x0.dtype)
    qualities=torch.tensor([old['metrics']['after'][side]['quality'] for side in SIDES])
    with torch.no_grad():
        ref_joints=body(frame,x0)[1]
    def evaluate(x,with_grad,use_wilor):
        v,j = body(frame,x)
        projections=[]
        for view in frame.eyes:
            points = j[torch.tensor(ids)]
            xyz=points.reshape(-1,3)@view.transform_t[:3,:3].T+view.transform_t[:3,3]
            projections.append(project(xyz,view.cam).reshape(2,10,2))
        uv=torch.stack(projections)
        dist=(uv-obs).norm(dim=-1)
        roots=(2*(torch.sqrt(1+(dist[:,:,1:5]/6)**2)-1)).mean((0,2))
        seg,per_eye=frame.observed(v,with_grad)
        wil=v.new_zeros(())
        if use_wilor or not with_grad:
            for view,weight in target_views:
                wil=wil+weight*view.compare(v,with_grad)[0]/4
        wrist_reg=(x[WRISTS]-x0[WRISTS]).square().mean()
        finger_reg=(x[23:]/.52).square().mean()
        loss=.01*(qualities*roots).mean()+ref.WEIGHT*seg+.003*wrist_reg+.0008*finger_reg
        if use_wilor:
            loss=loss+WILOR_WEIGHT*wil
        if with_grad:
            return loss
        rows={}
        offset=(j[:22]-ref_joints[:22]).norm(dim=1).max().item()*1000
        assert offset<.002
        for n,side in enumerate(SIDES):
            label=side[0].upper()+'_hand'
            def errs(sl):
                return (dist[:,n,sl].square().mean().sqrt().item())
            q=j[ids[n][:5]].detach().numpy()
            observed=palm.triangulate(obs[:,n,:5].numpy(),list(map(read,ref.cameras())),frame.eyes[1].transform)
            rows[side]=dict(wrist_px=errs(0),root_px=errs(slice(1,5)),tips_px=errs(slice(5,10)),thumbtip_px=errs(5),pinkytip_px=errs(9),
                normal_vs_stereo=palm.angle(palm.normal(q,side),palm.normal(observed,side)),
                segment_edge_px=[d.get(label) for d in per_eye],uv=uv[:,n].detach().numpy().tolist(),
                target=obs[:,n].tolist(),tip_gaps=dist[:,n,5:].tolist())
        return dict(objective=float(loss),segment_edge_px=float(seg)*12,wilor_weighted_edge_px=float(wil)*12,body_joint_max_mm=offset,hands=rows)
    results={}
    for mode in MODES:
        enabled=mode=='wilor'
        movable=WRISTS+list(range(23,53))
        free=x0[movable].clone().requires_grad_(True)
        optimizer=torch.optim.Adam([free],lr=.018)
        best_loss=float('inf')
        for step in range(STEPS+1):
            x=x0.index_copy(0,torch.tensor(movable),free)
            loss=evaluate(x,True,enabled)
            assert torch.isfinite(loss)
            if loss.item()<best_loss:
                best_loss=loss.item()
                best=x.detach().clone()
                best_step=step
            if step<STEPS:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    cap=free.new_tensor([np.deg2rad(100.)]*2+[np.deg2rad(45.)]*30)[:,None]
                    free.mul_((cap/free.norm(dim=1,keepdim=True).clamp_min(1e-8)).clamp_max(1.))
        with torch.no_grad():
            rec=evaluate(best,False,enabled)
            assert abs(rec['objective']-best_loss)<1e-5
            if 'start' not in results:
                results['start']=evaluate(x0,False,False)
        path=OUT/'poses'/(key+'_'+mode+'.npz')
        np.savez_compressed(path,locked=best.numpy(),betas=pack['betas'],global0=pack['global0'])
        results[mode]=dict(**rec,step=best_step,pose_sha256=sha(path))
        print(key,mode,best_step,best_loss,flush=True)
    write(RUN/'evidence'/(key+'.json'),dict(results=results,visibility=visibility,start_sha256=sha(base),exit_status=0))

if __name__=='__main__':
    run(int(sys.argv[1]))
