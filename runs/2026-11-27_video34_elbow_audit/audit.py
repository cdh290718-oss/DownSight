"""Matched elbow diagnostics and counterfactual shoulder/elbow fits."""
from pathlib import Path
import sys
import cv2
import numpy as np
import torch

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
PRIOR = RUN.parent/'2026-11-27_video34_uniform_v5'
sys.path.insert(0,str(PRIOR))
from base import OUT, read, write, sha, cameras
import finish_frame as original
from calibrate import projection
from render_selection import free_render, title

def units(v):
    return v/np.linalg.norm(v)

def angle(a,b):
    return float(np.rad2deg(np.arccos(np.clip(units(a)@units(b),-1,1))))

def pack_pose(layer,pack,x):
    return original.w.old.mesh(layer,x,pack)

def elbow_observations(row,names,pred,key):
    eyes=[]
    for n,side in enumerate(('left','right')):
        xy = pred[key+'_'+side+'.png']['xy']
        eyes.append([next(p['xy'][n] for p in row['points'] if p['name']==s+'_Elbow')
                     for s in names]+[xy[62],xy[41]])
    return np.asarray(eyes)

def replay(pack, target, mask, transforms, model, cams, row, key, wrist_weight):
    ref=original.w.ref
    start=pack['locked']
    init=dict(body_pose=start[:21],delta=start[21],translation=start[22],global0=pack['global0'],betas=pack['betas'])
    frame=ref.FixedFeet(key,row,init,model,cams,read(OUT/'fit/0/protocol.json'))
    seed=torch.tensor(start)
    mask=torch.tensor(mask)
    target=seed.new_tensor(target)
    move=torch.tensor([15,16,17,18])
    delta=torch.zeros_like(seed[move],requires_grad=True)
    optimizer=torch.optim.LBFGS([delta],max_iter=100,line_search_fn='strong_wolfe')
    def trial():
        return seed.index_copy(0,move,seed[move]+delta)
    def closure():
        optimizer.zero_grad()
        j=original.w.paired.body(frame,trial())[1][[18,19,20,21]]
        uv=torch.stack([original.w.paired.project(j@e.transform_t[:3,:3].T+e.transform_t[:3,3],e.cam) for e in frame.eyes])
        error=(uv-target).square().sum(-1)
        def group(sl):
            use=mask[:,sl]
            return (error[:,sl]*use).sum()/use.sum().clamp_min(1)
        loss=group(slice(2))+wrist_weight*group(slice(2,4))+25*delta.square().sum(1).mean()
        loss.backward()
        return loss
    optimizer.step(closure)
    return trial().detach().numpy()

def analyze():
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    ref=original.w.ref
    ref.OLD=ref.OUT=OUT
    ref.cameras=cameras
    ref.compare.OUT=OUT
    ref.prev.ref.camera_files=ref.prev.base.fit.camera_files=cameras
    model,cam_class=ref.prev.ref.common.setup_model()
    layer=model.layer['neutral'].eval()
    faces=np.asarray(model.face)
    weights=layer.lbs_weights.argmax(1).numpy()
    cams=list(map(read,cameras()))
    dp=OUT/'predictions.json'
    assert sha(dp)==read(PRIOR/'evidence/detection.json')['predictions_sha256']
    pred=read(dp)
    prepared=OUT/'fit/0/prepared.json'
    assert sha(prepared)==read(PRIOR/'evidence/inputs.json')['shards']['0']
    rows={}
    cache={}
    for row in read(prepared)['items']:
        key=row['pair']
        src=read(PRIOR/'evidence'/(key+'_render.json'))
        raw=OUT/'poses'/(key+'.npz')
        assert sha(raw)==src['source']['v4_sha256']
        palm=OUT/'palm/poses'/(key+'.npz')
        if src['source']['palm_fallback']:
            palm=raw
        assert sha(palm)==src['source']['palm_sha256']
        final=OUT/'poses'/(key+'_v5.npz')
        exported=OUT/'export'/src['obj']
        assert sha(exported)==src['obj_sha256']
        base=dict(np.load(raw)); moved=dict(np.load(palm)); finished=dict(np.load(final))
        moved['locked'][22]+=np.array(src['source']['purple_translation_m'])
        x=moved['locked'];y=finished['locked']
        before_hand=y.copy();before_hand[19:21]=x[19:21];before_hand[23:]=x[23:]
        transforms=[np.array(row['inputs']['views'][s]['camera']['left_to_view']) for s in ('left','right')]
        target=elbow_observations(row,['L','R'],pred,key)
        mask=((target>=0)&(target<[1280,720])).all(2)
        assert mask.ravel().tolist()==src['source']['valid_elbow_wrist_by_eye']
        assert np.array_equal(x[:15],y[:15]) and np.array_equal(x[21:23],y[21:23])
        calc=replay(moved,target,mask,transforms,model,cam_class,row,key,4)
        assert np.array_equal(calc,before_hand)
        stages={'v4':base['locked'],'purple_seed':x,
                'elbow_only':replay(moved,target,mask,transforms,model,cam_class,row,key,0),
                'equal_weight':replay(moved,target,mask,transforms,model,cam_class,row,key,1),
                'arm4':before_hand,'v5':y}
        geo={k:pack_pose(layer,finished,p) for k,p in stages.items()}
        from render_result import load_obj
        v,f=load_obj(exported)
        assert np.array_equal(f,faces) and np.max(abs(v-geo['v5'][0]))<1e-8
        cache[key]=(geo,original.w.old.mat(y[21])@original.w.old.mat(finished['global0']))
        points={}
        for i,side in enumerate(['left','right']):
            shoulder,elbow,wrist=[16+i,18+i,20+i]
            _,j0=geo['purple_seed']; v1,j1=geo['arm4']
            ids=(np.linalg.norm(v1-j1[elbow],axis=1)<.065)&np.isin(weights,[shoulder,elbow])
            assert ids.any()
            stages_out={}
            for name,(mesh,j) in geo.items():
                observed=np.stack([projection(j[[elbow,wrist]]@t[:3,:3].T+t[:3,3],c)
                                   for t,c in zip(transforms,cams)])
                gaps=np.linalg.norm(observed-target[:,[i,i+2]],axis=2)
                a=j[elbow]-j[shoulder];b=j[wrist]-j[elbow]
                stages_out[name]=dict(center_xyz_m=j[elbow].tolist(),elbow_gaps_px=gaps[mask[:,i],0].tolist(),
                    wrist_gaps_px=gaps[mask[:,i+2],1].tolist(),flexion_deg=angle(a,b),
                    forearm_shift_from_purple_deg=angle(b,j0[wrist]-j0[elbow]))
            d=float(1000*np.max(np.linalg.norm(geo['v5'][0][ids]-v1[ids],axis=1)))
            points[side]=dict(stages=stages_out,
                hand_stage_elbow_drift_mm=float(1000*np.linalg.norm(geo['v5'][1][elbow]-j1[elbow])),
                hand_stage_elbow_patch_max_mm=d,patch_vertices=int(ids.sum()))
        rows[key]=dict(arms=points,replayed_arm4_exact=True,pose_sha256=sha(final),obj_sha256=sha(exported),
                       palm_fallback=src['source']['palm_fallback'])
    render_and_write(rows,cache,faces)

def render_and_write(rows,cache,faces):
    arms=[(key,s,r) for key,frame in rows.items() for s,r in frame['arms'].items()]
    def difference(r,a,b):
        return abs(r['stages'][a]['flexion_deg']-r['stages'][b]['flexion_deg'])
    ranked=sorted(arms,key=lambda entry:difference(entry[2],'arm4','elbow_only'),reverse=True)
    picks=[next(x for x in ranked if x[0].startswith(tag)) for tag in ('v003','v004')]
    images={}
    for key,side,_ in picks:
        geo,root=cache[key]
        basis=root@original.w.old.mat([0,np.pi/2,0])
        if side=='right':
            basis=root@original.w.old.mat([0,-np.pi/2,0])
        labels=[('v4','Original v4'),('purple_seed','Purple seed'),
                ('elbow_only','Elbow only'),('arm4','Elbow + wrist x4'),('v5','Final Dv5')]
        transformed=np.vstack([v@basis for v,j in geo.values()])
        center=(transformed.min(0)+transformed.max(0))/2
        scale=min(316/np.ptp(transformed[:,0]),440/np.ptp(transformed[:,1]))
        panels=[]
        index=0 if side=='left' else 1
        for stage,label in labels:
            v,j=geo[stage]
            panel=free_render(v,faces,basis,center,scale)
            pts=np.rint(((j[[16+index,18+index,20+index]]@basis-center)[:,:2])*[scale,-scale]+[175,240]).astype(int)
            cv2.polylines(panel,[pts],False,(95,140,227),2,cv2.LINE_AA)
            for i,(a,b) in enumerate(pts):
                cv2.circle(panel,(a,b),5,(40,70,220) if i==1 else (125,95,40),-1,cv2.LINE_AA)
            panels.append(title(panel,label))
        path=RUN/'assets'/(key+'.jpg')
        assert cv2.imwrite(str(path),title(np.hstack(panels),key+'   '+side+' arm  |  side view'),[cv2.IMWRITE_JPEG_QUALITY,91])
        images[path.name]=sha(path)
    summary={}
    for name in next(iter(arms))[2]['stages']:
        both={k:[p for _,_,r in arms for p in r['stages'][name][k+'_gaps_px']] for k in ['elbow','wrist']}
        summary[name]={k+'_rmse_px':float(np.sqrt(np.mean(np.square(v)))) for k,v in both.items()}
    diffs=[difference(r,'arm4','elbow_only') for _,_,r in arms]
    write(RUN/'evidence/results.json',dict(rows=rows,summary=summary,images=images,n=len(arms),frames=len(rows),
        chosen=[(a,b) for a,b,_ in picks],hand_stage_max_elbow_mm=max(r['hand_stage_elbow_drift_mm'] for _,_,r in arms),
        hand_stage_max_patch_mm=max(r['hand_stage_elbow_patch_max_mm'] for _,_,r in arms),
        wrist_weight_4_vs_0_deg=dict(median=float(np.median(diffs)),max=float(max(diffs)),over_20=sum(d>20 for d in diffs)),exit_status=0))
if __name__=='__main__':analyze()
