"""Fit just one wrist rotation per frozen WiLoR candidate and select by non-tip error."""
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import least_squares
from wrist_probe import RUN, old, CASES, SIDES, DIGITS, TIPS, JOINT_NAMES, sha, read, write, ARM_OUT
from compare_fit import FINGER_JOINTS,ROOTS

def main():
    torch.set_num_threads(1)
    model,_=old.ref.prev.ref.common.setup_model()
    layer=model.layer['neutral'].eval()
    cams=list(map(read,old.ref.cameras()))
    rows={r['pair']:r for r in read(old.ref.OLD/'fit/0/prepared.json')['items']}
    detfile=old.ref.OLD/'predictions.json'
    assert sha(detfile)==read(old.ref.PRIOR/'evidence/detection.json')['predictions_sha256']
    det=read(detfile)
    order=np.array([41]+[n for a in range(21,41,4) for n in range(a+3,a-1,-1)])
    names=['wrist']+[d+s for d in DIGITS for s in ('1','2','3','')]
    held={k:[] for k in ('selected','same_source')}
    metrics={}
    for key in CASES:
        file=ARM_OUT/'palm/poses'/(key+'.npz')
        assert sha(file)==read(old.BASE/'evidence'/(key+'.json'))['seed_sha256']
        p=dict(np.load(file))
        x0=p['locked']
        world=[old.mat(x0[21])@old.mat(p['global0'].reshape(3))]
        for n in range(1,22):
            world.append(world[int(layer.parents[n])]@old.mat(x0[n-1]))
        t=[np.asarray(rows[key]['inputs']['views'][eye]['camera']['left_to_view']) for eye in SIDES]
        def project(pos):
            return np.stack([old.projection(pos@tr[:3,:3].T+tr[:3,3],c) for tr,c in zip(t,cams)])
        metrics[key]={}
        for i,side in enumerate(SIDES):
            w=20+i
            ids=np.array([JOINT_NAMES.index(side+'_'+n) for n in names])
            target=np.stack([np.asarray(det[key+'_'+eye+'.png']['xy'])[order+(21 if i==0 else 0)] for eye in SIDES])
            inside=((target>=0)&(target<[1280,720])).all(-1)
            width=np.median(np.linalg.norm(target[:,ROOTS]-target[:,:1],axis=2),axis=1)
            valid=(inside&(np.linalg.norm(target-target[:,:1],axis=2)<3.2*width[:,None]))[:,FINGER_JOINTS]
            assert valid.any(1).all()
            candidates={}
            for source in ('cam3','cam4'):
                file=old.OUT/'poses'/(key+'_'+source+'_B_wrist.npz')
                assert sha(file)==read(old.RUN/'evidence'/(key+'.json'))['results'][source+'_B_wrist']['pose_sha256']
                x=x0.copy()
                finger=slice(23+15*i,38+15*i)
                x[finger]=np.load(file)['locked'][finger]  # only mapped WiLoR fingers
                _,j=old.mesh(layer,x,p)
                pivot=j[w]
                local=(j[ids]-pivot)@world[w]
                parent=world[18+i]
                initial=old.mat(x[w-1])
                def err(r):
                    pos=local@(parent@old.mat(r)).T+pivot
                    return project(pos)-target
                def fun(r):
                    delta=err(r)[:,FINGER_JOINTS]/6
                    return np.r_[delta[valid].ravel(),old.rv(initial.T@old.mat(r))/.65]
                sol=least_squares(fun,x[w-1],loss='soft_l1',max_nfev=180)
                assert sol.success
                x[w-1]=old.rv(old.mat(sol.x))
                _,jfit=old.mesh(layer,x,p)
                uv=project(jfit[ids])
                # SMPL-X fingertip mesh vertices include pose blend shapes. Use actual forward for scoring/evaluation.
                dist=np.linalg.norm(uv-target,axis=-1)
                rob=2*(np.sqrt(1+(dist[:,FINGER_JOINTS]/6)**2)-1)
                score=float(np.mean([z[m].mean() for z,m in zip(rob,valid)]))
                candidates[source]=dict(score=score,wrist_rotation=x[w-1].tolist(),tip_errors_px=dist[:,TIPS][inside[:,TIPS]].tolist(),
                                        wrist_drift_mm=float(np.linalg.norm(jfit[w]-pivot)*1000),evaluations=sol.nfev)
            winner=min(candidates,key=lambda c:candidates[c]['score'])
            bpath=old.BASE_OUT/'poses'/(key+'_B.npz')
            assert sha(bpath)==read(old.BASE/'evidence'/(key+'.json'))['results']['B']['pose_sha256']
            b=np.load(bpath)['locked'][w-1]
            prev=read(RUN.parent/'2026-11-27_stereo_hand_selection/evidence/selection.json')[key]['hands'][side]['source']
            for tag,s in [('selected',winner),('same_source',prev)]:
                held[tag].extend(candidates[s]['tip_errors_px'])
            for c in candidates.values():
                c['wrist_difference_to_B_deg']=float(np.rad2deg(np.linalg.norm(old.rv(old.mat(b).T@old.mat(c['wrist_rotation'])))))
            metrics[key][side]=dict(winner=winner,B_source=prev,candidates=candidates)
    result=dict(cases=metrics,pooled={k:dict(n=len(v),rmse_px=old.rms(v)) for k,v in held.items()},exit_status=0)
    write(RUN/'evidence/fixed_fingers.json',result)
    print(result['pooled'])
if __name__=='__main__':
    main()
