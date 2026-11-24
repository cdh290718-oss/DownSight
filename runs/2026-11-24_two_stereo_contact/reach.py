"""Two-hand contact rescue: transfer observed-fit wrist targets onto a feasible trunk by two-link IK."""
import sys
import numpy as np
import torch
from itertools import product
from scipy.spatial.transform import Rotation
from common_run import EV,OUT,inputs,prev,search
from fit_probe import Frame
from repair_local import guard


def globals_of(frame,x):
    rot=[search.rot(x[21].numpy())@search.rot(frame.g.numpy()[0])]
    for i in range(1,22):
        rot.append(rot[frame.p.fk.parents[i]]@search.rot(x[i-1].numpy()))
    return rot


def arm(frame,x,joint,world,reference_world,side,target):
    s,e,w=search.CHAINS[side]
    shoulder,elbow,wrist=joint[[s,e,w]]
    limb1,limb2=np.linalg.norm(elbow-shoulder),np.linalg.norm(wrist-elbow)
    axis=search.unit(target-shoulder)
    length=np.clip(np.linalg.norm(target-shoulder),abs(limb1-limb2)+.0001,limb1+limb2-.0001)
    endpoint=shoulder+axis*length
    along=(limb1**2-limb2**2+length**2)/(2*length)
    plane=search.unit(elbow-shoulder-axis*(axis@(elbow-shoulder)))
    bank=[]
    for theta in [-90,-45,0,45,90]:
        bend=Rotation.from_rotvec(axis*np.deg2rad(theta)).apply(plane)
        mid=shoulder+axis*along+bend*np.sqrt(max(0.,limb1**2-along**2))
        r1=search.turns((elbow-shoulder)[None],(mid-shoulder)[None])[0]
        rs=r1@world[s]
        r2=search.turns((r1@(wrist-elbow))[None],(endpoint-mid)[None])[0]
        re=r2@r1@world[e]
        locals_=np.stack([search.rv(world[frame.p.fk.parents[s]].T@rs),
                          search.rv(rs.T@re),search.rv(re.T@reference_world[w])])
        bank.append((theta,locals_))
    return bank


def main():
    sys.addaudithook(guard)
    pair='pair068'
    torch.set_num_threads(1)
    model,cam,cfg,data=inputs()
    frame=Frame(pair,*data[pair],model,cam,cfg)
    saved=dict(np.load(OUT/(pair+'_states.npz')))
    reference=torch.from_numpy(saved['outline'])
    refworld=globals_of(frame,reference)
    with torch.no_grad():
        rv,rj,_,_=frame.state(reference)
        hands=[np.unique(frame.faces[frame.labels==side+'_hand']) for side in ['L','R']]
        separation=search.unit((rv[hands[0]].mean(0)-rv[hands[1]].mean(0)).numpy())
        wrists=rj[0,[20,21]].numpy()
        centers=np.stack([rv[h].mean(0).numpy() for h in hands])
        midpoint=wrists.mean(0)
        offsets=wrists-midpoint
        accepted=[]
        log={}
        for name in ['hard','joint']:
            seed=torch.from_numpy(saved[name])
            _,j,_,_=frame.state(seed)
            world=globals_of(frame,seed)
            feasible=0
            attempts=0
            nearest=1000.
            for shift in [np.zeros(3),np.array([0.,0.,.04]),np.array([0.,0.,-.04])]:
                for spread in [0.,.012,.025,.04]:
                    wt=midpoint+shift+offsets+np.stack([separation,-separation])*spread/2
                    banks=[arm(frame,seed,j[0].numpy(),world,refworld,i,wt[i]) for i in range(2)]
                    for (la,left),(ra,right) in product(*banks):
                        trial=seed.clone()
                        trial[[15,17,19]]=torch.tensor(left,dtype=trial.dtype)
                        trial[[16,18,20]]=torch.tensor(right,dtype=trial.dtype)
                        trial[23:]=reference[23:]
                        trial=frame.cap(trial)
                        v,_,raw,positive=frame.state(trial)
                        attempts+=1
                        hits,active=frame.gate.check(v.numpy())
                        if len(active) or not positive:continue
                        feasible+=1
                        gap=frame.contact.measure(v.numpy())[frame.target][1]
                        nearest=min(nearest,gap)
                        objective=raw.item()+.008*frame.observed(v)[0].item()
                        accepted.append((gap,objective,raw.item(),name+'_'+str(attempts),trial.numpy()))
            log[name]=dict(attempts=attempts,feasible=feasible,minimum_gap_capped_mm=nearest if feasible else None)
            print(name,log[name],flush=True)
        # Retain winners of each fixed ranking as well as their nearest-contact alternatives.
        keep={}
        for fun in [lambda c:c[1],lambda c:c[2],lambda c:(c[0],c[1]),lambda c:(c[0]>8,c[1])]:
            keep.update({c[3]:c[4] for c in sorted(accepted,key=fun)[:8]})
        path=OUT/(pair+'_reach.npz')
        np.savez_compressed(path,**keep)
        prev.write(EV/'reach.json',dict(pair=pair,counts=log,kept=len(keep),point_source='outline-only single-frame model wrists',
                   parameter_sha256=prev.sha(path),exit_status=0))


if __name__=='__main__':main()
