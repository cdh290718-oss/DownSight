"""Two-frame constraint feasibility and visible-contour fitting, with label-free pose selection."""
import sys
from collections import Counter
import numpy as np
import torch
from common_run import (RUN,ROOT,OUT,EV,PAIRS,inputs,prev,search,Contact,groups,summary,
                        rotation,crossings)
from outline import Eye

WEIGHT=.008
STEPS=72
BACKTRACK=(1.,.5,.25,.0625)


class Gate:
    def __init__(self,faces,labels,rest):
        self.faces=faces
        self.labels=labels
        self.calls=0
        self.raw_pass=0
        original=crossings(rest,faces)
        self.template=summary(original,labels)
        self.zones={}
        # Restrict the exception to same-region pairs inside the template's original crossing neighbourhood.
        for part in np.unique(labels[original.ravel()]):
            h=original[(labels[original[:,0]]==part)&(labels[original[:,1]]==part)]
            zone=np.zeros(len(faces),bool)
            zone[h.ravel()]=True
            for _ in range(2):
                zone=(labels==part)&np.isin(faces,faces[zone]).any(1)
            self.zones[part]=zone

    def check(self,v):
        hits=crossings(v,self.faces)
        self.calls+=1
        self.raw_pass+=int(not len(hits))
        keep=np.ones(len(hits),bool)
        for zone in self.zones.values():
            keep &= ~(zone[hits[:,0]]&zone[hits[:,1]])
        return hits,hits[keep]


class Frame:
    def __init__(self,pair,item,orig,model,cam,cfg):
        self.pair=pair
        p=prev.base.trial.TorsoOnly([dict(item,condition='baseline')],cfg,model,cam)
        self.p=p
        self.layer=model.layer['neutral']
        self.faces=np.asarray(model.face,dtype=np.int32)
        self.labels=groups(self.layer,self.faces)
        self.gate=Gate(self.faces,self.labels,self.layer.v_template.detach().numpy())
        self.contact=Contact(self.layer,self.faces)
        self.target='L_hand/L_thigh' if pair=='pair055' else 'L_hand/R_hand'
        self.eyes=[Eye(item,side,camera,prev.read(path)['input_adapter'],self.labels,self.faces)
                   for side,camera,path in zip(['left','right'],p.cameras,prev.ref.camera_files())]
        self.g=torch.tensor(orig['global0'],dtype=torch.float32).reshape(1,3)
        self.beta=torch.zeros(1,10)
        self.v3=torch.tensor(np.concatenate((orig['body_pose'],orig['delta'].reshape(1,3),
                        orig['translation'].reshape(1,3),np.zeros((30,3))),axis=0),dtype=torch.float32)
        self.cold=torch.zeros_like(self.v3)
        self.cold[22]=p.tr.detach().reshape(3)
        assert torch.allclose(self.g,p.global0,atol=1e-5)
        self.logs=[]

    def state(self,x):
        b=x[:21].reshape(1,63)
        hand=x[23:].reshape(2,45)
        full=self.layer(global_orient=self.g,body_pose=b,betas=self.beta,
                        left_hand_pose=hand[:1],right_hand_pose=hand[1:],
                        jaw_pose=self.beta[:,:3],leye_pose=self.beta[:,:3],reye_pose=self.beta[:,:3],expression=self.beta)
        j=full.joints[0]
        v=(full.vertices[0]-j[:1])@rotation(x[21]).T+j[:1]+x[22]
        native,rh=self.p.fk(b.reshape(1,21,3),self.beta,self.g,x[21:22],x[22:23])
        self.p.bodyvar=b.reshape(1,21,3)[:,self.p.body_ids]
        loss,ok,_=self.p.objective((native,rh,self.beta))
        fingerprior=.0008*(hand/.52).square().mean()
        return v,native,loss[0]+fingerprior,bool(ok.item())

    def cap(self,x):
        xx=x.clone()
        ids=self.p.body_ids
        for idx,limit in [(ids,self.p.caps[:,None]),(np.arange(23,53),np.deg2rad(30.))]:
            a=xx[idx]
            xx[idx]=a*torch.clamp(torch.as_tensor(limit)/a.norm(dim=1,keepdim=True).clamp_min(1e-8),max=1.)
        xx[[11,14]]=0.
        return xx

    def observed(self,v,grad=False):
        values,metrics=zip(*(eye.compare(v,grad) for eye in self.eyes))
        return sum(values)/2,metrics

    def describe(self,x):
        with torch.no_grad():
            v,joint,raw,ok=self.state(x)
            outline,details=self.observed(v)
            hits,active=self.gate.check(v.numpy())
            contacts=self.contact.measure(v.numpy())
            return dict(base_loss=raw.item(),outline_loss=outline.item(),joint_feasible=ok,
                        outline_px_by_eye=details,raw_triangles=len(hits),active_triangles=len(active),
                        crossings=summary(hits,self.labels),active_crossings=summary(active,self.labels),
                        target_depth_gap_mm=contacts[self.target],all_contact_mm=contacts)

    def seedbank(self):
        # Repair infeasible v3 by attenuation in related chains. The unmodified v3 and neutral
        # are both examined. No reference joints or test-body labels enter this stage.
        groups_idx=[[0,1,3,4,6,7,9,10],[2,5,8],[12,13,15,16],[17,18,19,20]]
        yield 'v3',self.v3
        yield 'cold',self.cold
        for alpha in [.98,.94,.88,.8,.68,.5,.3,0.]:
            seed=self.v3.clone()
            seed[:21]*=alpha
            yield 'scale'+str(alpha),seed
        rng=np.random.default_rng(6491)
        for i in range(72):
            seed=self.v3.clone()
            for idx in groups_idx:
                seed[idx]*=rng.choice([.45,.65,.8,.9,.98,1.])
            yield 'chain'+str(i),seed

    def repair(self):
        seeds=[]
        stats=Counter()
        for tag,x in self.seedbank():
            with torch.no_grad():
                v,_,raw,ok=self.state(x)
                all_hits,active=self.gate.check(v.numpy())
                stats['tried']+=1
                stats['raw_zero']+=int(not len(all_hits) and ok)
                if len(active) or not ok:continue
                outline=self.observed(v)[0].item()
                seeds.append((tag,x.clone(),raw.item(),outline))
        stats['template_exception_feasible']=len(seeds)
        assert seeds
        return seeds,dict(stats)

    def descend(self,seed,hard,outline,tag):
        x=seed.detach().clone().requires_grad_(True)
        opt=torch.optim.Adam([x],lr=.009)
        best=x.detach().clone()
        bestloss=float('inf')
        backs=Counter()
        blocked=0
        for step in range(STEPS+1):
            opt.zero_grad()
            v,_,loss,ok=self.state(x)
            if outline:loss=loss+WEIGHT*self.observed(v,True)[0]
            assert torch.isfinite(loss)
            if ok and loss.item()<bestloss:
                bestloss=loss.item()
                best=x.detach().clone()
            if step==STEPS:break
            loss.backward()
            # Pose, wrist and fingers move independently. Translation is in metres.
            old=x.detach().clone()
            opt.step()
            with torch.no_grad():
                diff=x-old
                diff[22]*=.20
                accepted=False
                for scale in BACKTRACK:
                    trial=self.cap(old+scale*diff)
                    verts,_,_,positive=self.state(trial)
                    if positive and (not hard or not len(self.gate.check(verts.numpy())[1])):
                        x.copy_(trial)
                        backs[str(scale)]+=1
                        accepted=True
                        break
                if not accepted:
                    x.copy_(old)
                    blocked+=1
                    # Try disjoint updates when one active collision blocks the full Adam step.
                    for indices in [[19,20]+list(range(23,53)),[15,16,17,18],[0,1,3,4,6,7],[2,5,8,12,13],[21,22]]:
                        trial=x.clone()
                        trial[indices]+=diff[indices]*.5
                        trial=self.cap(trial)
                        verts,_,_,positive=self.state(trial)
                        if positive and not len(self.gate.check(verts.numpy())[1]):
                            x.copy_(trial)
                            backs['block']+=1
        self.logs.append(dict(seed=tag,hard=hard,outline=outline,steps=STEPS,best_loss=bestloss,
                              accepted_scales=dict(backs),blocked_steps=blocked))
        return best,bestloss

    def solve(self):
        choices={'v3':self.v3}
        seeds,stats=self.repair()
        for mode,hard,contour in [('outline',False,True),('hard',True,False),('joint',True,True)]:
            if hard:
                start=min(seeds,key=lambda s:s[2]+(WEIGHT*s[3] if contour else 0))
                starts=[(start[0],start[1])]
                if start[0]!='cold' and any(s[0]=='cold' for s in seeds):starts.append(('cold',self.cold))
            else:starts=[('v3',self.v3)]
            bestloss=float('inf')
            for tag,seed in starts:
                result,loss=self.descend(seed,hard,contour,tag)
                if loss<bestloss:choices[mode],bestloss=result,loss
            print(self.pair,mode,flush=True)
        rows={k:self.describe(x) for k,x in choices.items()}
        for name in ['hard','joint']:assert rows[name]['active_triangles']==0
        np.savez_compressed(OUT/(self.pair+'_states.npz'),**{k:x.numpy() for k,x in choices.items()})
        return dict(rows=rows,repair=stats,optimizer=self.logs,terminal_raw_zero={k:v['raw_triangles']==0 for k,v in rows.items()},
                    geometry_checks=self.gate.calls,raw_zero_checks=self.gate.raw_pass,template=self.gate.template,
                    exemptions={k:int(z.sum()) for k,z in self.gate.zones.items()},
                    view_support=[eye.support for eye in self.eyes],state_sha256=prev.sha(OUT/(self.pair+'_states.npz')))


def main(pair):
    from pathlib import Path
    def guard(event,args):
        if event=='open' and args and isinstance(args[0],str):
            if Path(args[0]).name in ('reference.json','validation_truth.json','samples.npz','test_manifest.json'):
                raise PermissionError('Body evaluation labels are excluded during selection')
    sys.addaudithook(guard)
    torch.set_num_threads(1)
    model,cam,cfg,data=inputs()
    item,orig=data[pair]
    frame=Frame(pair,item,orig,model,cam,cfg)
    rec=frame.solve()
    prev.write(EV/(pair+'_selection.json'),dict(rec,exit_status=0))


if __name__=='__main__':
    main(sys.argv[1])
