"""Search for feasible starts near v3; only independently gated full states leave repair."""
from collections import defaultdict,deque
from pathlib import Path
import sys
import numpy as np
import torch
from scipy.spatial import cKDTree
from common_run import EV,OUT,inputs,prev
from fit_probe import Frame


def guard(event,args):
    if event=='open' and args and isinstance(args[0],str):
        if Path(args[0]).name in ('reference.json','validation_truth.json','samples.npz','test_manifest.json'):
            raise PermissionError('Body evaluation labels are excluded during repair')


def repulsion(v,f,labels,hits):
    by_part=defaultdict(list)
    for a,b in hits:
        by_part[tuple(sorted([labels[a],labels[b]]))].append([a,b])
    costs=[]
    for pairs in by_part.values():
        ids=np.array(pairs,dtype=int)
        a,b=[v[torch.from_numpy(f[ids[:,i]].astype(np.int64))] for i in [0,1]]
        def side(x,y):
            normal=torch.nn.functional.normalize(torch.cross(x[:,1]-x[:,0],x[:,2]-x[:,0],dim=1),dim=1).detach()
            signed=((y-x.mean(1)[:,None,:])*normal[:,None,:]).sum(2)
            return torch.relu((.0015-signed)/.01).square().mean()
        costs.append((side(a,b)+side(b,a))/2)
    return sum(costs)


def distance_between(v,left,right):
    points=v.detach().numpy()
    ds,near=cKDTree(points[right]).query(points[left])
    i=ds.argmin()
    return (v[left[i]]-v[right[near[i]]]).norm()


def main(pair):
    sys.addaudithook(guard)
    torch.set_num_threads(1)
    model,cam,cfg,data=inputs()
    frame=Frame(pair,*data[pair],model,cam,cfg)
    ref=np.load(OUT/(pair+'_states.npz'))
    target=[np.unique(frame.faces[frame.labels==part]) for part in frame.target.split('/')]
    saved,history={},{}
    for start in ['v3','outline','hard']:
        initial=torch.from_numpy(ref[start])
        x=initial.clone().requires_grad_(True)
        optimizer=torch.optim.Adam([x],lr=.009)
        recent=deque(maxlen=5)
        record=[]
        for step in range(121):
            optimizer.zero_grad()
            verts,_,base,positive=frame.state(x)
            all_hits,active=frame.gate.check(verts.detach().numpy())
            raw_gap=distance_between(verts,*target)
            if not len(active) and positive and step%2==0:
                saved[start+'_'+str(step)]=x.detach().clone().numpy()
            record.append([len(all_hits),len(active),base.item(),raw_gap.item()*1000])
            recent.append(active)
            remembered=np.unique(np.vstack(list(recent)),axis=0)
            # Initial v3 has either hand-hand or hand-thigh contact; preserve its proximity in the repair proposal.
            cost=base+.006*torch.relu((raw_gap-.004)/.01).square()+.0005*(x[:21]-frame.v3[:21]).square().mean()
            if len(remembered):cost=cost+(.04 if step<60 else .12)*repulsion(verts,frame.faces,frame.labels,remembered)
            if step==120:break
            cost.backward()
            old=x.detach().clone()
            optimizer.step()
            with torch.no_grad():
                x[22]=old[22]+.15*(x[22]-old[22])
                x.copy_(frame.cap(x))
        history[start]=record
        print(pair,start,'minimum collisions',min(v[1] for v in record),'states',len(saved),flush=True)
    path=OUT/(pair+'_repair.npz')
    np.savez_compressed(path,**saved)
    prev.write(EV/(pair+'_repair.json'),dict(fields=['raw_crossings','outside_zones_crossings','base_loss','nearest_target_vertices_mm'],
            history=history,feasible_starts=len(saved),parameter_sha256=prev.sha(path),exit_status=0))


if __name__=='__main__':main(sys.argv[1])
