"""Render the archived selection in the same floor-aligned views for both arms."""
import sys
import cv2
import numpy as np
from compare import ROOT, RUN, OUT, base, read, write, sha

sys.path.insert(0,str(ROOT/'runs/2026-11-20_rw20000_frozen_mesh12'))
import gallery as g


def main():
    planes=read(ROOT/'runs/2026-11-20_rw20000_floor12/evidence/renders_floor.json')['rows']
    done=read(OUT/'fit_complete.json')['ids']
    params=dict(np.load(OUT/'budget_20000.npz'))
    model,_=base.common.setup_model()
    faces=np.asarray(model.face,np.int32)
    ref=read(g.RUN/'evidence/renders_rw20000.json')['rows']
    source=dict(np.load(base.OUT/'budget_20000.npz'))
    original_ids=read(base.OUT/'fit_complete.json')['ids']
    assert {r['pair'] for r in planes}=={r['pair'] for r in ref}
    records=[]
    for row in planes:
        p=row['pair']; k=original_ids.index([p,'rw20000','2d'])
        n=np.asarray(row['up'])
        f=(g.rot(source['delta'][k])@g.rot(source['global0'][k]))[:,2]
        f-=n*(f@n); f/=np.linalg.norm(f)
        x=np.cross(n,f)
        bases=[np.stack([x,n,f],1),np.stack([f,n,-x],1)]
        meshes=[]
        for mode in ['baseline','equal']:
            i=done.index([p,mode,'2d'])
            assert params['selected_step'][i]>=0
            par={a:params[b][i] for a,b in [('body_pose','body_pose'),('betas','betas'),('global_orient','global0'),('transl','translation')]}
            meshes.append(g.geometry(model.layer['neutral'],par,params['delta'][i],False,None)[0])
        scales=[]; centers=[]
        for basis in bases:
            both=np.concatenate(meshes)@basis
            centers.append((both.min(0)+both.max(0))/2)
            scales.append(min(315/np.ptp(both[:,0]),435/np.ptp(both[:,1])))
        panels=[]
        for j,b in enumerate(bases):
            panels.extend(g.free_render(v,faces,b,centers[j],min(scales)) for v in meshes)
        title=np.full((40,1400,3),246,np.uint8)
        for j,label in enumerate(['Grouped / front','Equal / front','Grouped / side','Equal / side']):
            cv2.putText(title,label,(12+350*j,27),0,.72,(55,55,55),1,cv2.LINE_AA)
        path=RUN/'assets'/p/'comparison.jpg'
        g.save(path,np.vstack([title,np.hstack(panels)]))
        records.append(dict(pair=p,subject=row['subject'],floor_valid=row['validation']['accepted'],images={path.name:sha(path)}))
    write(RUN/'evidence/renders_paired.json',dict(rows=records,command='<BODY_PYTHON> -Bu views.py',exit_status=0))


if __name__=='__main__':
    main()
