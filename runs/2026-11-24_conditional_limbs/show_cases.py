"""Render V3 and the locked re-ranked poses in the same two gallery views."""
import sys
import cv2
import numpy as np
from select_pose import OUT, ROOT, RUN, prev, search
sys.path.insert(0,str(ROOT/'runs/2026-11-20_rw20000_floor12'))
import render_gallery as render


def main():
    folder=ROOT/'outputs/fixed_floor24_20261123'
    selected=dict(np.load(OUT/'validation_selected.npz'))
    decisions=list(prev.read(OUT/'validation_decisions.json'))
    choices=prev.read(search.OUT/'validation_choices.json')
    bank=dict(np.load(search.OUT/'validation_bank.npz'))
    model,_=prev.ref.common.setup_model()
    layer=model.layer['neutral']
    records={}
    for key in ['val031','val055']:
        info=prev.read(folder/(key+'.json'))
        up=np.array(info['ground']['up'])
        fwd=np.array(info['front_forward_left_cv'])
        z=render.floor.unit(fwd-up*(up@fwd))
        x=render.floor.unit(np.cross(up,z))
        bases=[np.column_stack([x,up,z]),np.column_stack([z,up,-x])]
        r=choices[key]
        cases=[(bank,r['offset']+r['best']),(selected,decisions.index(key))]
        verts=[]
        for pack,i in cases:
            p={k:pack[n][i] for k,n in [('body_pose','body_pose'),('global_orient','global0'),('transl','translation'),('betas','betas')]}
            v,_,_,_=render.g.geometry(layer,p,pack['delta'][i],False,None)
            verts.append(v)
        views=[np.concatenate(verts)@b for b in bases]
        scale=min(min(313/np.ptp(v[:,0]),438/np.ptp(v[:,1])) for v in views)
        canvas=[]
        for tag,v in zip(['Golden D v3','RW direction prior'],verts):
            for title,b,coords in zip(['front','side'],bases,views):
                center=(coords.min(0)+coords.max(0))/2
                im=render.g.free_render(v,np.asarray(model.face),b,center,scale)
                im=cv2.copyMakeBorder(im,38,2,0,0,cv2.BORDER_CONSTANT,value=(245,245,245))
                cv2.putText(im,tag+' / '+title,(12,24),cv2.FONT_HERSHEY_SIMPLEX,.66,(40,40,40),1,cv2.LINE_AA)
                canvas.append(im)
        path=RUN/'assets'/key/'comparison.jpg'
        render.g.save(path,np.concatenate(canvas,axis=1))
        records[key]=prev.sha(path)
    prev.write(RUN/'evidence/views.json',records)


if __name__=='__main__':
    main()
