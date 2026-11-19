"""Orthographic OBJ side views with the estimated studio floor drawn at its fitted height."""
import cv2
import numpy as np
from experiment import RUN,OUT,common,read,write,sha
from plane import unit
import render
from experiment import KEYS

def main():
    results=read(RUN/'evidence/results_auto.json')['rows']
    floors=read(RUN/'evidence/floor.json')['rows']
    inputs=read(OUT/'prepared.json')['items']
    assets=RUN/'assets'
    images={}
    for fl in floors:
        pair=fl['pair'];up=np.array(fl['up'])
        pose=next(i for i in inputs if i['pair']==pair)
        forward=np.array(pose['initial_body_rotation'])[:,2]
        x=unit(forward-up*(forward@up))
        basis=np.stack([x,up,np.cross(x,up)],axis=1)
        data=[]
        for mode in KEYS.values():
            r=next(r for r in results if r['pair']==pair and r['condition']==mode)
            obj=OUT/'objects'/pair/r['obj']
            assert sha(obj)==r['obj_sha256']
            data.append(render.load_obj(obj))
        union=np.vstack([v@basis for v,_ in data])
        lo,hi=union.min(0),union.max(0)
        lo[1]=min(lo[1],-fl['plane_distance_m'])
        center=(lo+hi)/2
        scale=min(300/(hi[0]-lo[0]),415/(hi[1]-lo[1]))
        ground=round(240+(fl['plane_distance_m']+center[1])*scale)
        panels=[]
        for mode,(v,f) in zip(KEYS.values(),data):
            im=render.free_render(v,f,basis,center,scale)
            cv2.line(im,(5,ground),(345,ground),(70,145,85),1,cv2.LINE_AA)
            label=np.full((35,350,3),245,np.uint8)
            cv2.putText(label,mode,(12,24),cv2.FONT_HERSHEY_SIMPLEX,.65,(30,30,30),1,cv2.LINE_AA)
            panels.append(np.vstack([label,im]))
        out=assets/(pair+'__floor.jpg')
        out.parent.mkdir(parents=True,exist_ok=True)
        render.save(out,np.vstack([np.hstack(panels[:2]),np.hstack(panels[2:])]))
        images[out.name]=sha(out)
    write(RUN/'evidence/floor_views.json',dict(images=images,description='Same side view and scale across four modes within a pose. Green = height of estimated floor.'))
if __name__=='__main__':main()
