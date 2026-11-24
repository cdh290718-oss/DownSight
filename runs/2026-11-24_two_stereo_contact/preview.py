"""Small derived diagrams: visible hand boundaries only, no original image pixels."""
import cv2
import numpy as np
import open3d as o3d
import torch
from common_run import RUN,OUT,PAIRS,inputs
from fit_probe import Frame
from outline import STRIDE,boundary

INK=(53,44,36)
MASK=(220,220,218)
SAPIENS=(200,110,0)
MODEL=(60,55,215)


def rendered(eye,v):
    vv=v@eye.transform[:3,:3].T+eye.transform[:3,3]
    keep=(vv[eye.faces,2]>1e-5).all(1)
    scene=o3d.t.geometry.RaycastingScene(nthreads=1)
    scene.add_triangles(o3d.core.Tensor(np.asarray(vv,np.float32)),o3d.core.Tensor(eye.faces[keep].astype(np.uint32)))
    ids=scene.cast_rays(eye.rays,nthreads=1)['primitive_ids'].numpy()
    valid=ids<int(keep.sum())
    label=np.full(len(ids),'background',dtype='<U16')
    label[valid]=eye.labels[np.flatnonzero(keep)[ids[valid]]]
    return label.reshape(eye.shape)


def main():
    torch.set_num_threads(1)
    model,cam,cfg,items=inputs()
    panels=[]
    for pair in PAIRS:
        item,orig=items[pair]
        frame=Frame(pair,item,orig,model,cam,cfg)
        states=dict(np.load(OUT/(pair+'_final.npz')))
        with torch.no_grad():
            vs={k:frame.state(torch.from_numpy(x))[0].numpy() for k,x in states.items()}
        for side,eye in zip(['left','right'],frame.eyes):
            image=np.load(OUT/'seg/frames'/(pair+'_'+side)/'labels.npy')[STRIDE//2::STRIDE,STRIDE//2::STRIDE]
            pred_masks=[image==ident for ident in (6,15)]
            renderings={name:rendered(eye,v) for name,v in vs.items()}
            crops=[]
            for mode in ['v3','hard','outline','joint']:
                cell=np.zeros((*image.shape,3),np.uint8)+249
                cell[image!=0]=MASK
                for mask in pred_masks:
                    cell[boundary(mask)]=SAPIENS
                for part in ['L_hand','R_hand']:
                    mask=renderings[mode]==part
                    cell[boundary(mask)]=MODEL
                panel=cv2.resize(cell,(218,218))
                cv2.rectangle(panel,(0,0),(217,23),(250,250,250),-1)
                cv2.putText(panel,mode,(8,17),cv2.FONT_HERSHEY_SIMPLEX,.45,INK,1,cv2.LINE_AA)
                crops.append(panel)
            row=np.hstack(crops)
            bar=np.full((24,row.shape[1],3),250,np.uint8)
            cv2.putText(bar,pair+'  |  '+side+' eye',(8,17),cv2.FONT_HERSHEY_SIMPLEX,.5,INK,1,cv2.LINE_AA)
            panels.extend([bar,row])
    legend=np.full((30,872,3),250,np.uint8)
    for x,name,color in [(10,'Sapiens2 hand boundary',SAPIENS),(325,'Visible SMPL-X hand boundary',MODEL)]:
        cv2.line(legend,(x,17),(x+22,17),color,2)
        cv2.putText(legend,name,(x+29,21),cv2.FONT_HERSHEY_SIMPLEX,.46,INK,1,cv2.LINE_AA)
    assert cv2.imwrite(str(RUN/'evidence/stereo_comparison.png'),np.vstack([legend]+panels))


if __name__=='__main__':main()
