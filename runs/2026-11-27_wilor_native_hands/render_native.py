"""Validate native MANO reconstruction and render its original placement and wrist-relative views."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from egosmplx_pipeline import wilor_predict as wmodel
from wilor.models import load_wilor

RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[1]
CACHE=ROOT/'outputs/wilor_stereo_fingers_20261126'
OLD=ROOT/'outputs/realvideo4_dv3_20261125'
read=lambda p:json.loads(p.read_text())

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):
            h.update(block)
    return h.hexdigest()

def unit(v):
    return v/np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),1e-12)

def raster(uv,xyz,faces,shape,color):
    # Nearest depth per subpixel sample; interpolated vertex normals soften facet boundaries.
    tri=xyz[faces]
    face_normals=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    normals=np.zeros_like(xyz)
    for k in range(3):
        np.add.at(normals,faces[:,k],face_normals)
    shade=.46+.53*np.abs(unit(normals)@unit(np.array([-.35,-.52,-.83])))
    depth=np.full(shape,np.inf)
    rgb=np.zeros((*shape,3),np.uint8)
    for face in faces:
        coords=uv[face]
        x0,y0=np.maximum(np.floor(coords.min(0)).astype(int),0)
        x1,y1=np.minimum(np.ceil(coords.max(0)).astype(int),[shape[1]-1,shape[0]-1])
        if x0>x1 or y0>y1: continue
        a,b,c=coords
        ab,ac=b-a,c-a
        den=ab[0]*ac[1]-ac[0]*ab[1]
        if abs(den)<1e-8: continue
        yy,xx=np.mgrid[y0:y1+1,x0:x1+1]
        dx,dy=xx-a[0],yy-a[1]
        wb=(dx*ac[1]-dy*ac[0])/den
        wc=(ab[0]*dy-ab[1]*dx)/den
        wa=1-wb-wc
        z=xyz[face,2]
        inv=wa/z[0]+wb/z[1]+wc/z[2]
        zz=1/np.maximum(inv,1e-10)
        sub=depth[y0:y1+1,x0:x1+1]
        visible=(wa>=0)&(wb>=0)&(wc>=0)&(zz<sub)
        sub[visible]=zz[visible]
        light=(wa*shade[face[0]]/z[0]+wb*shade[face[1]]/z[1]+wc*shade[face[2]]/z[2])*zz
        rgb[y0:y1+1,x0:x1+1][visible]=np.clip(light[visible,None]*color,0,255)
    return rgb,np.isfinite(depth)

def mesh_views(v,j,faces):
    up=unit(j[9]-j[0])
    across=j[5]-j[17]
    across=unit(across-up*(across@up))
    normal=unit(np.cross(across,up))
    panels=[]
    for a in np.deg2rad([12,78]):
        mat=np.column_stack([np.cos(a)*across+np.sin(a)*normal,-up,np.cos(a)*normal-np.sin(a)*across])
        xyz=v@mat
        xyz=xyz-(xyz.min(0)+xyz.max(0))/2
        uv=xyz[:,:2]*((256-30)/np.ptp(xyz[:,:2],axis=0).max())+128
        xyz[:,2]+=10
        rgb,mask=raster(uv,xyz,faces,(256,256),np.array([215,223,230]))
        canvas=np.full(rgb.shape,(36,36,33),np.uint8)
        canvas[mask]=rgb[mask]
        panels.append(canvas)
    return panels

def label(img,s):
    panel=cv2.copyMakeBorder(img,29,0,0,0,cv2.BORDER_CONSTANT,value=(33,32,31))
    cv2.putText(panel,s,(7,20),cv2.FONT_HERSHEY_SIMPLEX,.48,(248,248,248),1,cv2.LINE_AA)
    return panel

def main():
    config=read(ROOT/'configs/pipeline.head01.developer.json')
    paths=config['paths']
    assert digest(Path(paths['wilor_checkpoint']))==config['checkpoint_sha256']['wilor']
    model,cfg=load_wilor(paths['wilor_checkpoint'],paths['wilor_config'])
    model.cuda().eval()
    mano=model.mano
    reference=read(RUN.parent/'2026-11-26_wilor_stereo_fingers/evidence/replay.json')
    guide=read(ROOT/'outputs/realvideo4_palm_20261126/guided.json')['selected']
    cases={}
    images={}
    for key in ('clip003','clip004'):
        rows=[]
        batches={}
        for eye in ('left','right'):
            file=OLD/'images'/(key+'_'+eye+'.png')
            assert digest(file)==reference[key+'_'+eye+'_L']['image_sha256']
            photo=cv2.imread(str(file))
            centers=np.array([guide[key+'_'+eye][s]['crop_center'] for s in ('L','R')])
            boxes=np.column_stack([centers-photo.shape[1]/16,centers+photo.shape[1]/16])
            ds=wmodel.ViTDetDataset(cfg,photo,boxes,np.array([0,1]),rescale_factor=2.)
            batch=wmodel.recursive_to(next(iter(torch.utils.data.DataLoader(ds,batch_size=2))),'cuda')
            with torch.inference_mode():
                out=model(batch)
                cam=out['pred_cam'].clone()
                sign=2*batch['right']-1
                cam[:,1]*=sign
                sizes=batch['img_size'].float()
                focal=cfg.EXTRA.FOCAL_LENGTH/cfg.MODEL.IMAGE_SIZE*sizes.max()
                trans=wmodel.cam_crop_to_full(cam,batch['box_center'].float(),batch['box_size'].float(),sizes,focal)
                joints=out['pred_keypoints_3d'].clone()
                verts=out['pred_vertices'].clone()
                joints[:,:,0]*=sign[:,None]
                verts[:,:,0]*=sign[:,None]
            batches[eye]=(verts.cpu().numpy(),joints.cpu().numpy(),trans.cpu().numpy(),float(focal),sizes.cpu().numpy())
        for side in ('L','R'):
            for n,eye in enumerate(('left','right')):
                tag=key+'_'+eye+'_'+side
                path=CACHE/(tag+'.npz')
                data=np.load(path)
                photo=OLD/'images'/(key+'_'+eye+'.png')
                assert digest(photo)==reference[tag]['image_sha256']
                im=cv2.imread(str(photo))
                h,w=im.shape[:2]
                index=('L','R').index(side)
                vs,js,ts,focal,sizes=batches[eye]
                v,j,t=vs[index],js[index],ts[index]
                faces=np.asarray(data['faces'])
                assert np.array_equal(faces,mano.faces)
                if side=='L':
                    faces=faces[:,::-1]
                uv=wmodel.project(v,t,focal,sizes[index])
                joint_uv=wmodel.project(j,t,focal,sizes[index])
                vgap=float(np.linalg.norm(uv-data['verts_uv'],axis=1).max())
                jgap=float(np.linalg.norm(joint_uv-data['uv'],axis=1).max())
                assert min((v+t)[:,2])>0 and max(vgap,jgap)<.05,(tag,vgap,jgap)
                assert np.max(abs(joint_uv-guide[key+'_'+eye][side]['uv']))<.1
                # One crop includes the wrist target, the native prediction and in-frame image hand landmarks.
                source=read(OLD/'predictions.json')[key+'_'+eye+'.png']['xy']
                pts=np.asarray(source)[21+(21 if side=='L' else 0):42+(21 if side=='L' else 0)]
                guide_wrist=pts[-1]
                palm_width=np.median(np.linalg.norm(pts[[7,11,15,19]]-guide_wrist,axis=1))
                pts=pts[np.linalg.norm(pts-guide_wrist,axis=1)<3.2*palm_width]
                pts=pts[((pts>=0)&(pts<[w,h])).all(1)]
                combo=np.clip(np.vstack([uv,pts]),[0,0],[w-1,h-1])
                low,high=combo.min(0),combo.max(0)
                span=int(min(h,max(165,np.ptp(combo,axis=0).max()+44)))
                x,y=np.clip(np.round((low+high-span)/2).astype(int),[0,0],[w-span,h-span])
                crop=cv2.resize(im[y:y+span,x:x+span],(256,256))
                crop_uv=(uv-[x,y]+.5)*256/span-.5
                rgb,mask=raster(crop_uv,v+t,faces,crop.shape[:2],np.array([118,205,255]))
                overlay=crop.copy()
                overlay[mask]=np.rint(.28*crop[mask]+.72*rgb[mask]).astype('uint8')
                assert mask.any()
                panels=[crop,overlay]+mesh_views(v,j,faces)
                cols=['photo','native overlay','MANO / view 1','MANO / view 2']
                rows.append(np.hstack([label(p,f'{side}  cam{3+n}  |  '+s) for p,s in zip(panels,cols)]))
                cases[tag]=dict(input_pack_sha256=digest(path),photo_sha256=digest(photo),vertex_replay_max_px=vgap,joint_replay_max_px=jgap,
                                crop_xywh=[int(x),int(y),span,span],faces=len(faces),vertices=len(v))
        result=np.vstack(rows)
        dest=RUN/'assets'/(key+'.jpg')
        assert cv2.imwrite(str(dest),result,[cv2.IMWRITE_JPEG_QUALITY,94])
        images[key]=dict(sha256=digest(dest),size=list(result.shape[1::-1]))
    (RUN/'evidence/results.json').write_text(json.dumps(dict(cases=cases,images=images,exit_status=0),indent=2)+'\n')

if __name__=='__main__': main()
