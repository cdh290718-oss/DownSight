"""Side-by-side raw photos, native WiLoR silhouettes and fitted full-body visibility."""
import json
import zipfile
from pathlib import Path
import sys
import cv2
import numpy as np
import open3d as o3d
import torch
from context_run import RUN,OUT,PREVIOUS,ROOT,ref,read,write,sha,SIDES,paired
MODES = ["control","wilor","anchored"]
import gallery as old
from common_run import groups
from render_result import geometry,rot,free_render,load_obj,text,DARK


def save(images,name,img):
    path=RUN/'assets'/(name+'.jpg')
    assert cv2.imwrite(str(path),img,[cv2.IMWRITE_JPEG_QUALITY,93])
    images[name]=dict(size=list(img.shape[1::-1]),sha256=sha(path))


def export(key,mode,v,faces):
    dest=OUT/'export'/(key+'_'+mode+'_cam3_m.obj')
    with dest.open('w') as out:
        out.write('# '+key+' '+mode+' cam3 meters\n')
        np.savetxt(out,v,fmt='v %.9f %.9f %.9f')
        np.savetxt(out,faces+1,fmt='f %d %d %d')
    loaded,topo=load_obj(dest)
    assert np.array_equal(topo,faces) and np.max(abs(v-loaded))<1e-8
    return loaded,dest


def raycast(vertices,faces,labels,rays):
    scene=o3d.t.geometry.RaycastingScene(nthreads=1)
    indices=np.flatnonzero((vertices[faces,2]>1e-5).all(1))
    scene.add_triangles(o3d.core.Tensor(np.ascontiguousarray(vertices,dtype='float32')),o3d.core.Tensor(faces[indices].astype('uint32')))
    hit=scene.cast_rays(rays,nthreads=1)
    hit_ids=hit['primitive_ids'].numpy()
    valid=hit_ids<len(indices)
    part=np.zeros(valid.shape,np.uint8)
    cat=np.where(labels=='L_hand',1,np.where(labels=='R_hand',2,0))
    part[valid]=cat[indices[hit_ids[valid]]]
    normals=hit['primitive_normals'].numpy()
    shade=(75+155*np.abs(normals@np.array([.18,-.38,-.90]))).clip(0,255).astype('uint8')
    return part,valid,shade


def edge(img,mask,color):
    c,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img,c,-1,color,1,cv2.LINE_AA)


def main():
    torch.set_num_threads(1)
    model,Camera=ref.prev.ref.common.setup_model()
    layer=model.layer['neutral'].eval()
    faces=np.asarray(model.face,dtype=np.int32)
    labels=groups(layer,faces)
    cams=list(map(Camera,ref.cameras()))
    yy,xx=np.indices((720,1280))
    px=np.stack((xx,yy),axis=-1).reshape(-1,2)
    rays=[]
    for cam in cams:
        r=np.zeros((720,1280,6),np.float32)
        r[:,:,3:]=cam.camera2world_ray(px).reshape(720,1280,3)
        rays.append(o3d.core.Tensor(r))
    images,hashes={},{}
    for row in read(ref.OLD/'fit/0/prepared.json')['items']:
        key=row['pair']
        rec=dict(read(PREVIOUS/'evidence'/(key+'.json'))['results'])
        rec['anchored']=read(RUN/'evidence'/(key+'.json'))['results']['wilor']
        frame={}
        for mode in MODES:
            file=OUT/'poses'/(key+'_wilor.npz') if mode=='anchored' else paired.OUT/'poses'/(key+'_'+mode+'.npz')
            assert sha(file)==rec[mode]['pose_sha256']
            pack=np.load(file)
            x=pack['locked']
            params=dict(body_pose=x[:21],betas=pack['betas'],global_orient=pack['global0'],transl=x[22],left_hand_pose=x[23:38],right_hand_pose=x[38:])
            v,j,head,chest=geometry(layer,params,x[21],False,None)
            assert np.allclose(head,chest)
            if mode=='anchored':
                v,file=export(key,mode,v,faces)
                hashes[file.name]=sha(file)
            frame[mode]=(v,j)
        basis=rot(x[21])@rot(pack['global0'])
        obj_panels={m:[] for m in MODES}
        for view,mat in [('front',basis),('side',basis@rot([0.,np.pi/2,0.]))]:
            coords=np.vstack([v@mat for v,j in frame.values()])
            center=(coords.max(0)+coords.min(0))/2
            scale=min(315/np.ptp(coords[:,0]),440/np.ptp(coords[:,1]))
            for mode,(v,j) in frame.items():
                tile=free_render(v,faces,mat,center,scale)
                text(tile,mode+'  '+view,(8,23))
                obj_panels[mode].append(tile)
        save(images,key+'_obj',np.vstack([np.hstack(obj_panels[m]) for m in MODES]))
        zoom={s:[] for s in SIDES}
        for n,(eye,cam,ray) in enumerate(zip(SIDES,cams,rays)):
            im=cv2.imread(str(ref.OLD/'images'/(key+'_'+eye+'.png')))
            T=np.array(row['inputs']['views'][eye]['camera']['left_to_view'])
            masks={}
            for mode,(v,j) in frame.items():
                parts,valid,shade=raycast(v@T[:3,:3].T+T[:3,3],faces,labels,ray)
                masks[mode]=parts
                if mode=='anchored':
                    overlay=im.copy()
                    overlay[valid]=(.50*im[valid]+.50*shade[valid,None]).astype('uint8')
                    for side in SIDES:
                        h=rec[mode]['hands'][side]
                        for a,b in zip(h['target'][n],h['uv'][n]):
                            cv2.drawMarker(overlay,tuple(np.rint(a).astype(int)),(255,220,20),cv2.MARKER_TILTED_CROSS,7,1,cv2.LINE_AA)
                            cv2.circle(overlay,tuple(np.rint(b).astype(int)),3,(30,130,255),1,cv2.LINE_AA)
                    save(images,key+'_cam'+str(n+3),overlay)
            for k,side in enumerate(SIDES):
                name=key+'_'+eye+'_'+side[0].upper()+'.png'
                raw=cv2.imread(str(paired.OUT/'masks'/name),0)>0
                anchored=cv2.imread(str(OUT/'masks'/name),0)>0
                region=raw|anchored|(masks['control']==k+1)|(masks['wilor']==k+1)|(masks['anchored']==k+1)
                y,xpix=np.where(region)
                xy=np.asarray(rec['wilor']['hands'][side]['target'][n])
                points=np.vstack([xy,[xpix.min(),y.min()],[xpix.max(),y.max()]])
                span=int(min(720,max(145,np.ptp(points,axis=0).max()+28)))
                a,b=np.clip(((points.max(0)+points.min(0)-span)/2).astype(int),0,[1280-span,720-span])
                tiles=[]
                for name,mask,color in [('Photo',None,None),('Native target',raw,(197,70,255)),('Anchored target',anchored,(240,188,40)),('A: image outline',masks['control']==k+1,(40,210,255)),('B: native',masks['wilor']==k+1,(35,225,95)),('C: anchored',masks['anchored']==k+1,(45,105,255))]:
                    crop=im[b:b+span,a:a+span].copy()
                    if mask is not None:
                        edge(crop,mask[b:b+span,a:a+span],color)
                    tile=cv2.resize(crop,(250,250))
                    tile=cv2.copyMakeBorder(tile,30,0,0,0,cv2.BORDER_CONSTANT,value=DARK)
                    text(tile,f'{side[0].upper()} cam{n+3} | '+name,(4,20),.46)
                    tiles.append(tile)
                zoom[side].append(np.hstack(tiles))
        save(images,key+'_hands',np.vstack(zoom['left']+zoom['right']))
    archive=OUT/'export'/'realvideo4_wilor_wrist_C_OBJ.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name in hashes:
            z.write(OUT/'export'/name,name)
    with zipfile.ZipFile(archive) as z:
        import hashlib
        assert set(z.namelist())==set(hashes) and z.testzip() is None
        assert all(hashlib.sha256(z.read(p)).hexdigest()==h for p,h in hashes.items())
    write(RUN/'evidence/attachment_audit.json',dict(files_sha256={archive.name:sha(archive)},members=hashes,exit_status=0))
    write(RUN/'evidence/results.json',dict(images=images,exit_status=0))

if __name__=='__main__':
    main()
