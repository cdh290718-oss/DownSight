"""Compare the same fitted meshes in native palm views and calibrated stereo images."""
import hashlib
import importlib.util
import zipfile
import cv2
import numpy as np
import open3d as o3d
import torch

from transfer import RUN, OUT, BASE_OUT, ref, read, write, sha, SIDES, TIPS, DIGITS, JOINT_NAMES, palm_basis, mesh
from gallery import rot, free_render
from common_run import groups

spec = importlib.util.spec_from_file_location('stereo_render', RUN.parent/'2026-11-26_realvideo4_wilor_outline/render.py')
stereo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stereo)
stereo.OUT, stereo.RUN = OUT, RUN
MODES = ('B','cam3','cam4','cam3_wrist','cam4_wrist','cam3_B_wrist','cam4_B_wrist')
NATIVE = RUN.parent/'2026-11-27_wilor_native_hands'

def title(im,label):
    pic=cv2.copyMakeBorder(im,26,0,0,0,cv2.BORDER_CONSTANT,value=(33,33,33))
    cv2.putText(pic,label,(7,18),cv2.FONT_HERSHEY_SIMPLEX,.49,(250,250,250),1,cv2.LINE_AA)
    return pic

def comparison_views(key,models,faces,labels,images):
    reference=NATIVE/'assets'/(key+'.jpg')
    assert sha(reference)==read(NATIVE/'evidence/results.json')['images'][key]['sha256']
    native=cv2.imread(str(reference))
    rows=[]
    for i,side in enumerate(SIDES):
        name=side[0].upper()
        part=faces[labels==name+'_hand']
        indices=np.unique(part)
        ids=[JOINT_NAMES.index(side+'_'+s) for s in ['wrist']+[d+k for d in DIGITS for k in ['1','2','3','']]]
        for n,mode in enumerate(('cam3','cam4')):
            v,j=models[mode]
            across,up,normal=palm_basis(j[ids]).T
            tiles=[]
            for k,a in enumerate(np.deg2rad([12,78])):
                r=2*i+n
                original=native[r*285+29:(r+1)*285,256*(2+k):256*(3+k)]
                assert original.shape==(256,256,3)
                basis=np.column_stack([np.cos(a)*across+np.sin(a)*normal,up,-np.cos(a)*normal+np.sin(a)*across])
                xyz=v[indices]@basis
                center=(xyz.min(0)+xyz.max(0))/2
                scale=226/np.ptp(xyz[:,:2],axis=0).max()
                winding=part[:,::-1] if np.linalg.det(basis)<0 else part
                mapped=free_render(v,winding,basis,center,scale)[112:368,47:303].copy()
                background=mapped[0,0].copy()
                mapped[np.all(mapped==background,axis=2)]=(36,36,33)
                tiles.extend([title(original,f'{side} {mode} | MANO'),title(mapped,f'{side} {mode} | SMPL-X')])
            rows.append(np.hstack(tiles))
    stereo.save(images,key+'_native_transfer',np.vstack(rows))

def mark(im,hand,eye,ids):
    for idx in ids:
        if hand['inside'][eye][idx]:
            a=tuple(np.rint(hand['observation'][eye][idx]).astype(int))
            b=tuple(np.rint(hand['uv'][eye][idx]).astype(int))
            cv2.drawMarker(im,a,(255,225,15),cv2.MARKER_TILTED_CROSS,8,1,cv2.LINE_AA)
            cv2.circle(im,b,3,(30,130,250),1,cv2.LINE_AA)

def main():
    torch.set_num_threads(1)
    model,Camera=ref.prev.ref.common.setup_model()
    layer=model.layer['neutral'].eval()
    faces=np.asarray(model.face,dtype=np.int32)
    labels=groups(layer,faces)
    cams=list(map(Camera,ref.cameras()))
    y,x=np.indices((720,1280))
    px=np.stack((x,y),axis=-1).reshape(-1,2)
    rays=[]
    for cam in cams:
        r=np.zeros((720,1280,6),np.float32)
        r[:,:,3:]=cam.camera2world_ray(px).reshape(720,1280,3)
        rays.append(o3d.core.Tensor(r))
    images,hashes={},{}
    (OUT/'export').mkdir(exist_ok=True)
    for row in read(ref.OLD/'fit/0/prepared.json')['items'][2:4]:
        key=row['pair']
        rec=read(RUN/'evidence'/(key+'.json'))['results']
        models={}
        for mode in MODES:
            path=(BASE_OUT if mode=='B' else OUT)/'poses'/(key+'_'+mode+'.npz')
            assert sha(path)==rec[mode]['pose_sha256']
            p=np.load(path)
            q=p['locked']
            v,j=mesh(layer,q,p)
            v,obj=stereo.export(key,mode,v,faces)
            if mode!='B':
                hashes[obj.name]=sha(obj)
            models[mode]=v,j
        comparison_views(key,models,faces,labels,images)
        root=rot(q[21])@rot(p['global0'])
        bodies=[]
        for basis,label in [(root,'Front'),(root@rot([0,np.pi/2,0]),'Side')]:
            pts=np.vstack([v@basis for v,j in models.values()])
            center=(pts.max(0)+pts.min(0))*.5
            scale=min(315/np.ptp(pts[:,0]),440/np.ptp(pts[:,1]))
            bodies.append(np.hstack([title(free_render(models[m][0],faces,basis,center,scale),m+' '+label) for m in MODES[:5]]))
        stereo.save(images,key+'_body',np.vstack(bodies))
        zoom={'pure':[], 'wrist':[], 'B_wrist':[]}
        for n,(eye,ray) in enumerate(zip(SIDES,rays)):
            path=ref.OLD/'images'/(key+'_'+eye+'.png')
            assert sha(path)==read(ref.PRIOR/'evidence/frames.json')[int(key[-1])-1]['views'][eye]['frame']
            photo=cv2.imread(str(path))
            t=np.array(row['inputs']['views'][eye]['camera']['left_to_view'])
            masks={}
            for mode,(v,j) in models.items():
                part,valid,shade=stereo.raycast(v@t[:3,:3].T+t[:3,3],faces,labels,ray)
                masks[mode]=part
                if mode in ('cam3','cam4'):
                    im=photo.copy()
                    im[valid]=(.5*im[valid]+.5*shade[valid,None]).astype('uint8')
                    for side in SIDES:
                        mark(im,rec[mode]['hands'][side],n,range(21))
                    stereo.save(images,key+'_source_'+mode+'_view_cam'+str(3+n),im)
            for i,side in enumerate(SIDES):
                h=rec['B']['hands'][side]
                obs=np.array(h['observation'][n])
                ok=np.array(h['inside'][n])
                yy,xx=np.where(np.logical_or.reduce([m==i+1 for m in masks.values()]))
                pts=np.vstack((obs[np.array(TIPS)[ok[TIPS]]],[xx.min(),yy.min()],[xx.max(),yy.max()]))
                span=int(min(720,max(150,np.ptp(pts,axis=0).max()+32)))
                l,t0=np.clip(((pts.max(0)+pts.min(0)-span)*.5).astype(int),0,[1280-span,720-span])
                for group,extra in [('pure',''),('wrist','_wrist'),('B_wrist','_B_wrist')]:
                    tiles=[]
                    for mode,color in [('Photo',None),('B',(30,195,250)),('cam3'+extra,(60,220,95)),('cam4'+extra,(210,80,200))]:
                        im=photo.copy()
                        if color:
                            stereo.edge(im,masks[mode]==i+1,color)
                            mark(im,rec[mode]['hands'][side],n,[0]+list(TIPS))
                        tiles.append(title(cv2.resize(im[t0:t0+span,l:l+span],(285,285)),f'view {3+n} | {side} | {mode}'))
                    zoom[group].append(np.hstack(tiles))
        for group,tiles in zoom.items():
            stereo.save(images,key+'_stereo_'+group,np.vstack(tiles))
    archive=OUT/'export'/'hand_pose_transfer_OBJ.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name in hashes:
            z.write(OUT/'export'/name,name)
    with zipfile.ZipFile(archive) as z:
        assert set(z.namelist())==set(hashes) and z.testzip() is None
        assert all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in hashes.items())
    write(RUN/'evidence/attachment_audit.json',dict(files_sha256={archive.name:sha(archive)},members=hashes,exit_status=0))
    write(RUN/'evidence/results.json',dict(images=images,exit_status=0))

if __name__=='__main__':
    main()
