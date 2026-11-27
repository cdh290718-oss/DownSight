"""Render the paired full-body meshes, including their wrist-local geometry and held-out landmarks."""
import importlib.util
import zipfile
import hashlib
import cv2
import numpy as np
import open3d as o3d
import torch
from compare_fit import RUN, ROOT, OUT, ref, read, write, sha, SIDES, TIPS, JOINT_NAMES
from gallery import geometry,rot,free_render,load_obj
from common_run import groups

spec=importlib.util.spec_from_file_location('stereo_render',RUN.parent/'2026-11-26_realvideo4_wilor_outline/render.py')
stereo=importlib.util.module_from_spec(spec)
spec.loader.exec_module(stereo)
stereo.OUT=OUT
stereo.RUN=RUN

def unit(x):
    return x/np.linalg.norm(x)

def title(im,label):
    pic=cv2.copyMakeBorder(im,27,0,0,0,cv2.BORDER_CONSTANT,value=(30,30,30))
    cv2.putText(pic,label,(6,19),0,.5,(248,248,248),1,cv2.LINE_AA)
    return pic

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
    for row in read(ref.OLD/'fit/0/prepared.json')['items'][2:4]:
        key=row['pair']
        rec=read(RUN/'evidence'/(key+'.json'))['results']
        models={}
        for mode in ('A','B'):
            path=OUT/'poses'/(key+'_'+mode+'.npz')
            assert sha(path)==rec[mode]['pose_sha256']
            p=np.load(path)
            q=p['locked']
            params=dict(global_orient=p['global0'],body_pose=q[:21],transl=q[22],betas=p['betas'],left_hand_pose=q[23:38],right_hand_pose=q[38:])
            v,j,h,c=geometry(layer,params,q[21],False,None)
            assert np.allclose(h,c)
            v,obj=stereo.export(key,mode,v,faces)
            hashes[obj.name]=sha(obj)
            models[mode]=(v,j)
        root=rot(q[21])@rot(p['global0'])
        bodies=[]
        for basis,label in [(root,'Front'),(root@rot([0,np.pi/2,0]),'Side')]:
            pts=np.vstack([v@basis for v,j in models.values()])
            center=(pts.max(0)+pts.min(0))*.5
            scale=min(315/np.ptp(pts[:,0]),440/np.ptp(pts[:,1]))
            bodies.append(np.hstack([title(free_render(v,faces,basis,center,scale),mode+' '+label) for mode,(v,j) in models.items()]))
        stereo.save(images,key+'_body',np.vstack(bodies))
        mesh_rows=[]
        for side in SIDES:
            hand=faces[labels==side[0].upper()+'_hand']
            j=models['A'][1]
            def joint(name):
                return j[JOINT_NAMES.index(side+'_'+name)]
            up=unit(joint('middle1')-joint('wrist'))
            cross=joint('index1')-joint('pinky1')
            across=unit(cross-up*(cross@up))
            normal=unit(np.cross(across,up))
            cols=[]
            for angle in [0,75,180]:
                a=np.deg2rad(angle)
                basis=np.column_stack([across*np.cos(a)-normal*np.sin(a),up,across*np.sin(a)+normal*np.cos(a)])
                uv=np.vstack([v[np.unique(hand)]@basis for v,j in models.values()])
                center=(uv.min(0)+uv.max(0))*.5
                scale=min(325/np.ptp(uv[:,0]),335/np.ptp(uv[:,1]))
                for mode,(v,j) in models.items():
                    im=free_render(v,hand,basis,center,scale)[60:420]
                    cols.append(title(cv2.resize(im,(210,216)),f'{side} {mode} {angle} deg'))
            mesh_rows.append(np.hstack(cols))
        stereo.save(images,key+'_mesh_hands',np.vstack(mesh_rows))
        zoom=[]
        for n,(eye,cam,ray) in enumerate(zip(SIDES,cams,rays)):
            file=ref.OLD/'images'/(key+'_'+eye+'.png')
            assert sha(file)==read(ref.PRIOR/'evidence/frames.json')[int(key[-1])-1]['views'][eye]['frame']
            photo=cv2.imread(str(file))
            T=np.array(row['inputs']['views'][eye]['camera']['left_to_view'])
            masks={}
            for mode,(v,j) in models.items():
                part,valid,shade=stereo.raycast(v@T[:3,:3].T+T[:3,3],faces,labels,ray)
                masks[mode]=part
                if mode=='B':
                    im=photo.copy()
                    im[valid]=(.5*im[valid]+.5*shade[valid,None]).astype('uint8')
                    for side in SIDES:
                        h=rec[mode]['hands'][side]
                        for a,b,ok in zip(h['observation'][n],h['uv'][n],h['inside'][n]):
                            if ok:
                                cv2.drawMarker(im,tuple(np.rint(a).astype(int)),(255,225,15),cv2.MARKER_TILTED_CROSS,7,1,cv2.LINE_AA)
                                cv2.circle(im,tuple(np.rint(b).astype(int)),3,(30,130,250),1,cv2.LINE_AA)
                    stereo.save(images,key+'_cam'+str(n+3),im)
            for k,side in enumerate(SIDES):
                h=rec['B']['hands'][side]
                obs=np.array(h['observation'][n])
                ok=np.array(h['inside'][n])
                py,px=np.where((masks['A']==k+1)|(masks['B']==k+1))
                pts=np.vstack((obs[np.array(TIPS)[ok[TIPS]]], [px.min(),py.min()],[px.max(),py.max()]))
                span=int(min(720,max(150,np.ptp(pts,axis=0).max()+32)))
                l,t=np.clip(((pts.max(0)+pts.min(0)-span)*.5).astype(int),0,[1280-span,720-span])
                tiles=[]
                for mode,color in [('Photo',None),('A',(35,195,250)),('B',(60,220,95))]:
                    im=photo.copy()
                    if color:
                        stereo.edge(im,masks[mode]==k+1,color)
                        for idx in np.array(TIPS)[ok[TIPS]]:
                            obs_pt=tuple(np.rint(obs[idx]).astype(int))
                            uv=tuple(np.rint(rec[mode]['hands'][side]['uv'][n][idx]).astype(int))
                            cv2.line(im,obs_pt,uv,(140,140,140),1,cv2.LINE_AA)
                            cv2.drawMarker(im,obs_pt,(255,225,15),cv2.MARKER_TILTED_CROSS,7,1,cv2.LINE_AA)
                            cv2.circle(im,uv,3,(30,130,250),1,cv2.LINE_AA)
                    tiles.append(title(cv2.resize(im[t:t+span,l:l+span],(295,295)),f'{key} cam{n+3} {side} {mode}'))
                zoom.append(np.hstack(tiles))
        stereo.save(images,key+'_stereo_hands',np.vstack(zoom))
    zfile=OUT/'export'/'stereo_fingers_AB_OBJ.zip'
    with zipfile.ZipFile(zfile,'w',zipfile.ZIP_DEFLATED) as z:
        for name in hashes:
            z.write(OUT/'export'/name,name)
    with zipfile.ZipFile(zfile) as z:
        assert set(z.namelist())==set(hashes) and z.testzip() is None
        assert all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in hashes.items())
    write(RUN/'evidence/attachment_audit.json',dict(files_sha256={zfile.name:sha(zfile)},members=hashes,exit_status=0))
    write(RUN/'evidence/results.json',dict(images=images,exit_status=0))
if __name__=='__main__':
    main()
