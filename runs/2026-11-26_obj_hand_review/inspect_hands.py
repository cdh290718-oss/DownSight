"""Inspect hashed final meshes in a wrist-local coordinate frame."""
from pathlib import Path
import sys, json
import numpy as np
import cv2

RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[1]
sys.path.insert(0,str(RUN.parent/'2026-11-26_realvideo34_purple_hand_a'))
from gallery_a import paired, geometry, read, sha, load_obj, JOINT_NAMES
PREV=RUN.parent/'2026-11-26_purple_arm_hand_a'
OUT=ROOT/'outputs/purple_arm_hand_a_20261126'
DIGITS=['thumb','index','middle','ring','pinky']

def unit(v):
    return v/np.linalg.norm(v)

def angle(a,b):
    return float(np.rad2deg(np.arccos(np.clip(unit(a)@unit(b),-1,1))))

def tile(img,label):
    bar=np.full((27,img.shape[1],3),36,np.uint8)
    cv2.putText(bar,label,(5,19),0,.48,(255,255,255),1,cv2.LINE_AA)
    return np.vstack([bar,img])

def render(v,faces,basis):
    p=v@basis
    xy=p[:,:2]*[1,-1]
    active=xy[np.unique(faces)]
    scale=215/max(np.ptp(active,axis=0))
    px=np.rint((xy-(active.min(0)+active.max(0))/2)*scale+120).astype(np.int32)
    tri=p[faces]
    normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    normal/=np.linalg.norm(normal,axis=1,keepdims=True).clip(1e-10)
    light=unit(np.array([-.35,.55,1]))
    shade=np.clip(142+85*(normal@light),54,235).astype(int)
    img=np.full((240,240,3),245,np.uint8)
    for i in np.argsort(tri[:,:,2].mean(1)):
        cv2.fillConvexPoly(img,px[faces[i]],(int(shade[i]),)*3,cv2.LINE_AA)
    return img

model,_=paired.ref.prev.ref.common.setup_model()
layer=model.layer['neutral'].eval()
labels=layer.lbs_weights.argmax(1).numpy()
archive=read(PREV/'evidence/gallery.json')['rows']
result={}
for key,row in archive.items():
    obj=OUT/row['obj']; pose=OUT/'poses'/(key+'_control.npz')
    assert sha(obj)==row['sha256'] and sha(pose)==row['pose_sha256']
    v,faces=load_obj(obj)
    p=np.load(pose); x=p['locked']
    vertices,j,_,_=geometry(layer,dict(global_orient=p['global0'],body_pose=x[:21],transl=x[22],betas=p['betas'],left_hand_pose=x[23:38],right_hand_pose=x[38:]),x[21],False,None)
    assert np.max(abs(vertices-v))<1e-8
    rows=[]
    result[key]={'obj_sha256':sha(obj),'pose_sha256':sha(pose),'hands':{}}
    for side,w,f in [('left',20,25),('right',21,40)]:
        joint=lambda name:j[JOINT_NAMES.index(side+'_'+name)]
        up=unit(joint('middle1')-joint('wrist'))
        across=joint('index1')-joint('pinky1')
        across=unit(across-up*(up@across))
        front=unit(np.cross(across,up))
        hand=faces[np.isin(labels,[w]+list(range(f,f+15)))[faces].all(1)]
        panels=[]
        for degrees,label in [(0,'Face A'),(180,'Face B'),(75,'Oblique')]:
            rad=np.deg2rad(degrees)
            axes=np.column_stack([across*np.cos(rad)-front*np.sin(rad),up,across*np.sin(rad)+front*np.cos(rad)])
            panels.append(tile(render(v,hand,axes),side+' | '+label))
        errors=[]
        for eye,cam in [('left','cam3'),('right','cam4')]:
            pts=row['views'][cam]['hands'][side]
            errors.append([p['gap_px'] for p in pts[5:]])
            xy=np.array([p['det'] for p in pts])
            span=max(145,int(np.ptp(xy,axis=0).max()+55))
            start=np.clip(np.rint((xy.max(0)+xy.min(0)-span)/2).astype(int),[0,0],[1280-span,720-span])
            a,b=start
            image=paired.ref.OLD/'images'/(key+'_'+eye+'.png')
            assert sha(image)==row['views'][cam]['source_image_sha256']
            crop=cv2.imread(str(image))[b:b+span,a:a+span]
            panels.append(tile(cv2.resize(crop,(240,240)),side+' | '+cam))
        rows.append(np.hstack(panels))
        entry={}
        for n,d in enumerate(DIGITS):
            q=np.array([joint(d+str(i)) for i in [1,2,3]]+[joint(d)])
            dirs=np.diff(q,axis=0)
            entry[d]=dict(turn_deg=[angle(dirs[0],dirs[1]),angle(dirs[1],dirs[2])],tip_error_px=[e[n] for e in errors])
        result[key]['hands'][side]=entry
    assert cv2.imwrite(str(RUN/'assets'/(key+'.jpg')),np.vstack(rows))
(RUN/'evidence/geometry.json').write_text(json.dumps(result,indent=2))
