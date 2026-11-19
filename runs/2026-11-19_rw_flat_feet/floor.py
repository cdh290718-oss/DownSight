"""Estimate ground normals from stereo image features on selected bare-carpet regions."""
import sys
from pathlib import Path
import cv2
import numpy as np
RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[1]
sys.path.insert(0,str(ROOT/'runs/2026-11-19_rw_position_only'))
import trial
from trial import common,read,write,sha
OUT=ROOT/'outputs/rw_flat_feet_20261119'

# Polygons inspected on 370 x 370 central left-eye thumbnails (original black bars omitted).
PATCHES=[
 [[(100,160),(151,154),(191,178),(203,212),(180,238),(156,233),(153,214),(131,203)],
  [(272,100),(288,98),(318,143),(319,191),(292,214),(277,202),(259,200),(241,212),(225,207),(249,168)]],
 [[(100,193),(145,192),(173,209),(217,193),(244,193),(219,244),(198,264),(131,281),(105,254),(113,226)],
  [(231,163),(248,142),(266,138),(269,163),(243,207),(227,212)]],
 [[(63,162),(117,157),(154,174),(214,158),(250,139),(266,149),(254,175),(230,207),(214,240),(148,249),(101,214)],
  [(216,213),(247,195),(287,192),(287,207),(251,240),(214,253)]],
 [[(58,181),(110,165),(139,194),(199,196),(221,188),(223,210),(201,236),(183,265),(144,269),(112,252),(84,236)],
  [(225,177),(245,172),(263,185),(233,220),(214,227)]]]
def estimate(pair,patches,cams,view):
    T=np.array(view['right']['camera']['left_to_view'])
    R,t=T[:3,:3],T[:3,3]
    o=-R.T@t
    imgs=[cv2.imread(str(common.ARCH/'body_input'/(pair+'_'+s+'.png')),0) for s in ['left','right']]
    mask=np.zeros_like(imgs[0])
    for poly in patches:
        points=np.rint(np.array(poly)*720/370+[280,0]).astype('int32')
        cv2.fillPoly(mask,[points],255)
    sift=cv2.SIFT_create(contrastThreshold=.003)
    kl,dl=sift.detectAndCompute(imgs[0],mask)
    kr,dr=sift.detectAndCompute(imgs[1],None)
    mat=cv2.BFMatcher()
    ab=mat.knnMatch(dl,dr,k=2)
    ba=mat.match(dr,dl)
    matches=[a for a,b in ab if a.distance<.86*b.distance and ba[a.trainIdx].trainIdx==a.queryIdx]
    # Exclude multiple SIFT orientations at the same location.
    used=set();xyz=[];uv=[]
    for m in matches:
        l,r=kl[m.queryIdx].pt,kr[m.trainIdx].pt
        cell=tuple(np.rint(l).astype(int))
        if cell in used:continue
        u,v=[cam.camera2world_ray(np.array([xy]))[0] for cam,xy in zip(cams,[l,r])]
        v=R.T@v
        ab=np.column_stack([u,-v])
        d=np.linalg.lstsq(ab,o,rcond=None)[0]
        x=(u*d[0]+o+v*d[1])/2
        if min(d)<.50 or max(d)>4 or np.linalg.norm(ab@d-o)>.045:continue
        used.add(cell);xyz.append(x);uv.append(l)
    print(pair,'SIFT',len(kl),len(matches),len(xyz),flush=True)
    return np.array(xyz),np.array(uv),mask

if __name__=='__main__':
    _,cam=common.setup_model()
    cams=[cam(p) for p in common.camera_files()]
    items=read(trial.OUT/'prepared.json')['items']
    for i,patch in enumerate(PATCHES):
        pair='body%02d'%(i+1)
        x,u,mask=estimate(pair,patch,cams,next(r['inputs']['views'] for r in items if r['pair']==pair))
        np.savez(OUT/(pair+'_floor.npz'),x=x,u=u,mask=mask)
