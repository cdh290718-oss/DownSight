"""Refine stereo carpet planes by normalized photometric matching over distributed pixels."""
import cv2
import numpy as np
from scipy.optimize import least_squares
from floor import OUT,RUN,common,trial,read,write,sha

def unit(x):return x/np.linalg.norm(x)
def angle(a,b):return float(np.rad2deg(np.arccos(np.clip(unit(a)@unit(b),-1,1))))
def ransac(x):
    rng=np.random.default_rng(314)
    ids=np.array([rng.choice(len(x),3,replace=False) for _ in range(2000)])
    a,b,c=x[ids].transpose(1,0,2)
    n=np.cross(b-a,c-a)
    n/=np.maximum(1e-10,np.linalg.norm(n,axis=1,keepdims=True))
    e=np.abs(x@n.T-(a*n).sum(1))
    keep=e[:,(e<.04).sum(0).argmax()]<.04
    pt=x[keep].mean(0)
    v=np.linalg.svd(x[keep]-pt,full_matrices=False)[2][-1]
    return v/(v@pt),keep

def sample(im,uv):
    x,y=uv.T
    ix=np.clip(x.astype(int),0,im.shape[1]-2);iy=np.clip(y.astype(int),0,im.shape[0]-2)
    fx=x-ix;fy=y-iy
    return (1-fx)*((1-fy)*im[iy,ix]+fy*im[iy+1,ix])+fx*((1-fy)*im[iy,ix+1]+fy*im[iy+1,ix+1])

def norm_image(im,sigma):
    a=cv2.GaussianBlur(im,(0,0),sigma)
    lo=cv2.GaussianBlur(a,(0,0),8.)
    var=cv2.GaussianBlur((a-lo)**2,(0,0),8.)
    return (a-lo)/np.sqrt(2.+var)

def refine(start,uv,images,cams,R,t):
    q=cams[0].camera2world_ray(uv.astype(float))
    p=start
    for scale in [1.8,.7]:
        l,r=[norm_image(im,scale) for im in images]
        y0=sample(l,uv)
        def residual(p):
            z=np.maximum(.06,q@p)
            xy=cams[1].world2camera(q@R.T/z[:,None]+t)
            return sample(r,xy)-y0
        p=least_squares(residual,p,loss='soft_l1',f_scale=.35,max_nfev=70).x
    return p

if __name__=='__main__':
    _,cam=common.setup_model()
    cams=[cam(p) for p in common.camera_files()]
    rows=[]
    for i in range(4):
        pair='body%02d'%(i+1)
        z=np.load(OUT/(pair+'_floor.npz'))
        item=next(x for x in read(trial.OUT/'prepared.json')['items'] if x['pair']==pair)
        t=np.array(item['inputs']['views']['right']['camera']['left_to_view'])
        imgs=[cv2.imread(str(common.ARCH/'body_input'/(pair+'_'+s+'.png')),0).astype(float) for s in ['left','right']]
        s,keep=ransac(z['x']);yy,xx=np.nonzero(z['mask'])
        uv=np.column_stack([xx,yy])[::5]
        h=((uv[:,0]//26+uv[:,1]//26)%2).astype(bool)
        ps=[refine(s,u,imgs,cams,t[:3,:3],t[:3,3]) for u in [uv,uv[h],uv[~h]]]
        diff=angle(ps[1],ps[2]);normal=-unit(ps[0])
        assert normal[2]<0 and .5<1/np.linalg.norm(ps[0])<3 and diff<8,(pair,diff)
        rows.append(dict(pair=pair,up=normal.tolist(),plane_distance_m=1/np.linalg.norm(ps[0]),
            photometric_checkerboard_disagreement_deg=diff,initial_sparse_vs_dense_deg=angle(s,ps[0]),
            sparse_points=len(z['x']),sparse_inliers=int(keep.sum()),photometric_points=len(uv)))
        print(rows[-1],flush=True)
    write(RUN/'evidence/floor.json',dict(rows=rows,method='Per-frame planes on manually selected bare carpet in left image, calibrated stereo SIFT ray triangulation and RANSAC initial plane, robust dense photometric refinement and spatial checkerboard consistency. No test body or foot labels used. Estimated floor, not independently measured gravity.'))
