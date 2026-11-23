"""Texture-only coarse-to-fine plane sweep for masks with insufficient sparse support."""
import cv2
import numpy as np
from plane import angle, norm_image, refine, sample, unit

def estimate(images,cams,T,polys):
    gray=[cv2.cvtColor(im,cv2.COLOR_BGR2GRAY).astype(float) for im in images]
    mask=np.zeros(gray[0].shape,np.uint8)
    cv2.fillPoly(mask,polys,255)
    mask=cv2.erode(mask,np.ones((9,9),np.uint8))
    y,x=np.nonzero(mask)
    pts=np.column_stack([x,y])[::4]
    checker=((pts[:,0]//36+pts[:,1]//36)%2)==0
    ims=[norm_image(im,1.5) for im in gray]
    R,t=T[:3,:3],T[:3,3]
    # Sweep signed plane normals covering all azimuths, then refine the best
    # supported warp independently on each checkerboard half.
    dirs=[]
    for deg in [0,20,40,60,78]:
        az=np.arange(0,360,30) if deg else [0]
        d=np.deg2rad(deg)
        dirs.extend([[np.sin(d)*np.cos(np.deg2rad(a)),np.sin(d)*np.sin(np.deg2rad(a)),np.cos(d)] for a in az])
    planes=np.array([n/h for n in np.array(dirs) for h in (.5,.7,.95,1.2,1.5,1.8,2.2)])
    def fit(pixels):
        u=pixels[::max(1,len(pixels)//600)]
        rays=cams[0].camera2world_ray(u.astype(float))
        left=sample(ims[0],u)
        best=None
        for plane in planes:
            inv=rays@plane
            if np.quantile(inv,.02)<.14:
                continue
            dest=cams[1].world2camera((rays@R.T)/inv[:,None]+t)
            if np.any(dest<1) or np.any(dest>=np.array(gray[1].shape[::-1])-2):
                continue
            err=np.mean(np.minimum(abs(left-sample(ims[1],dest)),1.25))
            if best is None or err<best[0]:
                best=(err,plane)
        if best is None:
            raise ValueError('No valid coarse stereo floor warp')
        return refine(best[1],pixels,gray,cams,R,t)
    a,b=fit(pts[checker]),fit(pts[~checker])
    p=refine((a+b)/2,pts,gray,cams,R,t)
    normal=-unit(p)
    h=1/np.linalg.norm(p)
    q=dict(accepted=False,checker_angle_deg=angle(a,b),checker_distance_diff_mm=abs(1/np.linalg.norm(a)-1/np.linalg.norm(b))*1000,
           dense_pixels=len(pts),validation='Provisional photometric estimate; insufficient independent sparse support.')
    return normal,h,q
