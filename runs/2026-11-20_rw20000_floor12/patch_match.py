"""Depth sweep for independent calibrated stereo correspondences on faint carpet."""
import cv2
import numpy as np
from assess import ransac, refine, unit, angle

# Distances along a unit camera ray in meters.
INVERSE_RANGE = np.linspace(1 / 4., 1 / .55, 125)


def remap(image, pixels):
    return cv2.remap(image, pixels[...,0].astype('f'), pixels[...,1].astype('f'),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def search(centers, left, right, cameras, rotation, translation):
    # A patch tangent to each central ray warps into the other fisheye image.
    offsets = np.arange(-8, 9, 2)
    patch = np.stack(np.meshgrid(offsets, offsets), axis=-1).reshape(-1,2)
    px = centers[:,None] + patch
    rays = cameras[0].camera2world_ray(px.reshape(-1,2)).reshape(len(centers),-1,3)
    central = rays[:,len(patch)//2]
    rays = rays / np.sum(rays*central[:,None], axis=-1, keepdims=True)
    rotated = rays.reshape(-1,3) @ rotation.T
    template = remap(left, px)
    template -= template.mean(1,keepdims=True)
    template /= np.maximum(np.linalg.norm(template,axis=1,keepdims=True),1.e-6)
    scores = []
    for inv in INVERSE_RANGE:
        xy = cameras[1].world2camera(rotated + inv*translation).reshape(px.shape)
        data = remap(right,xy)
        data -= data.mean(1,keepdims=True)
        scores.append(np.sum(template*data,axis=1)/np.maximum(1.e-6,np.linalg.norm(data,axis=1)))
    scores = np.array(scores).T
    ii = np.arange(len(centers))
    idx = np.argmax(scores,axis=1)
    maxscore = scores[ii,idx]
    distant = np.abs(np.arange(len(INVERSE_RANGE))[None,:] - idx[:,None]) >= 8
    distinct = maxscore - np.max(np.where(distant,scores,-2.),axis=1)
    ok = (idx>0)&(idx<len(INVERSE_RANGE)-1)&(maxscore>.82)&(distinct>.055)
    s = np.clip(idx,1,len(INVERSE_RANGE)-2)
    a,b,c = scores[ii,s-1],scores[ii,s],scores[ii,s+1]
    correction = np.clip((a-c)/(2*np.minimum(a+c-2*b,-1e-6)),-.5,.5)
    distances = 1/(INVERSE_RANGE[idx]+correction*(INVERSE_RANGE[1]-INVERSE_RANGE[0]))
    points = central*distances[:,None]
    return points, cameras[1].world2camera(points@rotation.T+translation), ok


def estimate(images,cameras,T,polys):
    gray = [cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) for im in images]
    mask = np.zeros_like(gray[0])
    cv2.fillPoly(mask,polys,255)
    interior = cv2.erode(mask,np.ones((19,19),np.uint8))
    pts = cv2.goodFeaturesToTrack(gray[0],450,.0002,9,mask=interior,blockSize=5)
    if pts is None:
        raise ValueError('No patch texture')
    centers = pts.reshape(-1,2)
    rot,t = T[:3,:3],T[:3,3]
    ims = [im.astype(np.float32) for im in gray]
    xyz, right, ok = search(centers,*ims,cameras,rot,t)
    back, left, reverse = search(right,ims[1],ims[0],cameras[::-1],rot.T,-rot.T@t)
    valid = ok & reverse & (np.linalg.norm(left-centers,axis=1)<.85)
    cloud = xyz[valid]
    if len(cloud)<3:
        raise ValueError('Fewer than three reciprocal correspondences')
    start,inside = ransac(cloud)
    y,x = np.nonzero(mask); uv = np.column_stack([x,y])[::5]
    half = (uv[:,0]//36+uv[:,1]//36)%2==0
    p,a,b = [refine(start,q,[im.astype(float) for im in gray],cameras,rot,t) for q in [uv,uv[half],uv[~half]]]
    up = -unit(p); height = 1/np.linalg.norm(p)
    distance_gap = abs(1/np.linalg.norm(a)-1/np.linalg.norm(b))*1000
    median = np.median(abs(cloud@up+height))*1000
    cells = len(np.unique(np.floor(centers[valid][inside]/36),axis=0))
    checker = angle(a,b)
    sparse_dense = angle(start,p)
    ok = bool(inside.sum()>=8 and cells>=5 and .45<height<3 and up[2]<0 and checker<5 and distance_gap<50 and median<50 and sparse_dense<10)
    quality = dict(sparse_matches=len(cloud),ransac_inliers=int(inside.sum()),dense_pixels=len(uv),
                   checker_angle_deg=checker, checker_distance_diff_mm=distance_gap,
                   sparse_median_residual_mm=median,sparse_vs_dense_deg=sparse_dense,
                   patch_candidates=len(centers),inlier_cells=cells,accepted=ok)
    return up,height,quality,centers[valid][inside]
