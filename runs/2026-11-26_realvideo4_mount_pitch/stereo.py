"""Independent appearance matches; retain residual outliers in the evidence."""
from pathlib import Path
import sys
import cv2
import numpy as np
RUN=Path(__file__).resolve().parent
sys.path.insert(0,str(RUN.parent/'2026-11-26_realvideo4_head_mount'))
from measure import image, T, rays, cams, write

def matches(im):
    detector=cv2.SIFT_create(nfeatures=4000)
    features=[]
    for frame in im:
        # Top half mainly contains the desk or room; exclude the wearer's face.
        gray=cv2.cvtColor(frame[:360],cv2.COLOR_BGR2GRAY)
        features.append(detector.detectAndCompute(gray,None))
    matcher=cv2.BFMatcher()
    def ratio(a,b):
        return {m.queryIdx:m.trainIdx for m,n in matcher.knnMatch(a,b,k=2) if m.distance<.72*n.distance}
    ab=ratio(features[0][1],features[1][1])
    ba=ratio(features[1][1],features[0][1])
    ids=[(i,j) for i,j in ab.items() if ba.get(j)==i]
    return [np.array([features[e][0][ij[e]].pt for ij in ids]) for e in range(2)]

def main():
    result={}
    for k in range(1,5):
        key=f'clip{k:03d}'
        a,b=matches([image(key,e) for e in range(2)])
        normals=np.cross(T[:3,3],rays(a,cams[0])@T[:3,:3].T)
        def residual(q):
            return (rays(q,cams[1])*normals).sum(1)
        grad=np.array([(residual(b+d)-residual(b-d))/.02 for d in [[.01,0],[0,.01]]])
        err=abs(residual(b))/np.maximum(np.linalg.norm(grad,axis=0),1e-12)
        result[key]=dict(n=len(a),median_px=np.median(err),p90_px=np.percentile(err,90),
                         pairs_cam3_cam4_px=np.stack([a,b],axis=1).tolist(),error_px=err.tolist())
        print(key,len(a),np.median(err),np.percentile(err,90))
    write(RUN/'evidence/stereo.json',dict(rows=result,exit_status=0))
if __name__=='__main__':main()
