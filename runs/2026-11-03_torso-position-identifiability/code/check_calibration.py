"""Estimate relative rotation and translation direction from training images only."""
import argparse,json
from pathlib import Path
import cv2,numpy as np
p=argparse.ArgumentParser();p.add_argument('--images',type=Path,required=True);p.add_argument('--calibration',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
c=json.loads(a.calibration.read_text());ds=[c['cam2_ds'],c['cam3_ds']];orb=cv2.ORB_create(nfeatures=5000,fastThreshold=12);matcher=cv2.BFMatcher(cv2.NORM_HAMMING)
def rays(uv,d):
 fx,fy,cx,cy,xi,alpha=d;mx=(uv[:,0]-cx)/fx;my=(uv[:,1]-cy)/fy;r2=mx*mx+my*my;disc=1-(2*alpha-1)*r2;mz=(1-alpha*alpha*r2)/(alpha*np.sqrt(np.maximum(disc,0))+1-alpha);inside=mz*mz+(1-xi*xi)*r2;k=(mz*xi+np.sqrt(np.maximum(inside,0)))/(mz*mz+r2);v=np.stack((k*mx,k*my,k*mz-xi),1);return v,(disc>0)&(inside>0)&(v[:,2]>.2)
a1=[];b1=[];frame_counts=[]
for frame in range(0,2700,270):
 ims=[cv2.imread(str(a.images/f'cam{cam}'/f'{frame:05d}.jpg'),0) for cam in (2,3)]; feats=[orb.detectAndCompute(im,None) for im in ims];matches=matcher.knnMatch(feats[0][1],feats[1][1],k=2);good=[m for m,n in matches if m.distance<.7*n.distance]; uv=[np.array([feats[i][0][(m.queryIdx if i==0 else m.trainIdx)].pt for m in good]) for i in (0,1)];v0,ok0=rays(uv[0],ds[0]);v1,ok1=rays(uv[1],ds[1]);ok=ok0&ok1;a1.extend(v0[ok,:2]/v0[ok,2,None]);b1.extend(v1[ok,:2]/v1[ok,2,None]);frame_counts.append(len(good))
x=np.array(a1);y=np.array(b1);cv2.setRNGSeed(1);E,mask=cv2.findEssentialMat(x,y,np.eye(3),method=cv2.RANSAC,prob=.999,threshold=.002);count,R,t,posemask=cv2.recoverPose(E,x,y,np.eye(3),mask=mask.copy());expectedR=np.array(c['stereo_R32']).T;expectedt=-expectedR@np.array(c['stereo_t32']);t=t.ravel();angle=lambda z:float(np.degrees(np.arccos(np.clip((np.trace(z)-1)/2,-1,1))));direction=float(np.degrees(np.arccos(np.clip(t@expectedt/(np.linalg.norm(t)*np.linalg.norm(expectedt)),-1,1))))
out={'training_frames':list(range(0,2700,270)),'matches_per_frame':frame_counts,'rays':len(x),'ransac_inliers':int(mask.sum()),'pose_inliers':int(count),'R_cam2_to_cam3':R.tolist(),'translation_direction_cam3':t.tolist(),'rotation_disagreement_deg':angle(R@expectedR.T),'translation_direction_disagreement_deg':direction,'note':'Training image feature matches; translation scale cannot be recovered from images alone. Comparison is to supplied calibration; this test does not establish its provenance or ground truth.'}
with a.output.open('x') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
