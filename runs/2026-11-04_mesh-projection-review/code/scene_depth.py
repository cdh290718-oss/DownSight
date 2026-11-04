"""Conservative sparse scene-depth mask from stereo images, for display only."""
import argparse,json
from pathlib import Path
import cv2,numpy as np
from rasterizer import project
p=argparse.ArgumentParser();p.add_argument('--selection',type=Path,required=True);p.add_argument('--images',type=Path,required=True);p.add_argument('--calibration',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);cal=json.loads(a.calibration.read_text());W,H=800,650;f=260.;K=np.array([[f,0,W/2],[0,f,H/2],[0,0,1.]]);R=np.array(cal['stereo_R32']);t=np.array(cal['stereo_t32']);R1,R2,P1,P2,Q,_,_=cv2.stereoRectify(K,np.zeros(5),K,np.zeros(5),(W,H),R.T,-R.T@t,flags=cv2.CALIB_ZERO_DISPARITY,alpha=0,newImageSize=(W,H));y,x=np.indices((H,W),dtype=np.float64)
def map_cam(rr,P,ds):
 ray=np.stack(((x-P[0,2])/P[0,0],(y-P[1,2])/P[1,1],np.ones_like(x)),axis=-1)@rr;u=project(ray,ds);return u[:,:,0].astype(np.float32),u[:,:,1].astype(np.float32)
m1=map_cam(R1,P1,cal['cam2_ds']);m2=map_cam(R2,P2,cal['cam3_ds'])
def matcher(min_disp):return cv2.StereoSGBM_create(minDisparity=min_disp,numDisparities=160,blockSize=5,P1=8*25,P2=32*25,disp12MaxDiff=1,uniquenessRatio=10,speckleWindowSize=80,speckleRange=2,mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
l=matcher(0);r=matcher(-160);stats=[]
for entry in json.loads(a.selection.read_text())['selected']:
 frame=entry['frame'];ims=[cv2.imread(str(a.images/f'cam{cam}'/f'{frame:05d}.jpg')) for cam in (2,3)];assert all(im is not None for im in ims);rect=[cv2.remap(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),*m,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT) for im,m in zip(ims,(m1,m2))];disp=l.compute(*rect).astype(np.float32)/16.;rev=r.compute(rect[1],rect[0]).astype(np.float32)/16.;xx=np.rint(x-disp).astype(np.int32);goodx=(xx>=0)&(xx<W);xc=np.clip(xx,0,W-1);yy=np.arange(H)[:,None];rdisp=rev[yy,xc];img=rect[0].astype(np.float32);mean=cv2.boxFilter(img,-1,(9,9));var=cv2.boxFilter(img*img,-1,(9,9))-mean*mean;p3=cv2.reprojectImageTo3D(disp,Q).reshape(H,W,3)@R1;distance=np.linalg.norm(p3,axis=2);good=goodx&(disp>1)&(np.abs(disp+rdisp)<1.5)&(var>36)&np.isfinite(p3).all(-1)&(p3[:,:,2]>.1)&(distance<1.2)
 results={};st={}
 for c in (2,3):
  if c==2:ux,uy=m1;dep=distance
  else:ux=m2[0][yy,xc];uy=m2[1][yy,xc];dep=np.linalg.norm((p3-t)@R,axis=2)
  px=np.rint(ux[good]).astype(int);py=np.rint(uy[good]).astype(int);z=dep[good].astype(np.float32);h,w=ims[c-2].shape[:2];ok=(px>=0)&(px<w)&(py>=0)&(py<h)&np.isfinite(z);zbuf=np.full(h*w,np.inf,np.float32);np.minimum.at(zbuf,py[ok]*w+px[ok],z[ok]);zbuf=zbuf.reshape(h,w);zbuf=cv2.erode(zbuf,np.ones((3,3),np.uint8));results[f'cam{c}_depth']=zbuf;st[f'cam{c}_valid_pixels']=int(np.isfinite(zbuf).sum())
 np.savez_compressed(a.output/f'{frame:05d}.npz',**results);stats.append({'frame':frame,'rectified_consistent_samples':int(good.sum()),**st});print('frame',frame,st,flush=True)
(a.output/'stats.json').write_text(json.dumps({'note':'Sparse stereo image depth, LR check and 3px splat; approximate visibility only; no ground-truth scene depth.','frames':stats},indent=2)+'\n')
