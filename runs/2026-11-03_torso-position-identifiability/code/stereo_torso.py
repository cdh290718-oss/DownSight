"""Rectify calibrated fisheye images and derive sparse samples of dense stereo clothing/torso observations, without labels."""
import argparse,json
from pathlib import Path
import cv2,numpy as np
p=argparse.ArgumentParser();p.add_argument('--images',type=Path,required=True);p.add_argument('--seg',type=Path,required=True);p.add_argument('--calibration',type=Path,required=True);p.add_argument('--frames',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
cal=json.loads(a.calibration.read_text());obs=np.load(a.frames,allow_pickle=False);frames=obs['frames']; W,H=800,650; f=260.;K=np.array([[f,0,W/2],[0,f,H/2],[0,0,1.]])
R=np.array(cal['stereo_R32']); t=np.array(cal['stereo_t32']);r21=R.T;t21=-R.T@t
R1,R2,P1,P2,Q,_,_=cv2.stereoRectify(K,np.zeros(5),K,np.zeros(5),(W,H),r21,t21,flags=cv2.CALIB_ZERO_DISPARITY,alpha=0,newImageSize=(W,H))
y,x=np.indices((H,W),dtype=np.float64)
def map_cam(rr,P,ds):
 ray=np.stack(((x-P[0,2])/P[0,0],(y-P[1,2])/P[1,1],np.ones_like(x)),axis=-1)@rr
 X,Y,Z=np.moveaxis(ray,-1,0);fx,fy,cx,cy,xi,alpha=ds;d1=np.sqrt(X*X+Y*Y+Z*Z);z1=xi*d1+Z;d2=np.sqrt(X*X+Y*Y+z1*z1);den=alpha*d2+(1-alpha)*z1
 u=fx*X/np.maximum(den,1e-12)+cx;v=fy*Y/np.maximum(den,1e-12)+cy
 u=np.where(den>0,u,-1);v=np.where(den>0,v,-1)
 return u.astype(np.float32),v.astype(np.float32)
m1=map_cam(R1,P1,cal['cam2_ds']);m2=map_cam(R2,P2,cal['cam3_ds'])
# Reverse disparity independently estimates d_right=x_right-x_left.
def matcher(min_disp):return cv2.StereoSGBM_create(minDisparity=min_disp,numDisparities=160,blockSize=5,P1=8*25,P2=32*25,disp12MaxDiff=1,uniquenessRatio=10,speckleWindowSize=80,speckleRange=2,mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
l=matcher(0);r=matcher(-160);points=[];counts=[];stats=[]
for frame in frames:
 ims=[];masks=[]
 for cam,maps in [(2,m1),(3,m2)]:
  im=cv2.imread(str(a.images/f'cam{cam}'/f'{int(frame):05d}.jpg')); assert im is not None
  seg=np.load(a.seg/f'{int(frame):05d}_cam{cam}.npy',allow_pickle=False)
  ims.append(cv2.remap(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),*maps,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT))
  mask=np.isin(seg,[11,20,22,23]).astype(np.uint8); masks.append(cv2.remap(mask,*maps,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT))
 disp=l.compute(*ims).astype(np.float32)/16.; rev=r.compute(ims[1],ims[0]).astype(np.float32)/16.
 xx=np.rint(x-disp).astype(np.int32);goodx=(xx>=0)&(xx<W);xc=np.clip(xx,0,W-1);yy=np.arange(H)[:,None];rdisp=rev[yy,xc];rmask=masks[1][yy,xc]
 # Local variation and left-right agreement reject flat regions and ambiguous occlusion.
 img=ims[0].astype(np.float32);mean=cv2.boxFilter(img,-1,(9,9));var=cv2.boxFilter(img*img,-1,(9,9))-mean*mean
 points3=cv2.reprojectImageTo3D(disp,Q).reshape(H,W,3)@R1
 good=goodx&(disp>1)&(np.abs(disp+rdisp)<1.5)&(masks[0]>0)&(rmask>0)&(var>36)&np.isfinite(points3).all(-1)&(points3[:,:,2]>.10)&(np.linalg.norm(points3,axis=2)<1.2)
 # Uniform image-space grid, one point per 8x8 pixel cell; fixed seed and capped sample.
 sampled=good & ((x.astype(np.int32)%8)==4) & ((y.astype(np.int32)%8)==4);p3=points3[sampled]
 if len(p3)>512:p3=p3[np.linspace(0,len(p3)-1,512,dtype=int)]
 padded=np.zeros((512,3),np.float32);padded[:len(p3)]=p3;points.append(padded);counts.append(len(p3));
 stats.append({'frame':int(frame),'count':len(p3),'range_m_median':float(np.median(np.linalg.norm(p3,axis=1))) if len(p3) else None,'disparity_px_median':float(np.median(disp[sampled])) if len(p3) else None,'mask_pixels_cam2_cam3':[int(z.sum()) for z in masks]});print('DONE',stats[-1],flush=True)
with a.output.open('xb') as f:np.savez_compressed(f,frames=frames,points_cam2=np.array(points),counts=np.array(counts),rectification_R1=R1,rectification_R2=R2,P1=P1,P2=P2,Q=Q)
print('metadata',json.dumps({'method':'SGBM, both mask classes 11/20/22/23, 1.5 px left-right, 8px image sampling','stats':stats,'translation_magnitude_m':float(np.linalg.norm(t))}),flush=True)
