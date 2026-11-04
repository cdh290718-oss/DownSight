"""Verify scene stereo rectification and fisheye mapping without labels."""
import argparse,json
from pathlib import Path
import cv2,numpy as np
from rasterizer import project
p=argparse.ArgumentParser();p.add_argument('--calibration',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();cal=json.loads(a.calibration.read_text());W,H=800,650;f=260.;K=np.array([[f,0,W/2],[0,f,H/2],[0,0,1.]]);R=np.array(cal['stereo_R32']);t=np.array(cal['stereo_t32']);R1,R2,P1,P2,Q,_,_=cv2.stereoRectify(K,np.zeros(5),K,np.zeros(5),(W,H),R.T,-R.T@t,flags=cv2.CALIB_ZERO_DISPARITY,alpha=0,newImageSize=(W,H));errors=[]
for y in (150,300,450):
 for x in (300,400,500,600):
  for d in (20,40,80):
   v=Q@np.array([x,y,d,1.]);v=v[:3]/v[3];p2=v@R1;p3=(p2-t)@R;r2=np.array([(x-P1[0,2])/P1[0,0],(y-P1[1,2])/P1[1,1],1.])@R1;r3=np.array([(x-d-P2[0,2])/P2[0,0],(y-P2[1,2])/P2[1,1],1.])@R2;e2=float(np.linalg.norm(project(p2,cal['cam2_ds'])-project(r2,cal['cam2_ds'])));e3=float(np.linalg.norm(project(p3,cal['cam3_ds'])-project(r3,cal['cam3_ds'])));errors.append([e2,e3])
err=np.array(errors);assert np.max(err)<1e-5;a.output.write_text(json.dumps({'samples':len(errors),'max_reprojection_difference_px_cam2_cam3':err.max(0).tolist(),'method':'project Q-reconstructed rays and compare independent native DS rectification mapping; no labels','pass':True},indent=2)+'\n');print('PASS',len(errors),err.max(0))
