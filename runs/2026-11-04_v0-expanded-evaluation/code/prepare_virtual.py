"""Reproject existing native Gen images into the historical network virtual camera."""
import argparse,json
from pathlib import Path
import cv2,numpy as np
p=argparse.ArgumentParser();p.add_argument('--images',type=Path,required=True);p.add_argument('--calibration',type=Path,required=True);p.add_argument('--virtual-calibration',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
cal=json.loads(a.calibration.read_text());vcal=json.loads(a.virtual_calibration.read_text());fx,fy,cx,cy,xi,al=vcal['sourceParams'];y,x=np.indices((720,1280),dtype=np.float64);u=(x-cx)/fx;v=(y-cy)/fy;r2=u*u+v*v;mz=(1-al*al*r2)/(al*np.sqrt(np.maximum(0,1-(2*al-1)*r2))+1-al);k=(mz*xi+np.sqrt(np.maximum(0,mz*mz+(1-xi*xi)*r2)))/(mz*mz+r2);rays=np.stack([k*u,k*v,k*mz-xi],axis=-1);R=np.array(cal['cam3_R_virtual_from_gen']);rays=rays@R;X,Y,Z=np.moveaxis(rays,-1,0);fx,fy,cx,cy,xi,al=cal['cam2_ds'];d=np.sqrt(X*X+Y*Y+Z*Z);z1=xi*d+Z;den=al*np.sqrt(X*X+Y*Y+z1*z1)+(1-al)*z1;mx=(fx*X/den+cx).astype(np.float32);my=(fy*Y/den+cy).astype(np.float32)
for pth in sorted((a.images/'cam2').glob('*.jpg')):
 im=cv2.imread(str(pth));assert im is not None;dest=a.output/('gen_'+pth.stem+'.png');assert cv2.imwrite(str(dest),cv2.remap(im,mx,my,cv2.INTER_LINEAR));print('done',pth.stem,flush=True)
