"""Noiseless synthetic Double Sphere projection and ray-triangulation unit check."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--calibration',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();c=json.loads(a.calibration.read_text());R=np.array(c['stereo_R32']);t=np.array(c['stereo_t32']);d=[np.array(c['cam2_ds']),np.array(c['cam3_ds'])];rng=np.random.default_rng(9);xyz=np.c_[rng.uniform(-.2,.2,1000),rng.uniform(-.15,.2,1000),rng.uniform(.2,1.,1000)];q=(xyz-t)@R
def project(v,d):
 fx,fy,cx,cy,xi,alpha=d;x,y,z=v.T;d1=np.sqrt(x*x+y*y+z*z);z1=xi*d1+z;d2=np.sqrt(x*x+y*y+z1*z1);den=alpha*d2+(1-alpha)*z1;return np.c_[fx*x/den+cx,fy*y/den+cy]
def unproject(uv,d):
 fx,fy,cx,cy,xi,alpha=d;mx=(uv[:,0]-cx)/fx;my=(uv[:,1]-cy)/fy;r2=mx*mx+my*my;disc=1-(2*alpha-1)*r2;assert (disc>0).all();mz=(1-alpha*alpha*r2)/(alpha*np.sqrt(disc)+1-alpha);inside=mz*mz+(1-xi*xi)*r2;assert (inside>0).all();k=(mz*xi+np.sqrt(inside))/(mz*mz+r2);v=np.c_[k*mx,k*my,k*mz-xi];return v/np.linalg.norm(v,axis=1,keepdims=True)
x=unproject(project(xyz,d[0]),d[0]);y=unproject(project(q,d[1]),d[1])@R.T;z=(x*y).sum(1);u=x@t;v=y@t;den=1-z*z;aa=(u-z*v)/den;bb=(z*u-v)/den;pred=(aa[:,None]*x+t+bb[:,None]*y)*.5;err=np.linalg.norm(pred-xyz,axis=1);assert err.max()<1e-8
with a.output.open('x') as f:json.dump({'test':'noiseless DS project/unproject/stereo triangulation','random_seed':9,'points':len(err),'max_error_m':float(err.max()),'median_error_m':float(np.median(err)),'status':'passed','note':'Algebra-only test; it does not test real correspondence accuracy or full fisheye occlusion.'},f,indent=2);f.write('\n')
