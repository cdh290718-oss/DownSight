"""Synthetic geometry validation for the actual polynomial stereo cameras."""
import argparse,json
from pathlib import Path
import numpy as np
from prepare import triangulate,write,read,Camera
p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();m=read(a.manifest);frames=m['pairs'][0]['frames'];cams=[Camera(f['calibration']) for f in frames];A=np.array(read(m['calibration_source'])['poseCamera2']['A']);R=A[:3,:3];t=A[:3,3]/1000;cases=[]
for x in [-.15,0.05,.15]:
 for y in [-.1,.1,.2]:
  for z in [.4,.7,1.]:
   pt=np.array([x,y,z]);uv=[cams[0].world2camera(pt[None])[0],cams[1].world2camera((R@pt+t)[None])[0]];q=triangulate(cams,R,t,uv)
   if q['accepted']:
    error=float(np.linalg.norm(np.array(q['point_cam3_m'])-pt));C=np.array(q['covariance_cam3_m2']);rotation_error=float(np.max(np.abs(np.linalg.eigvalsh(R@C@R.T)-np.linalg.eigvalsh(C))));cases.append(dict(point=pt.tolist(),error_m=error,covariance_rotation_eigenvalue_error=rotation_error))
assert len(cases)>10 and max(q['error_m'] for q in cases)<1e-6
write(a.out,dict(passed=True,cases=cases,maximum_error_m=max(q['error_m'] for q in cases)));print('PASS',len(cases),'synthetic points')
