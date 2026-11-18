import numpy as np
from scipy.spatial.transform import Rotation as R
z=np.load('outputs/gen_head_camera_alignment_20261118/parsed_body_vio.npz');p=z['positions'];q=z['quaternions_xyzw'];rr=R.from_quat(q.reshape(-1,4)).as_matrix().reshape(-1,22,3,3);pa=[-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19]
for kind in ('global','local','root_global_rest_global'):
 r=rr.copy()
 if kind=='local':
  for j in range(1,22):r[:,j]=r[:,pa[j]]@rr[:,j]
 if kind=='root_global_rest_global':
  for j in range(1,22):r[:,j]=r[:,0]@rr[:,j]
 for who in ('parent','child'):
  errs=[]
  for j in range(1,22):
   d=p[:,j]-p[:,pa[j]];d/=np.linalg.norm(d,axis=-1)[:,None];v=np.einsum('nji,nj->ni',r[:,pa[j] if who=='parent' else j],d);mu=v.mean(0);mu/=np.linalg.norm(mu);e=np.degrees(np.arccos(np.clip(v@mu,-1,1)));errs.append(e.mean())
  print(kind,who,np.round(errs,3),flush=True)
