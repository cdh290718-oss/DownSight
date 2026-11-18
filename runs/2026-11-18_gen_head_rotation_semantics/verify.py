"""Test whether stored orientations carry information beyond parent-to-child directions."""
import json,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
ROOT=Path(__file__).resolve().parents[2];RUN=Path(__file__).resolve().parent
z=np.load(ROOT/'outputs/gen_head_camera_alignment_20261118/parsed_body_vio.npz');p=z['positions'];q=z['quaternions_xyzw'];parents=[-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19]
r=R.from_quat(q.reshape(-1,4)).as_matrix().reshape(-1,22,3,3)
d=p[:,1:]-p[:,np.array(parents[1:])];d/=np.linalg.norm(d,axis=-1)[...,None]
pitch=np.arcsin(np.clip(d[...,1],-1,1));yaw=np.arctan2(-d[...,0],-d[...,2]);generated=R.from_euler('y',yaw.ravel())*R.from_euler('x',pitch.ravel());stored=R.from_quat(q[:,1:].reshape(-1,4));error=np.degrees((generated.inv()*stored).magnitude()).reshape(-1,21)
def stats(a):
 a=np.asarray(a);return {'count':int(a.size),'mean':float(a.mean()),'max':float(a.max()),'p95':float(np.percentile(a,95))}
# Independent held-out frames after first half: same formula, no estimated parameter.
cut=len(p)//2
axiserr=np.linalg.norm(-r[:,1:,:,2]-d,axis=-1)
head=R.from_quat(q[:,15]);twisted=head*R.from_euler('z',90,degrees=True)
ambiguity=np.linalg.norm(head.apply([0,0,-1])-twisted.apply([0,0,-1]),axis=-1)
assert error.max()<1e-9 and axiserr.max()<1e-12 and ambiguity.max()<1e-12
out={'command':'<PYTHON_ENV>/bin/python runs/2026-11-18_gen_head_rotation_semantics/verify.py','exit_status':0,'source_mcap_sha256':'00f34e66eacf623d03564571608665a5f6893be102f42c492785697c376e95c1','decoded_input_sha256':hashlib.sha256((ROOT/'outputs/gen_head_camera_alignment_20261118/parsed_body_vio.npz').read_bytes()).hexdigest(),'frames':len(p),'nonroot_joints':21,'head_index':15,'parent_index':12,'formula':'d=normalize(p_child-p_parent); pitch=asin(d_y); yaw=atan2(-d_x,-d_z); R=Ry(yaw) @ Rx(pitch)','fitted_parameters':0,'all_orientation_reconstruction_deg':stats(error),'heldout_second_half_reconstruction_deg':stats(error[cut:]),'head_orientation_reconstruction_deg':stats(error[:,14]),'per_joint_max_deg':error.max(0).tolist(),'minus_z_direction_residual':stats(axiserr),'fixed_world_y_horizontal_axis_R10_abs':stats(abs(r[:,1:,1,0])),'nonidentifiability_example':{'local_z_twist_deg':90,'head_bone_direction_change':stats(ambiguity),'orientation_change_deg':stats(np.degrees((head.inv()*twisted).magnitude()))},'conclusion':'Stored nonroot orientation is exactly reconstructible from incoming bone direction alone. It contains no independent anatomical head twist measurement. A fixed axis conversion cannot supply missing head twist. Production implementation not inspected; numerical equality is the evidence.'}
(RUN/'evidence/rotation_semantics.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
