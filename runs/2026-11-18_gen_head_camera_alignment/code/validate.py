"""Validate exported transforms, timestamp matching, and legacy input identity."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from analyze import read,stats,ROOT,OUT,RUN
from export_assets import project

assets=json.loads((OUT/'head_camera_20pairs.json').read_text());z=np.load(OUT/'parsed_body_vio.npz');old=np.load(ROOT.parent/'gen_egosmplx_run/dfm_reference_virtual.npz');head=np.load(OUT/'head_all_frames.npz');frames=z['frames'];checks={};summary={}
lt=np.array([x['log_ns'] for x in read(1)],np.int64)
for ch in (6,11,12,13):
 ts=np.array([x['log_ns'] for x in read(ch)],np.int64);idx=np.searchsorted(ts,lt);hi=np.clip(idx,0,len(ts)-1);lo=np.clip(idx-1,0,len(ts)-1);match=np.where(abs(ts[lo]-lt)<=abs(ts[hi]-lt),lo,hi);assert (match==np.arange(len(lt))).all();checks[f'channel_{ch}_nearest_is_same_index']=True
assert len(assets['pairs'])==len(frames)==20 and len(head['world_position'])==5379
checks['export_counts']=True
rworld=Rot.from_quat(z['vio_quaternions_xyzw']).as_matrix();pworld=z['vio_positions'];roundtrip=[];leg=[];uvround=[];rotvalid=[]
for view in ('cam3','cam4'):
 c=assets['cameras'][view];t=np.array(c['T_vio_from_virtual']);r=rworld@t[:3,:3];p=pworld+np.einsum('nij,j->ni',rworld,t[:3,3]);hp=head[view+'_position'];recovered=np.einsum('nij,nj->ni',r,hp)+p;roundtrip.extend(np.linalg.norm(recovered-z['positions'][:,15],axis=-1)*1000)
 # Hold one archive-derived rig transform fixed and check all remaining frames.
 r0=z[view+'_R_world_from_virtual'][0];p0=z[view+'_t_world_from_virtual'][0];rb=rworld[0].T@r0;tb=rworld[0].T@(p0-pworld[0]);rr=rworld[frames]@rb;tt=pworld[frames]+np.einsum('nij,j->ni',rworld[frames],tb);pred=np.einsum('nji,nkj->nki',rr,z['positions'][frames]-tt[:,None,:]);leg.extend(np.linalg.norm(pred-old[view+'_body_xyz'],axis=-1).ravel()*1000)
 u,_=project(old[view+'_body_xyz'],c['virtual_params_fx_fy_cx_cy_xi_alpha']);uvround.extend(np.linalg.norm(u-old[view+'_body_uv'],axis=-1).ravel())
 rh=head[view+'_R_schema_interpretation'];rotvalid.extend(np.linalg.norm(np.swapaxes(rh,1,2)@rh-np.eye(3),axis=(1,2)))
 summary[view+'_head_new_vs_archive_position_shift_mm']=stats(np.linalg.norm(hp[frames]-old[view+'_body_xyz'][:,15],axis=-1)*1000)
 assert np.allclose(np.linalg.det(rh),1)
assert max(roundtrip)<1e-8 and max(leg)<1e-8 and max(uvround)<1e-7 and max(rotvalid)<1e-10
checks.update(official_head_roundtrip=True,first_frame_legacy_rig_reproduces_all_20_pairs=True,legacy_double_sphere_projection=True,all_rotation_matrices_SO3=True)
summary.update(official_head_roundtrip_mm=stats(roundtrip),legacy_first_frame_rig_holdout_mm=stats(leg),legacy_projection_roundtrip_px=stats(uvround),rotation_orthogonality=stats(rotvalid))
# Head center is not the VIO IMU center; descriptive DFM reference statistic only.
hdev=np.einsum('nji,nj->ni',rworld,z['positions'][:,15]-pworld)
summary['head_to_device_origin_distance_mm']=stats(np.linalg.norm(hdev,axis=-1)*1000)
summary['head_in_device_position_mean_m']=hdev.mean(0).tolist();summary['head_in_device_position_deviation_from_mean_mm']=stats(np.linalg.norm(hdev-hdev.mean(0),axis=-1)*1000)
for c in (8,9):
 data=read(c);summary[f'validity_channel_{c}']={'count':len(data)}
 if c==8:
  message=data[0]['message'];summary[f'validity_channel_{c}']['ranges']=[{'start_s':x.get('start_time_s',0.),'end_s':x['end_time_s'],'valid':x.get('is_valid',False)} for x in message['ranges']]
  summary[f'validity_channel_{c}']['invalid_reason_codes']=sorted(set(x['invalid_code'] for x in message['invalid_ranges']))
 if c==9:
  from collections import Counter
  # Field-wise value frequencies, no full metadata stream published.
  summary[f'validity_channel_{c}']['field_names']=sorted(set(k for x in data for k in x['message']))
  summary[f'validity_channel_{c}']['top_level_scalar_frequencies']={k:dict(Counter(str(x['message'].get(k,'<absent>')) for x in data)) for k in ['is_valid','valid'] if any(k in x['message'] for x in data)}
summary['input_hashes']=[{'resource':name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for name,path in [('raw_mcap',ROOT/'data/0d7da991acf44b46ba41710a8960209c.mcap'),('legacy_reference',ROOT.parent/'gen_egosmplx_run/dfm_reference_virtual.npz'),('legacy_manifest',ROOT.parent/'gen_egosmplx_run/frames.json')]]
assert summary['input_hashes'][0]['sha256']==assets['source_sha256'];checks['source_MCAP_unchanged']=True
r={'checks':checks,'summary':summary,'command':'<PYTHON_ENV>/bin/python runs/2026-11-18_gen_head_camera_alignment/code/validate.py','exit_status':0}
(RUN/'evidence/validation.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
