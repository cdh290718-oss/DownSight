"""Numerical convention audit. Hypotheses are recorded, not silently treated as truth."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as Rot
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'outputs/gen_head_camera_alignment_20261118';RUN=Path(__file__).resolve().parents[1]
def read(ch):return json.loads((OUT/f'channel_{ch}.json').read_text())
def vec(d,keys):return [d.get(k,0.) for k in keys]
def pose(p):return vec(p['position'],'xyz'),vec(p['orientation'],'xyzw')
def stats(x):
 x=np.asarray(x);return dict(count=int(x.size),min=float(x.min()),mean=float(x.mean()),median=float(np.median(x)),p95=float(np.percentile(x,95)),max=float(x.max()))
def rigid(a,b):
 am=a.mean(0);bm=b.mean(0);u,s,v=np.linalg.svd((a-am).T@(b-bm));r=v.T@np.diag([1,1,np.linalg.det(v.T@u.T)])@u.T;return r,bm-r@am

def main():
 rows=read(6);P=np.array([[pose(p)[0] for p in x['message']['poses']] for x in rows]);Q=np.array([[pose(p)[1] for p in x['message']['poses']] for x in rows]);vio=read(11);V=np.array([pose(x['message']['pose'])[0] for x in vio]);VQ=np.array([pose(x['message']['pose'])[1] for x in vio]);VR=Rot.from_quat(VQ).as_matrix();z=np.load(ROOT.parent/'gen_egosmplx_run/dfm_reference_virtual.npz');frames=z['frames'];rawc={v:read(ch)[0]['message'] for v,ch in [('cam3',2),('cam4',14)]};out={}
 times={str(c):np.array([x['log_ns'] for x in read(c)],np.int64) for c in [1,6,11,12,13]}
 out['temporal']={c:{'count':len(t),'duration_s':float((t[-1]-t[0])/1e9),'nonincreasing_steps':int((np.diff(t)<=0).sum()),'step_ms':stats(np.diff(t)/1e6)} for c,t in times.items()}
 out['temporal']['same_index_offset_ms']={c:stats((t-times['1'])/1e6) for c,t in times.items() if len(t)==len(times['1'])}
 out['body']={'pose_counts':sorted(set(len(x['message']['poses']) for x in rows)),'frame_ids':sorted(set(x['message']['frame_id'] for x in rows)),'quaternion_norm_error':stats(abs(np.linalg.norm(Q,axis=-1)-1)),'finite_positions':bool(np.isfinite(P).all()),'finite_quaternions':bool(np.isfinite(Q).all())}
 out['cameras']={};mapping={};orientation={}
 for view,c in rawc.items():
  rv=z[view+'_R_virtual_from_gen'];a=z[view+'_body_xyz'];b=P[frames]
  fits=[];rs=[];ts=[]
  for aw,bw in zip(a,b):
   r,t=rigid(aw,bw);rs.append(r);ts.append(t);fits.extend(np.linalg.norm(aw@r.T+t-bw,axis=-1))
  rs=np.array(rs);ts=np.array(ts);mapping[view]=(rs,ts)
  out['cameras'][view]={'raw':{k:c[k] for k in ['width','height','distortion_model','D','K','R','P','frame_id','T_b_c']},'R_virtual_from_raw':rv.tolist(),'virtual_rotation_deg':float(Rot.from_matrix(rv).magnitude()*180/np.pi),'archived_world_body_rigid_match_mm':stats(np.array(fits)*1000)}
  # Compare recovered world<-raw (body-derived diagnostic only) with VIO + calibration hypotheses.
  rwr=rs@rv;twr=ts;test=[]
  for layout in ['t_xyz_q_xyzw','t_xyz_q_wxyz']:
   t=np.array(c['T_b_c'][:3]);q=np.array(c['T_b_c'][3:]);q=q if layout.endswith('xyzw') else q[[1,2,3,0]];r=Rot.from_quat(q).as_matrix()
   for inverse in [False,True]:
    rc=r.T if inverse else r;tc=-r.T@t if inverse else t
    rpred=VR[frames]@rc;tpred=V[frames]+np.einsum('nij,j->ni',VR[frames],tc)
    test.append({'layout':layout,'invert_T':inverse,'rotation_deg':stats(Rot.from_matrix(np.swapaxes(rpred,1,2)@rwr).magnitude()*180/np.pi),'translation_mm':stats(np.linalg.norm(tpred-twr,axis=-1)*1000)})
  out['cameras'][view]['VIO_extrinsic_hypotheses_vs_archive']=test
  rbv=np.swapaxes(VR[frames],1,2)@rwr;tbv=np.einsum('nji,nj->ni',VR[frames],twr-V[frames]);rm=Rot.from_matrix(rbv).mean();out['cameras'][view]['archive_implied_T_body_from_raw']={'R_mean':rm.as_matrix().tolist(),'t_mean_m':tbv.mean(0).tolist(),'R_spread_deg':stats((rm.inv()*Rot.from_matrix(rbv)).magnitude()*180/np.pi),'t_spread_mm':stats(np.linalg.norm(tbv-tbv.mean(0),axis=-1)*1000)}
 # Compare whether quaternions describe local/global joint orientations via bone invariance.
 parents=[-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19]
 for layout in ['xyzw','wxyz_in_xyzw_fields']:
  qs=Q if layout=='xyzw' else Q[:,:,[1,2,3,0]]
  local=Rot.from_quat(qs.reshape(-1,4)).as_matrix().reshape(len(P),22,3,3)
  for mode in ['as_global','FK_from_local']:
   r=local.copy()
   if mode=='FK_from_local':
    for j in range(1,22):r[:,j]=r[:,parents[j]]@local[:,j]
   bones=[]
   for j in range(1,22):
    delta=P[:,j]-P[:,parents[j]];d=np.einsum('nji,nj->ni',r[:,parents[j]],delta);norm=np.linalg.norm(d,axis=-1);dn=d/norm[:,None];mean=dn.mean(0);mean/=np.linalg.norm(mean);angle=np.rad2deg(np.arccos(np.clip(dn@mean,-1,1)))
    bones.append({'parent':parents[j],'child':j,'direction_spread_deg':stats(angle),'length_m':stats(norm)})
   rel=np.swapaxes(VR,1,2)@r[:,15];mean=Rot.from_matrix(rel).mean();orientation[layout+'_'+mode]={'bones':bones,'head_relative_VIO_rotation_spread_deg':stats((mean.inv()*Rot.from_matrix(rel)).magnitude()*180/np.pi),'head_relative_VIO_mean_R':mean.as_matrix().tolist()}
 out['orientation_hypotheses']=orientation
 (OUT/'parsed_body_vio.npz').write_bytes(b'') if False else None
 np.savez(OUT/'parsed_body_vio.npz',positions=P,quaternions_xyzw=Q,vio_positions=V,vio_quaternions_xyzw=VQ,frames=frames,**{v+'_R_world_from_virtual':m[0] for v,m in mapping.items()},**{v+'_t_world_from_virtual':m[1] for v,m in mapping.items()})
 out['command']='<PYTHON_ENV>/bin/python runs/2026-11-18_gen_head_camera_alignment/code/analyze.py';out['exit_status']=0
 (RUN/'evidence/alignment.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps({k:v for k,v in out.items() if k!='orientation_hypotheses'},indent=2))
 for key,v in orientation.items():print(key,'head spread',v['head_relative_VIO_rotation_spread_deg'],'mean bone spread',np.mean([b['direction_spread_deg']['mean'] for b in v['bones']]))
if __name__=='__main__':main()
