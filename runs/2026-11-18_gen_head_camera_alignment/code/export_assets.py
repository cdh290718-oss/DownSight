"""Export synchronized head candidates and official camera chain; preserve the legacy frame separately."""
import hashlib,json
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from analyze import read,stats,ROOT,OUT,RUN

R_BODY=np.array([[0.,0.,-1.],[1.,0.,0.],[0.,-1.,0.]])
def trans(r,t):
 a=np.eye(4);a[:3,:3]=r;a[:3,3]=t;return a

def project(p,d):
 fx,fy,cx,cy,xi,alpha=d
 n=np.linalg.norm(p,axis=-1);zx=p[...,2]+xi*n
 den=alpha*np.sqrt(p[...,0]**2+p[...,1]**2+zx**2)+(1-alpha)*zx
 uv=p[...,:2]/den[...,None]*[fx,fy]+[cx,cy]
 w1=(1-alpha)/alpha if alpha>.5 else alpha/(1-alpha);w2=(w1+xi)/np.sqrt(2*w1*xi+xi*xi+1)
 valid=(den>1e-8)&(p[...,2]>-w2*n)&np.isfinite(uv).all(-1)
 return uv,valid

def unproject(uv,d):
 fx,fy,cx,cy,xi,alpha=d;m=(uv-[cx,cy])/[fx,fy];r2=(m*m).sum(-1)
 disc=1-(2*alpha-1)*r2;valid=disc>=0
 mz=(1-alpha*alpha*r2)/(alpha*np.sqrt(np.maximum(disc,0))+1-alpha)
 k=(mz*xi+np.sqrt(np.maximum(mz*mz+(1-xi*xi)*r2,0)))/(mz*mz+r2)
 ray=k[...,None]*np.concatenate([m,mz[...,None]],-1);ray[...,2]-=xi
 return ray,valid

def main():
 z=np.load(OUT/'parsed_body_vio.npz');old=np.load(ROOT.parent/'gen_egosmplx_run/dfm_reference_virtual.npz');a=json.loads((RUN/'evidence/alignment.json').read_text());frames=z['frames'];p=z['positions'];q=z['quaternions_xyzw'];vp=z['vio_positions'];vr=Rot.from_quat(z['vio_quaternions_xyzw']).as_matrix();headR=Rot.from_quat(q[:,15]).as_matrix();times=read(1);rtimes=read(13);btime=read(6);vtime=read(11)
 report={'source_sha256':json.loads((RUN/'evidence/extraction.json').read_text())['source_sha256'],'head_index':15,'head_index_status':'matches prior SMPL BODY22 convention and official BODY_EDGES topology; original message has no joint names','rotation_status':'world orientation according to PosesInFrame schema, SMPL-X head axes not validated; do not treat as calibrated golden D anchor','cameras':{},'selected_pairs':[],'command':'<PYTHON_ENV>/bin/python runs/2026-11-18_gen_head_camera_alignment/code/export_assets.py','exit_status':0}
 assets={'source_sha256':report['source_sha256'],'head_index':15,'semantics':report['rotation_status'],'T_A_from_B_convention':'column vectors p_A = R_A_from_B p_B + t_A_from_B','R_BODY':R_BODY.tolist(),'cameras':{},'pairs':[]};cal={};cp={};cr={};legacyR={}
 for view,ch in [('cam3',1),('cam4',13)]:
  raw=a['cameras'][view]['raw'];tb=raw['T_b_c'];tbc=trans(Rot.from_quat(tb[3:]).as_matrix(),tb[:3]);rv=old[view+'_R_virtual_from_gen'];d=np.array(raw['D']);dv=d.copy();dv[2:4]=[639.5,359.5];braw=trans(R_BODY,np.zeros(3))@tbc;bvirt=braw@trans(rv.T,np.zeros(3));cal[view]=bvirt
  r_wv=vr@bvirt[:3,:3];t_wv=vp+np.einsum('nij,j->ni',vr,bvirt[:3,3]);r_vw=np.swapaxes(r_wv,1,2)
  points=np.einsum('nij,nkj->nki',r_vw,p-t_wv[:,None,:]);rh=r_vw@headR;uv,valid=project(points,dv);cp[view]=points;cr[view]=rh
  rr=z[view+'_R_world_from_virtual'];tt=z[view+'_t_world_from_virtual'];lr=np.swapaxes(rr,1,2)@headR[frames];legacyR[view]=lr
  legacy_p=np.einsum('nji,nj->ni',rr,p[frames,15]-tt)
  archived=np.array(a['cameras'][view]['archive_implied_T_body_from_raw']['R_mean']);archive_t=np.array(a['cameras'][view]['archive_implied_T_body_from_raw']['t_mean_m'])
  aligned_rotdiff=np.rad2deg(Rot.from_matrix(braw[:3,:3].T@archived).magnitude());tdiff=np.linalg.norm(archive_t-braw[:3,3])*1000
  olduv=old[view+'_body_uv'];oldvalid=(olduv[...,0]>=0)&(olduv[...,0]<1280)&(olduv[...,1]>=0)&(olduv[...,1]<720)&valid[frames]
  oldanchors=json.loads((ROOT/'outputs/gen_three_models_20261118/prepared.json').read_text())['items'];oldhead={x['pair']:np.array(x['anchor']['rotation']) for x in oldanchors if x['condition']=='rw500'}
  angles=[Rot.from_matrix(oldhead[f'gen_{int(f):05d}'].T@lr[i]).magnitude()*180/np.pi for i,f in enumerate(frames)] if view=='cam3' else None
  # Recreate first virtual RGB from raw source. Only image processing, no synthetic image editing.
  capture=cv2.VideoCapture(str(OUT/f'image_{ch}_00000.h264'));decoded,image=capture.read();capture.release();assert decoded;yy,xx=np.mgrid[:720,:1280];grid=np.stack([xx,yy],-1);rays,rayvalid=unproject(grid,dv);orig= rays@rv
  mx,mask=project(orig,d);mx[~(mask&rayvalid)]=-1000;recreated=cv2.remap(image,mx.astype(np.float32),None,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
  target=cv2.imread(str(ROOT.parent/f'gen_egosmplx_run/images/{view}/gen_00000.png'));diff=np.abs(recreated.astype(float)-target.astype(float));cv2.imwrite(str(OUT/f'recreated_{view}_00000.png'),recreated)
  roundtrip,_=project(rays,dv)
  report['cameras'][view]={'T_vio_from_raw_camera':braw.tolist(),'T_vio_from_virtual_camera':bvirt.tolist(),'raw_vs_archive_rotation_deg':float(aligned_rotdiff),'raw_vs_archive_optical_center_distance_mm':float(tdiff),'official_vs_archive_joint_position_mm':stats(np.linalg.norm(points[frames]-old[view+'_body_xyz'],axis=-1)*1000),'official_vs_archive_2d_shift_on_old_inframe_px':stats(np.linalg.norm(uv[frames]-olduv,axis=-1)[oldvalid]),'legacy_head_position_roundtrip_mm':stats(np.linalg.norm(legacy_p-old[view+'_body_xyz'][:,15],axis=-1)*1000),'new_vs_old_head_rotation_deg_not_accuracy':stats(angles) if angles else None,'first_image_recreation':{'mean_absolute_uint8_difference':float(diff.mean()),'max_absolute_uint8_difference':float(diff.max()),'exact_pixel_fraction':float((diff==0).all(-1).mean()),'ds_project_unproject_max_px':float(np.linalg.norm(roundtrip-grid,axis=-1)[rayvalid].max())}}
  assets['cameras'][view]={'raw_camera_id':2 if view=='cam3' else 3,'raw_size':[raw['width'],raw['height']],'raw_model':'double_sphere','raw_params_fx_fy_cx_cy_xi_alpha':d.tolist(),'T_calibration_base_from_raw':tbc.tolist(),'R_virtual_from_raw':rv.tolist(),'virtual_size':[1280,720],'virtual_params_fx_fy_cx_cy_xi_alpha':dv.tolist(),'T_vio_from_virtual':bvirt.tolist()}
  # Private overlay for visual inspection of calibration changes.
  im=target.copy()
  for pt in olduv[0]:
   if np.isfinite(pt).all() and (abs(pt)<10000).all():cv2.circle(im,tuple(np.round(pt).astype(int)),5,(0,180,255),2)
  for j,pt in enumerate(uv[0]):
   if valid[0,j] and (abs(pt)<10000).all():cv2.drawMarker(im,tuple(np.round(pt).astype(int)),(255,255,0),cv2.MARKER_CROSS,12,2)
  cv2.imwrite(str(OUT/f'calibration_overlay_{view}_00000.png'),im)
 t_rl=np.linalg.inv(cal['cam4'])@cal['cam3'];archive=json.loads((ROOT/'experiments/gen_stereo_analysis_20261030_20261103/first_stereo_20261031/transform.json').read_text());oldT=trans(np.array(archive['R_cam4_from_cam3']),np.array(archive['t_cam4_from_cam3_m']))
 report['stereo']={'T_right_virtual_from_left_virtual':t_rl.tolist(),'baseline_mm':float(np.linalg.norm(t_rl[:3,3])*1000),'archive_baseline_mm':float(np.linalg.norm(oldT[:3,3])*1000),'relative_rotation_difference_deg':float(Rot.from_matrix(oldT[:3,:3].T@t_rl[:3,:3]).magnitude()*180/np.pi),'translation_vector_difference_mm':float(np.linalg.norm(t_rl[:3,3]-oldT[:3,3])*1000)};assets['T_right_virtual_from_left_virtual']=t_rl.tolist()
 for i,f in enumerate(frames):
  f=int(f);sync={'frame':f,'left_body_dt_us':(btime[f]['log_ns']-times[f]['log_ns'])/1e3,'right_left_dt_us':(rtimes[f]['log_ns']-times[f]['log_ns'])/1e3,'vio_left_dt_us':(vtime[f]['log_ns']-times[f]['log_ns'])/1e3}
  ranges=read(8)[0]['message']['ranges']; in_valid_range=any(int(x.get('start_timestamp_ns',0))<=times[f]['log_ns']<int(x['end_timestamp_ns']) and x.get('is_valid',False) for x in ranges)
  sync['time_range_valid']=in_valid_range;sync['frame_validity_flag']=read(9)[f]['message'].get('is_valid',False)
  report['selected_pairs'].append(sync)
  assets['pairs'].append({'frame':f,'left_log_ns':times[f]['log_ns'],'right_log_ns':rtimes[f]['log_ns'],'head_world_position_m':p[f,15].tolist(),'head_message_quaternion_xyzw':q[f,15].tolist(),'head_world_R_schema_interpretation':headR[f].tolist(),'vio_device_world_position_m':vp[f].tolist(),'vio_device_world_R':vr[f].tolist(),'head_virtual':{v:{'position_m':cp[v][f,15].tolist(),'R_schema_interpretation':cr[v][f].tolist()} for v in cp},'legacy_frame_candidate':{v:{'position_m':old[v+'_body_xyz'][i,15].tolist(),'R_schema_interpretation':legacyR[v][i].tolist()} for v in cp},'ready_as_smplx_anchor':False,'validity':{'time_range_valid':in_valid_range,'frame_flag':sync['frame_validity_flag']}})
 (OUT/'head_camera_20pairs.json').write_text(json.dumps(assets,indent=2)+'\n')
 np.savez(OUT/'head_all_frames.npz',log_time_ns=np.array([x['log_ns'] for x in times],np.int64),world_position=p[:,15],world_quaternion_xyzw=q[:,15],vio_device_world_position=vp,vio_device_world_R=vr,**{v+'_position':cp[v][:,15] for v in cp},**{v+'_R_schema_interpretation':cr[v] for v in cr})
 report['selected_time_range_valid_count']=sum(x['time_range_valid'] for x in report['selected_pairs']);report['selected_time_range_invalid_frames']=[x['frame'] for x in report['selected_pairs'] if not x['time_range_valid']]
 report['asset_manifest']=[{'location':'<WORKSPACE>/outputs/gen_head_camera_alignment_20261118/'+f,'sha256':hashlib.sha256((OUT/f).read_bytes()).hexdigest(),'bytes':(OUT/f).stat().st_size,'access':'local data disk only, not published'} for f in ['head_camera_20pairs.json','head_all_frames.npz']]
 (RUN/'evidence/export.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({k:v for k,v in report.items() if k not in ('selected_pairs','asset_manifest')},indent=2))
if __name__=='__main__':main()
