"""Matched-scale skeleton comparisons: GT, replay control, two-hip target intervention."""
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import ROOT,RUN,OUT,OLD,read
from egosmplx_pipeline.io import load
inputs=read(OUT/'inputs.json');metrics=read(RUN/'evidence/metrics.json')
edges=[(0,1),(0,2),(1,3),(2,4),(3,5),(4,6),(5,14),(5,15),(5,16),(6,17),(6,18),(6,19),(0,7),(7,8),(7,9),(8,10),(9,11),(10,12),(11,13)]
series=[('Ground truth','#269447'),('Control','#2865cc'),('GT hip targets','#ed7d23')];records=[]
for side in ['left','right']:
 camera=inputs['views'][side]['camera'];T=np.array(camera['left_to_view']);source=OLD/'dataset/samples/000001'/(side+'_egosmplx65.npz')
 with np.load(source) as data:truth=data['joints_3d_camera_m'][:25].copy();inside=data['in_frame_no_occlusion_test'][:25].copy()
 points=[truth]
 for label in ['control_left','gt_hips_left']:
  p=load(OUT/label/'final.npz');j=p['smplx_joint_cam'][:25];points.append(j@T[:3,:3].T+T[:3,3])
 angular=[]
 for p in points:angular.append(np.rad2deg(np.column_stack([np.arctan2(p[:,0],p[:,2]),np.arctan2(p[:,1],np.hypot(p[:,0],p[:,2]))])))
 fig,axes=plt.subplots(1,2,figsize=(12.8,7.2),dpi=100)
 for j,((name,color),p,uv) in enumerate(zip(series,points,angular)):
  valid=inside&(p[:,2]>0)
  axes[0].scatter(uv[valid,0],uv[valid,1],s=18,c=color,label=name,zorder=3)
  axes[1].scatter(p[:,2],p[:,1],s=18,c=color,label=name,zorder=3)
  for a,b in edges:
   if valid[a] and valid[b]:axes[0].plot(uv[[a,b],0],uv[[a,b],1],color=color,lw=1.6,alpha=.85)
   axes[1].plot(p[[a,b],2],p[[a,b],1],color=color,lw=1.6,alpha=.85)
 axes[0].set(xlabel='Ray azimuth (degrees)',ylabel='Ray downward elevation (degrees)',title='Camera ray angles; GT in-frame joints');axes[0].invert_yaxis();axes[0].set_aspect('equal',adjustable='datalim')
 axes[1].scatter([0],[0],c='black',marker='^',s=55,label='Camera');axes[1].set(xlabel='Optical depth Z (m)',ylabel='Camera Y (m)',title='Side view; all body25 landmarks');axes[1].invert_yaxis();axes[1].set_aspect('equal',adjustable='datalim')
 for ax in axes:ax.grid(alpha=.25);ax.legend(fontsize=8)
 stage=next(v for v in metrics['views'] if v['side']==side)['stages'];text='Pelvis Y error: control %.2f mm -> GT hips %.2f mm'%(stage['control_left']['pelvis_error_xyz_mm'][1],stage['gt_hips_left']['pelvis_error_xyz_mm'][1]);fig.suptitle(('cam3 (left)' if side=='left' else 'cam4 (right)')+' | Only two hip target coordinates changed\n'+text,fontsize=13)
 fig.tight_layout(rect=[0,0,1,.91]);path=RUN/'assets'/('skeleton_'+side+'.png');fig.savefig(path,dpi=100);limits=[dict(x=list(ax.get_xlim()),y=list(ax.get_ylim())) for ax in axes];plt.close(fig)
 records.append(dict(side=side,image=str(path.relative_to(RUN)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),gt_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),series=[a for a,b in series],axis_limits=limits,edges=edges))
q=dict(exit_status=0,plots=records,scope='All three conditions use identical axes within each camera; side panel uses equal metric axis aspect. Left panel is undistorted ray angles, not a distorted pixel projection. GT in-frame flags are not occlusion visibility; all25 points shown in side view. Native Head anchor is not itself one of body25, which includes facial surface landmarks.',source='Cached joints verified against reconstruction by fitter and replay checks; GT only read for evaluation.')
(RUN/'evidence/skeleton_plot.json').write_text(json.dumps(q,indent=2)+'\n');print('skeleton plots',len(records))
