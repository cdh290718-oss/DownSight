"""Two pairwise comparisons against common GT with identical axes within each camera."""
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import RUN,OUT,OLD,read
from egosmplx_pipeline.io import load
inputs=read(OUT/'inputs.json');metrics=read(RUN/'evidence/metrics.json');records=[]
edges=[(0,1),(0,2),(1,3),(2,4),(3,5),(4,6),(5,14),(5,15),(5,16),(6,17),(6,18),(6,19),(0,7),(7,8),(7,9),(8,10),(9,11),(10,12),(11,13)]
for side in ['left','right']:
 cam=inputs['views'][side]['camera'];T=np.array(cam['left_to_view']);source=OLD/'dataset/samples/000001'/(side+'_egosmplx65.npz')
 with np.load(source) as z:gt=z['joints_3d_camera_m'][:25];inside=z['in_frame_no_occlusion_test'][:25]
 points={'GT':gt}
 for c in ['control','gt_hips','gt_body']:
  p=load(OUT/(c+'_left')/'final.npz');j=p['smplx_joint_cam'][:25];points[c]=j@T[:3,:3].T+T[:3,3]
 angular={k:np.rad2deg(np.column_stack([np.arctan2(p[:,0],p[:,2]),np.arctan2(p[:,1],np.hypot(p[:,0],p[:,2]))])) for k,p in points.items()}
 uv_all=np.concatenate([angular[k][inside&(p[:,2]>0)] for k,p in points.items()]);xyz_all=np.concatenate(list(points.values()))
 xlim=[float(uv_all[:,0].min()-3),float(uv_all[:,0].max()+3)];ylim=[float(uv_all[:,1].max()+3),float(uv_all[:,1].min()-3)]
 zlim=[float(min(0,xyz_all[:,2].min())-.05),float(xyz_all[:,2].max()+.05)];metricY=[float(xyz_all[:,1].max()+.05),float(min(0,xyz_all[:,1].min())-.05)]
 for before in ['control','gt_hips']:
  fig,axes=plt.subplots(1,2,figsize=(12.8,7.2),dpi=100)
  for key,color,label in [('GT','#269447','Ground truth'),(before,'#2865cc','Automatic' if before=='control' else 'Only GT hips'),('gt_body','#ed7d23','All 12 body targets GT')]:
   p=points[key];uv=angular[key];valid=inside&(p[:,2]>0);axes[0].scatter(uv[valid,0],uv[valid,1],s=18,c=color,label=label,zorder=3);axes[1].scatter(p[:,2],p[:,1],s=18,c=color,label=label,zorder=3)
   for a,b in edges:
    if valid[a] and valid[b]:axes[0].plot(uv[[a,b],0],uv[[a,b],1],color=color,lw=1.6,alpha=.85)
    axes[1].plot(p[[a,b],2],p[[a,b],1],color=color,lw=1.6,alpha=.85)
  axes[0].set(xlabel='Ray azimuth (deg)',ylabel='Ray downward elevation (deg)',title='Camera ray angles; GT in-frame',xlim=xlim,ylim=ylim);axes[0].set_aspect('equal',adjustable='box')
  axes[1].scatter([0],[0],c='black',marker='^',s=55,label='Camera');axes[1].set(xlabel='Optical depth Z (m)',ylabel='Camera Y (m)',title='Side view (equal metric aspect)',xlim=zlim,ylim=metricY);axes[1].set_aspect('equal',adjustable='box')
  for ax in axes:ax.grid(alpha=.25);ax.legend(fontsize=8)
  fig.suptitle(('cam3' if side=='left' else 'cam4')+' | '+before+' vs all-body GT targets\nHead anchor, initialization, uncertainty and constraints unchanged',fontsize=13);fig.tight_layout(rect=[0,0,1,.91]);path=RUN/'assets'/('skeleton_'+side+'_'+before+'.png');fig.savefig(path,dpi=100);plt.close(fig)
  records.append(dict(image=str(path.relative_to(RUN)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),side=side,before=before,ray_axis_limits=[xlim,ylim],side_axis_limits=[zlim,metricY],gt_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
(RUN/'evidence/skeleton_plots.json').write_text(json.dumps(dict(exit_status=0,scope='Pairwise plots use identical axis limits for the same camera. Ray angles are not distorted pixel coordinates. GT in-frame is not occlusion visibility; side plot shows body25, including unobserved feet and facial surface landmarks.',plots=records),indent=2)+'\n');print('plots',len(records))
