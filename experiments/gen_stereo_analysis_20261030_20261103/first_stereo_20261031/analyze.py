import numpy as np, json, csv, os, tempfile
from scipy.optimize import least_squares
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
base='/vepfs-mlp2/mlp-public/huyaoqing/EgoFishPose/experiments/gen_egosmplx_two_frames_20261030'
root='/vepfs-mlp2/mlp-public/huyaoqing/gen_egosmplx_run'
out=tempfile.mkdtemp(prefix='stereo_diagnostic_20261031_',dir=base)
p=np.load(root+'/dfm_reference_virtual.npz'); a=p['cam3_body_xyz'][0]; b=p['cam4_body_xyz'][0]; am=a.mean(0);bm=b.mean(0);u,_,v=np.linalg.svd((a-am).T@(b-bm));R=v.T@np.diag([1,1,np.linalg.det(v.T@u.T)])@u.T;t=bm-R@am
print('OUT',out,'baseline',np.linalg.norm(t),'R',R,'t',t)
cal=[json.load(open(root+'/calibration/'+x+'.egosmplx.json'))['sourceParams'] for x in ['gen_cam2_virtual_cam3','gen_cam3_virtual_cam4']]
ks=[{x['name']:x for x in json.load(open(base+f'/experiment_C/output/observations/gen_00000_cam{i}/sapiens/keypoints.json'))['keypoints']} for i in (3,4)]
names=['left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist','left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle']
def ray(x,c):
 fx,fy,cx,cy,xi,al=c; mx=(x[0]-cx)/fx; my=(x[1]-cy)/fy; r2=mx*mx+my*my; mz=(1-al*al*r2)/(al*np.sqrt(1-(2*al-1)*r2)+1-al); k=(mz*xi+np.sqrt(mz*mz+(1-xi*xi)*r2))/(mz*mz+r2); v=np.array([k*mx,k*my,k*mz-xi]);return v/np.linalg.norm(v)
def proj(x,c):
 fx,fy,cx,cy,xi,al=c; d1=np.linalg.norm(x);z=xi*d1+x[2];d2=np.sqrt(x[0]**2+x[1]**2+z*z); den=al*d2+(1-al)*z;return np.array([fx*x[0]/den+cx,fy*x[1]/den+cy])
def epi(z):return np.dot(ray(z[2:],cal[1]),np.cross(t,R@ray(z[:2],cal[0])))
def grad(f,z,h=.01):return np.array([(f(z+np.eye(len(z))[i]*h)-f(z-np.eye(len(z))[i]*h))/(2*h) for i in range(len(z))])
def solve(z):
 r1=ray(z[:2],cal[0]); r2=R.T@ray(z[2:],cal[1]); C=-R.T@t; q=np.linalg.lstsq(np.stack((r1,-r2),axis=1),C,rcond=None)[0];return r1*q[0],q

def err_for_point(x):
 z=np.r_[proj(x,cal[0]),proj(R@x+t,cal[1])]; g=grad(lambda z:np.linalg.norm(solve(z)[0]),z);return 3*np.linalg.norm(g)
rows=[]
for name in names:
 p1=ks[0][name];p2=ks[1][name];z=np.array([p1['x'],p1['y'],p2['x'],p2['y']]);g=grad(epi,z);res=abs(epi(z))/np.linalg.norm(g);x,q=solve(z); pos=all(q>0); dist=np.linalg.norm(x) if pos else np.nan; angle=np.degrees(np.arccos(np.clip(np.dot(ray(z[:2],cal[0]),R.T@ray(z[2:],cal[1])),-1,1))); sig=3*np.linalg.norm(grad(lambda zz: np.linalg.norm(solve(zz)[0]),z)) if pos else np.nan
 row=dict(joint=name,cam3_x=round(z[0],1),cam3_y=round(z[1],1),cam4_x=round(z[2],1),cam4_y=round(z[3],1),score3=round(p1['score'],3),score4=round(p2['score'],3),sampson_px=round(res,2),flag=res>9,forward=pos,ray_angle_deg=round(angle,2),range_m=round(dist,3),range_sigma_m_3px=round(sig,3));rows.append(row); print(row)
with open(out+'/joints.csv','w') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
fig,axs=plt.subplots(1,2,figsize=(20,7)); labels={'left_shoulder':'L shoulder','right_shoulder':'R shoulder','left_elbow':'L elbow','right_elbow':'R elbow','left_wrist':'L wrist','right_wrist':'R wrist','left_hip':'L hip','right_hip':'R hip','left_knee':'L knee','right_knee':'R knee','left_ankle':'L ankle','right_ankle':'R ankle'}
for i,ax in enumerate(axs):
 ax.imshow(Image.open(base+f'/experiment_C/output/observations/images/gen_00000_cam{i+3}.png'));ax.set_title(f'cam{i+3}: red = Sampson > 9 px; cyan = not flagged',fontsize=13);ax.axis('off')
 for row in rows:
  xx=row[f'cam{i+3}_x'];yy=row[f'cam{i+3}_y'];col='#ff4238' if row['flag'] else '#00f5ff';ax.scatter([xx],[yy],s=90,facecolor=col,edgecolor='black',linewidth=1.3,zorder=4); ax.annotate(f"{labels[row['joint']]} {row['sampson_px']:.1f}",(xx,yy),xytext=(5,-8),textcoords='offset points',color='white',fontsize=8,bbox=dict(facecolor='black',alpha=.72,pad=1),zorder=5)
fig.tight_layout();fig.savefig(out+'/keypoints.png',dpi=150);plt.close(fig)
print('sensitivity')
for name in ['left_wrist','right_wrist','left_hip','right_knee']:
 z=np.array([ks[0][name]['x'],ks[0][name]['y']]);ray1=ray(z,cal[0]); print(name,[(d,round(err_for_point(ray1*d),3)) for d in [.25,.5,.75,1,1.5,2]])
open(out+'/transform.json','w').write(json.dumps({'R_cam4_from_cam3':R.tolist(),'t_cam4_from_cam3_m':t.tolist(),'baseline_m':np.linalg.norm(t),'method':'rigid alignment of paired DFM body coordinates, used to recover relative camera geometry only'},indent=2))
