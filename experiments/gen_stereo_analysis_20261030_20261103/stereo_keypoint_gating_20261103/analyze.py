import argparse, json, csv, hashlib
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from PIL import Image, ImageDraw

p=argparse.ArgumentParser(); p.add_argument('--data',type=Path,required=True);p.add_argument('--cal',type=Path,required=True);p.add_argument('--out',type=Path,required=True); a=p.parse_args()
a.out.mkdir(exist_ok=True,parents=True); (a.out/'overlays').mkdir(exist_ok=True)
d=np.load(a.data/'det_sapiens2.npz'); ref=np.load(a.data/'dfm_reference.npz'); cal=json.loads(a.cal.read_text()); ds=[np.array(cal['cam2_ds']),np.array(cal['cam3_ds'])]; R=np.array(cal['stereo_R32']); t=np.array(cal['stereo_t32']); names=d['names'].tolist(); select=names[5:63]; frames=list(range(0,5131,270)); lookup={(int(c),int(f)):i for i,(c,f) in enumerate(zip(d['cams'],d['frames']))}; ri={int(f):i for i,f in enumerate(ref['frames'])}

def unproj(uv,d):
 fx,fy,cx,cy,xi,al=d; x=(uv[0]-cx)/fx;y=(uv[1]-cy)/fy;r=x*x+y*y; z=(1-al*al*r)/(al*np.sqrt(max(0,1-(2*al-1)*r))+1-al); k=(z*xi+np.sqrt(z*z+(1-xi*xi)*r))/(z*z+r);v=np.array([k*x,k*y,k*z-xi]);return v/np.linalg.norm(v)
def proj(x,d):
 fx,fy,cx,cy,xi,al=d; n=np.linalg.norm(x); z=xi*n+x[2]; den=al*np.sqrt(x[0]**2+x[1]**2+z*z)+(1-al)*z;return np.array([fx*x[0]/den+cx,fy*x[1]/den+cy])
def project(x):return np.array([proj(x,ds[0]),proj(R.T@(x-t),ds[1])])
def tri(uv):
 v=unproj(uv[0],ds[0]);w=R@unproj(uv[1],ds[1]); z=np.linalg.lstsq(np.stack([v,-w],axis=1),t,rcond=None)[0];return (v*z[0]+t+w*z[1])/2,z
rows=[]; points={}; images={}; base_names=[s+'_'+n for s in ('left','right') for n in ('elbow','wrist','thumb4','forefinger4','middle_finger4','ring_finger4','pinky_finger4')]
for f in frames:
 xy=np.array([d['xy'][lookup[c,f]] for c in (2,3)]); sc=np.array([d['score'][lookup[c,f]] for c in (2,3)]); temp={}
 for name in select:
  j=names.index(name); uv=xy[:,j].astype(float); scores=sc[:,j]; border=np.minimum.reduce([uv[:,0],1600-uv[:,0],uv[:,1],1300-uv[:,1]]); old=(np.array([name in base_names]*2)&(scores>=.3)&(border>40)); reason=[]; corrected=None; xyz=None; cov=None; epi=None; depth=None; std=None
  finite=np.isfinite(uv).all() and np.isfinite(scores).all(); admiss=finite and np.all(scores>=.15) and np.all(border>10)
  if admiss:
   raw,z=tri(uv); fit=least_squares(lambda x:(project(x)-uv).reshape(-1),raw,max_nfev=50); xyz=fit.x; corrected=project(xyz); epi=float(np.linalg.norm((corrected-uv).reshape(-1))); depth=float(np.linalg.norm(xyz));
   try:
    cov=64*np.linalg.inv(fit.jac.T@fit.jac);std=float(np.sqrt(np.linalg.eigvalsh(cov).max()))
   except np.linalg.LinAlgError: std=float('inf')
   if np.min(z)<=0 or depth<.12 or depth>2.5:reason.append('depth_out_of_range')
   if epi>8:reason.append('reprojection_correction_over_8px')
   if std>.30 or not np.isfinite(std):reason.append('depth_uncertainty_over_30cm')
  else:reason.append('low_score_or_border_or_nonfinite')
  temp[name]=dict(uv=uv,scores=scores,border=border,old=old,reason=reason,corrected=corrected,xyz=xyz,cov=cov,epi=epi,depth=depth,std=std)
 for part in ['shoulder','elbow','wrist','hip','knee','ankle']:
  l=temp['left_'+part];r=temp['right_'+part]
  if l['xyz'] is not None and r['xyz'] is not None and np.linalg.norm(l['xyz']-r['xyz'])<.03 and np.max(np.linalg.norm(l['uv']-r['uv'],axis=1))<20:
   l['reason'].append('left_right_duplicate_ambiguous');r['reason'].append('left_right_duplicate_ambiguous')
 for name,q in temp.items():
  ambiguous='left_right_duplicate_ambiguous' in q['reason']; stereo=not q['reason']; v2=(q['scores']>=.3)&(q['border']>40)&np.isfinite(q['uv']).all(axis=1)&(not ambiguous)&('reprojection_correction_over_8px' not in q['reason'])&('depth_out_of_range' not in q['reason']); status='stereo_3d' if stereo else ('2d_only' if np.any(v2) else 'discard'); use=np.ones(2,dtype=bool) if stereo else v2
  row=dict(frame=f,joint=name,status=status,reason='stereo_consistent_candidate' if stereo else ';'.join(q['reason']),old_cam2=int(q['old'][0]),old_cam3=int(q['old'][1]),use_cam2=int(use[0]),use_cam3=int(use[1]),score_cam2=float(q['scores'][0]),score_cam3=float(q['scores'][1]),border_cam2_px=float(q['border'][0]),border_cam3_px=float(q['border'][1]),raw_uv=q['uv'].tolist(),corrected_uv=q['corrected'].tolist() if stereo else None,xyz_cam2_m=q['xyz'].tolist() if stereo else None,cov_cam2_m2=q['cov'].tolist() if stereo else None,correction_norm_px=q['epi'],range_m=q['depth'],largest_sigma_m=q['std'])
  rows.append(row);points[f,name]=row

changes=[]
for row in rows:
 for ci,cam in enumerate((2,3)):
  old=row['old_cam'+str(cam)];new=row['use_cam'+str(cam)]
  if old!=new: changes.append((row,cam,'added' if new else 'removed'))
# One individually labelled crop per changed view, and a single indexed contact sheet.
thumbs=[]
for k,(row,cam,kind) in enumerate(changes):
 key=(row['frame'],cam)
 if key not in images:images[key]=Image.open(a.data/'frames'/('cam%d'%cam)/('%05d.jpg'%row['frame'])).convert('RGB')
 im=images[key];u,v=row['raw_uv'][cam-2]; x=int(np.clip(u,0,1599));y=int(np.clip(v,0,1299)); crop=im.crop((x-95,y-80,x+95,y+80)); draw=ImageDraw.Draw(crop);draw.ellipse((90,75,100,85),outline='yellow',width=3)
 if row['corrected_uv'] is not None:
  xx,yy=row['corrected_uv'][cam-2];dx=xx-u;dy=yy-v;draw.line((95,80,95+dx,80+dy),fill='cyan',width=2);draw.ellipse((92+dx,77+dy,98+dx,83+dy),outline='cyan',width=2)
 draw.rectangle((0,0,190,20),fill='black');draw.text((2,3),f'{k:04d} {row["frame"]:05d} c{cam} {kind}',fill='white');draw.rectangle((0,140,190,160),fill='black');draw.text((2,143),row['joint'],fill='white')
 crop.save(a.out/'overlays'/f'{k:04d}.jpg',quality=78);thumbs.append(crop.resize((114,96)))
cols=12;sheet=Image.new('RGB',(cols*114,((len(thumbs)+cols-1)//cols)*96),'white')
for k,im in enumerate(thumbs):sheet.paste(im,((k%cols)*114,(k//cols)*96))
sheet.save(a.out/'contact_sheet.jpg',quality=82)
with (a.out/'decisions.json').open('w') as h:json.dump(rows,h,indent=2,allow_nan=False)
with (a.out/'changes.csv').open('w',newline='') as h:
 w=csv.writer(h,lineterminator="\n");w.writerow(['id','frame','camera','joint','change','status','reason','overlay']);
 for k,(r,c,kind) in enumerate(changes):w.writerow([k,r['frame'],c,r['joint'],kind,r['status'],r['reason'],f'overlays/{k:04d}.jpg'])
with (a.out/'counts.csv').open('w',newline='') as h:
 w=csv.writer(h,lineterminator="\n");w.writerow(['frame','joint','old_views','new_views','stereo_3d','2d_only','discard']);
 for r in rows:w.writerow([r['frame'],r['joint'],r['old_cam2']+r['old_cam3'],r['use_cam2']+r['use_cam3'],int(r['status']=='stereo_3d'),int(r['status']=='2d_only'),int(r['status']=='discard')])
wrist=[]
for f in frames:
 for side,ix in [('left',20),('right',21)]:
  r=points[f,side+'_wrist']; i=ri[f]
  if ref['cam2_body_valid'][i,ix] and r['status']=='stereo_3d':
   raw,_=tri(np.array(r['raw_uv']));gt=ref['cam2_body_xyz'][i,ix]; wrist.append(dict(frame=f,joint=side+'_wrist',before_cm=float(100*np.linalg.norm(raw-gt)),after_cm=float(100*np.linalg.norm(np.array(r['xyz_cam2_m'])-gt))))
summary=dict(frames=frames,scope=select,counts={s:sum(r['status']==s for r in rows) for s in ['stereo_3d','2d_only','discard']},old_views=sum(r['old_cam2']+r['old_cam3'] for r in rows),new_views=sum(r['use_cam2']+r['use_cam3'] for r in rows),added=sum(c[2]=='added' for c in changes),removed=sum(c[2]=='removed' for c in changes),wrist=wrist,baseline_m=float(np.linalg.norm(t)),sha256={x:hashlib.sha256((a.data/x).read_bytes()).hexdigest() for x in ('det_sapiens2.npz','dfm_reference.npz')})
with (a.out/'summary.json').open('w') as h:json.dump(summary,h,indent=2,allow_nan=False)
print(json.dumps({k:v for k,v in summary.items() if k not in ('scope','wrist')},indent=2));print('wrist_n',len(wrist))
