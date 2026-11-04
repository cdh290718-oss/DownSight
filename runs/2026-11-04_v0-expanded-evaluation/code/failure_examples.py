"""Summarize DFM-defined visibility groups and export a few annotated examples."""
import argparse,json
from pathlib import Path
import cv2,numpy as np
p=argparse.ArgumentParser();p.add_argument('--metrics',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--images',type=Path,required=True);p.add_argument('--fits',type=Path,required=True);p.add_argument('--calibration',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
d=json.loads(a.metrics.read_text());r=np.load(a.reference);cal=json.loads(a.calibration.read_text());R=np.array(cal['stereo_R32']);t=np.array(cal['stereo_t32']);lookup={int(f):i for i,f in enumerate(r['frames'])};summary={}
for mode,rows in d['per_frame'].items():
 summary[mode]={};pred={}
 for path in sorted((a.fits/mode).glob('batch_*.npz')):
  with np.load(path) as data:
   for j,f in enumerate(data['frame_ids']):pred[int(f)]=data['joints_cam2'][j]
 for group in ('either_view_wrist_outside','hands_within_10cm','hands_21_points_within_5cm_exploratory'):
  items=[(row['frame'],side,v) for row in rows for side,v in row['sides'].items() if v['dfm_valid'] and v[group]];fail=[x for x in items if x[2]['wrist_error_cm'] is None or x[2]['wrist_error_cm']>5];summary[mode][group]={'count_frames':len({x[0] for x in items}),'count_wrist_frames':len(items),'gt5_or_missing':len(fail),'failure_fraction':len(fail)/len(items) if items else None,'examples':[]}
  ranked=sorted(fail or items,key=lambda x:-(x[2]['wrist_error_cm'] if x[2]['wrist_error_cm'] is not None else 1e9))[:2]
  for frame,side,v in ranked:
   idx=lookup[frame];joint=20 if side=='left' else 21;panels=[]
   for cam in (2,3):
    im=cv2.imread(str(a.images/f'cam{cam}'/f'{frame:05d}.jpg'));uv=r[f'cam{cam}_body_uv'][idx,joint]
    if np.isfinite(uv).all():
     xy=tuple(int(x) for x in uv);h,w=im.shape[:2]
     if 0<=xy[0]<w and 0<=xy[1]<h:cv2.drawMarker(im,xy,(0,255,0),cv2.MARKER_CROSS,40,4)
     else:
      edge=(int(np.clip(xy[0],15,w-16)),int(np.clip(xy[1],15,h-16)));cv2.circle(im,edge,12,(0,255,0),4);cv2.putText(im,'DFM outside', (max(10,edge[0]-240),max(30,edge[1]-20)),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,255,0),2)
    if frame in pred:
     xyz=pred[frame][12 if side=='left' else 13];xyz=xyz if cam==2 else (xyz-t)@R;fx,fy,cx,cy,xi,alpha=cal[f'cam{cam}_ds'];xx,yy,zz=xyz;d1=np.linalg.norm(xyz);z1=xi*d1+zz;den=alpha*np.sqrt(xx*xx+yy*yy+z1*z1)+(1-alpha)*z1
     if den>0:
      point=(int(fx*xx/den+cx),int(fy*yy/den+cy));cv2.drawMarker(im,point,(0,0,255),cv2.MARKER_CROSS,40,4)
    cv2.putText(im,f'{mode} {side} frame {frame:05d} cam{cam}',(20,45),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2);cv2.putText(im,'Green: DFM wrist; Red: fitted wrist',(20,90),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2);panels.append(cv2.resize(im,(800,650)))
   name=f'{mode}_{group}_{side}_{frame:05d}.jpg';cv2.imwrite(str(a.output/name),np.hstack(panels),[cv2.IMWRITE_JPEG_QUALITY,90]);summary[mode][group]['examples'].append({'frame':frame,'side':side,'error_cm':v['wrist_error_cm'],'image':name})
(a.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
