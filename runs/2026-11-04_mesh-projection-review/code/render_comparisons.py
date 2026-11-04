"""Render v0/stage-two triangles, reference skeleton and automatic detections."""
import argparse,json
from pathlib import Path
import cv2,numpy as np
from rasterizer import project,rays,render
p=argparse.ArgumentParser();p.add_argument('--selection',type=Path,required=True);p.add_argument('--meshes',type=Path,required=True);p.add_argument('--scene-depth',type=Path,required=True);p.add_argument('--images',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--sapiens',type=Path,required=True);p.add_argument('--wilor',type=Path,required=True);p.add_argument('--calibration',type=Path,required=True);p.add_argument('--fits',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);sel=json.loads(a.selection.read_text())['selected'];ref=np.load(a.reference,allow_pickle=False);sap=np.load(a.sapiens,allow_pickle=False);wil=np.load(a.wilor,allow_pickle=False);top=np.load(a.meshes/'topology.npz');faces=top['faces'];head=top['head_weight'];cal=json.loads(a.calibration.read_text());R=np.array(cal['stereo_R32']);t=np.array(cal['stereo_t32']);refi={int(x):i for i,x in enumerate(ref['frames'])};obsi={int(x):i for i,x in enumerate(sap['frames'])};wi={int(x):i for i,x in enumerate(wil['frames'])};ray={c:rays(cal[f'cam{c}_ds'],1600,1300) for c in (2,3)};wristnames=sap['names'].tolist();hand_edges=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)]
def circle(im,uv,color,radius=5,width=-1):
 if np.isfinite(uv).all() and -100<uv[0]<1700 and -100<uv[1]<1400:cv2.circle(im,tuple(np.rint(uv).astype(int)),radius,color,width,cv2.LINE_AA)
def line(im,a,b,color,width=3):
 if np.isfinite(a).all() and np.isfinite(b).all() and max(np.max(np.abs(a)),np.max(np.abs(b)))<10000:cv2.line(im,tuple(np.rint(a).astype(int)),tuple(np.rint(b).astype(int)),color,width,cv2.LINE_AA)
def overlays(im,frame,c):
 i=refi[frame];k=obsi[frame];w=wi[frame];out=im.copy();body=ref[f'cam{c}_body_uv'][i];hand=ref[f'cam{c}_hand_uv'][i]
 for side,si,shoulder,elbow,wrist in [('L',0,16,18,20),('R',1,17,19,21)]:
  for x,y in ((shoulder,elbow),(elbow,wrist)):line(out,body[x],body[y],(30,240,80),4)
  for x,y in hand_edges:line(out,hand[si,x],hand[si,y],(30,240,80),3)
  for uv in hand[si]:circle(out,uv,(30,240,80),3)
  j=wristnames.index(('left' if si==0 else 'right')+'_wrist');uv=sap['xy'][k,c-2,j];conf=sap['score'][k,c-2,j]
  if np.isfinite(conf) and conf>=.3 and np.isfinite(uv).all():
   circle(out,uv,(0,180,255),11,3)
   if 0<=uv[0]<1600 and 0<=uv[1]<1300:cv2.putText(out,'S'+side,tuple(np.rint(uv+[12,-12]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,180,255),2,cv2.LINE_AA)
  pts=wil['xy'][w,c-2,4+20*si:24+20*si];valid=wil['valid'][w,c-2,4+20*si:24+20*si]
  for uv,good in zip(pts,valid):
   if good:circle(out,uv,(255,90,30),5,-1)
 return out

def banner(im,label):
 im=im.copy();cv2.rectangle(im,(0,0),(im.shape[1],32),(18,18,18),-1);cv2.putText(im,label,(8,23),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2,cv2.LINE_AA);return im

def crop_bounds(frame,c,si):
 i=refi[frame];h=ref[f'cam{c}_hand_uv'][i,si];wrist=ref[f'cam{c}_body_uv'][i,20+si];valid=np.isfinite(h).all(1)&(np.abs(h[:,0])<5000)&(np.abs(h[:,1])<5000);pts=h[valid]
 if len(pts):lo=pts.min(0);hi=pts.max(0);center=(lo+hi)/2;size=max(300.,min(650.,max(hi-lo)*2.0))
 else:center=wrist if np.isfinite(wrist).all() else np.array([800.,650.]);size=400.
 center=np.clip(center,[0,0],[1599,1299]);size=int(round(size));x0=int(np.clip(center[0]-size/2,0,1600-size));y0=int(np.clip(center[1]-size/2,0,1300-size));return x0,y0,size
records=[]
for entry in sel:
 frame=entry['frame'];mesh=np.load(a.meshes/f'{frame:05d}.npz');scene=np.load(a.scene_depth/f'{frame:05d}.npz');fullrows=[];stats=[]
 for c in (2,3):
  original=cv2.imread(str(a.images/f'cam{c}'/f'{frame:05d}.jpg'));assert original is not None and original.shape[:2]==(1300,1600);renders=[]
  for mode,key in [('v0','v0_vertices'),('two-stage','stage_vertices')]:
   verts=mesh[key].astype(np.float64);verts=verts if c==2 else (verts-t)@R;color,depth,hp,st=render(verts,faces,head,cal[f'cam{c}_ds'],ray[c]);valid=np.isfinite(depth);sd=scene[f'cam{c}_depth'];blocked=valid&np.isfinite(sd)&(sd+.04<depth);visible=valid&~blocked;out=original.copy();out[visible]=(.42*original[visible]+.58*color[visible]).astype(np.uint8);out=overlays(out,frame,c);renders.append(out);st.update(frame=frame,camera=c,method=mode,scene_depth_available_on_mesh=int((valid&np.isfinite(sd)).sum()),scene_occluded_pixels=int(blocked.sum()),visible_head_after_scene=int(((hp>.5)&visible).sum()));stats.append(st)
  full=np.hstack([banner(cv2.resize(renders[0],(800,650)),f'CAM{c} / V0   |   mesh self-z + stereo scene mask'),banner(cv2.resize(renders[1],(800,650)),f'CAM{c} / V0 + HAND STAGE')]);fullrows.append(full)
  crops=[]
  for si,side in enumerate(('LEFT','RIGHT')):
   x,y,size=crop_bounds(frame,c,si)
   for mode,im in zip(('V0','HAND STAGE'),renders):crops.append(banner(cv2.resize(im[y:y+size,x:x+size],(400,400)),f'{side} / {mode}  |  DFM crop'))
  fullrows.append(np.hstack(crops))
 title=np.zeros((76,1600,3),np.uint8);cv2.putText(title,f'FRAME {frame:05d}  |  DFM is reference, not truth  |  wrist L/R: {entry["wrists"]["left"]:.1f} / {entry["wrists"]["right"]:.1f} cm',(15,25),cv2.FONT_HERSHEY_SIMPLEX,.68,(255,255,255),2,cv2.LINE_AA);cv2.putText(title,'Mesh cyan; head magenta | DFM skeleton green | Sapiens wrist orange | matched WiLoR hand blue',(15,57),cv2.FONT_HERSHEY_SIMPLEX,.59,(230,230,230),2,cv2.LINE_AA);panel=np.vstack([title]+fullrows);name=f'frame_{frame:05d}.jpg';assert cv2.imwrite(str(a.output/name),panel,[cv2.IMWRITE_JPEG_QUALITY,91]);records.append({'frame':frame,'file':name,'cameras_and_modes':stats,'note':'Mesh has exact self-depth testing; scene-depth occlusion is conservative, partial and can be wrong.'});print('DONE',frame,flush=True)
(a.output/'render_stats.json').write_text(json.dumps({'frames':records},indent=2,allow_nan=False)+'\n')
