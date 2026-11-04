"""Image-backed wrist diagnostics; generated only with confirmed dataset publication authorization."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
p=argparse.ArgumentParser();p.add_argument('--metrics',type=Path,required=True);p.add_argument('--observations',type=Path,required=True);p.add_argument('--images',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,choices=[500,1500],default=500);a=p.parse_args();m=json.loads(a.metrics.read_text());obs=np.load(a.observations,allow_pickle=False);names=obs['names'].tolist();lookup={int(f):i for i,f in enumerate(obs['frames'])};a.output.mkdir(exist_ok=False,parents=True);size=400;top=76;bottom=30

def mark(draw,xy,color,kind,label):
 x,y=xy
 if kind=='circle':draw.ellipse((x-8,y-8,x+8,y+8),outline=color,width=3)
 elif kind=='square':draw.rectangle((x-8,y-8,x+8,y+8),outline=color,width=3)
 elif kind=='cross':draw.line((x-6,y-6,x+6,y+6),fill=color,width=3);draw.line((x-6,y+6,x+6,y-6),fill=color,width=3)
 elif kind=='diamond':draw.line([(x,y-10),(x+10,y),(x,y+10),(x-10,y),(x,y-10)],fill=color,width=3)
 draw.text((x+9,y-11),label,fill=color,stroke_width=1,stroke_fill='black')
for row in m['per_frame']:
 frame=row['frame'];image=Image.new('RGB',(size*2,(size+bottom)*2+top),'black');d=ImageDraw.Draw(image);d.text((8,5),f'Frame {frame:05d} | DFM green diamond; wrist detector yellow cross',fill='white');d.text((8,25),'red B=baseline; cyan G=original gate; white C=cross-hand; magenta W=no tips',fill='white');d.text((8,45),f'orange crosses: valid detector fingertip observations; comparison is {a.steps} steps',fill='white')
 for si,side in enumerate(('left','right')):
  r=row['wrists'][side]
  for ci,cam in enumerate((2,3)):
   im=Image.open(a.images/f'cam{cam}'/f'{frame:05d}.jpg').convert('RGB');cx,cy=r['raw_uv'][ci];x0=int(round(cx))-size//2;y0=int(round(cy))-size//2;crop=im.crop((x0,y0,x0+size,y0+size));xoff=ci*size;yoff=top+si*(size+bottom);image.paste(crop,(xoff,yoff));d=ImageDraw.Draw(image)
   def local(pt):return (pt[0]-x0+xoff,pt[1]-y0+yoff)
   def draw_inside(pt,color,shape,label):
    if x0<=pt[0]<x0+size and y0<=pt[1]<y0+size:mark(d,local(pt),color,shape,label)
   if r['reference_projected_uv'] is not None:draw_inside(r['reference_projected_uv'][ci],'lime','diamond','DFM')
   draw_inside(r['raw_uv'][ci],'yellow','cross','obs')
   for mode,c,shape,label in [('baseline','red','circle','B'),('published_gate','cyan','square','G'),('published_gate_cross_hand','white','diamond','C'),('wrist_only','magenta','circle','W')]:draw_inside(r['fits'][f'{mode}_{a.steps}']['projected_uv'][ci],c,shape,label)
   idx=lookup[frame]
   for prefix,label in [('thumb4','T'),('forefinger4','I'),('middle_finger4','M'),('ring_finger4','R'),('pinky_finger4','P')]:
    j=names.index(side+'_'+prefix)
    if obs['valid'][idx,ci,j]:draw_inside(obs['xy'][idx,ci,j],'orange','cross',label)
   d.rectangle((xoff,yoff+size,xoff+size,yoff+size+bottom),fill='black');d.text((xoff+6,yoff+size+5),f'{side} wrist / cam{cam} | B {r["fits"][f"baseline_{a.steps}"]["vs_dfm_cm"]:.1f} G {r["fits"][f"published_gate_{a.steps}"]["vs_dfm_cm"]:.1f} C {r["fits"][f"published_gate_cross_hand_{a.steps}"]["vs_dfm_cm"]:.1f} W {r["fits"][f"wrist_only_{a.steps}"]["vs_dfm_cm"]:.1f} cm',fill='white')
 image.save(a.output/f'frame_{frame:05d}.jpg',quality=90)
