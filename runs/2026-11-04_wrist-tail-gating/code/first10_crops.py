"""Earlier-frame overlays (DFM-head calibration frames, not independent dataset)."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
p=argparse.ArgumentParser();p.add_argument('--metrics',type=Path,required=True);p.add_argument('--observations',type=Path,required=True);p.add_argument('--images',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();m=json.loads(a.metrics.read_text());obs=np.load(a.observations,allow_pickle=False);names=obs['names'].tolist();lookup={int(f):i for i,f in enumerate(obs['frames'])};a.output.mkdir(exist_ok=False,parents=True);size=400;top=76;bottom=30
for row in m['per_frame']:
 frame=row['frame'];image=Image.new('RGB',(800,top+2*(size+bottom)),'black');draw=ImageDraw.Draw(image);draw.text((8,5),f'Frame {frame:05d} | green=DFM, yellow=detector, red=baseline, white=cross-hand',fill='white');draw.text((8,25),'orange crosses: accepted baseline fingertip observations; comparison = 1500 steps',fill='white');draw.text((8,45),'Earlier frames contributed to simulated head calibration; not an independent dataset',fill='white')
 for si,side in enumerate(('left','right')):
  r=row['wrists'][side]
  for ci,cam in enumerate((2,3)):
   raw=Image.open(a.images/f'cam{cam}'/f'{frame:05d}.jpg').convert('RGB');cx,cy=r['raw_uv'][ci];x0=int(round(cx))-size//2;y0=int(round(cy))-size//2;xoff=ci*size;yoff=top+si*(size+bottom);image.paste(raw.crop((x0,y0,x0+size,y0+size)),(xoff,yoff));draw=ImageDraw.Draw(image)
   def mark(pt,color,label,shape):
    if not(x0<=pt[0]<x0+size and y0<=pt[1]<y0+size):return
    x,y=pt[0]-x0+xoff,pt[1]-y0+yoff
    if shape=='cross':draw.line((x-6,y-6,x+6,y+6),fill=color,width=3);draw.line((x-6,y+6,x+6,y-6),fill=color,width=3)
    else:draw.ellipse((x-8,y-8,x+8,y+8),outline=color,width=3)
    draw.text((x+9,y-9),label,fill=color,stroke_width=1,stroke_fill='black')
   if r['reference_projected_uv'] is not None:mark(r['reference_projected_uv'][ci],'lime','DFM','circle')
   mark(r['raw_uv'][ci],'yellow','obs','cross')
   for key,color,label in [('baseline_1500','red','B'),('published_gate_cross_hand_1500','white','C')]:mark(r['fits'][key]['projected_uv'][ci],color,label,'circle')
   idx=lookup[frame]
   for finger,label in [('thumb4','T'),('forefinger4','I'),('middle_finger4','M'),('ring_finger4','R'),('pinky_finger4','P')]:
    j=names.index(side+'_'+finger)
    if obs['valid'][idx,ci,j]:mark(obs['xy'][idx,ci,j],'orange',label,'cross')
   draw.rectangle((xoff,yoff+size,xoff+size,yoff+size+bottom),fill='black');draw.text((xoff+6,yoff+size+5),f'{side} / cam{cam}',fill='white')
 image.save(a.output/f'frame_{frame:05d}.jpg',quality=90)
