"""Create local image-backed overlays and optional shareable geometry-only charts."""
import argparse,json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
p=argparse.ArgumentParser();p.add_argument('--metrics',type=Path,required=True);p.add_argument('--images',type=Path,required=True);p.add_argument('--local-output',type=Path,required=True);p.add_argument('--geometry-output',type=Path,required=True);a=p.parse_args();m=json.loads(a.metrics.read_text());a.local_output.mkdir(exist_ok=False,parents=True);a.geometry_output.mkdir(exist_ok=False,parents=True);scale=.4;W,H=640,520;header=70;footer=42;col={'reference':'lime','observation':'yellow','baseline':'red','published_gate':'cyan','cross_hand':'white','wrist_only':'magenta'}
def mark(draw,p,color,shape):
 x,y=p[0]*scale,p[1]*scale+header
 if shape=='L':draw.ellipse((x-7,y-7,x+7,y+7),outline=color,width=3)
 else:draw.rectangle((x-7,y-7,x+7,y+7),outline=color,width=3)
 draw.text((x+8,y-6),shape,fill=color)
def make(row,with_image):
 full=Image.new('RGB',(W*2,H+header+footer),'white');canvas=ImageDraw.Draw(full);frame=row['frame'];canvas.rectangle((0,0,W*2,header),fill='black');canvas.text((8,5),f'Frame {frame:05d} | circle L / square R | green DFM | yellow observation',fill='white');canvas.text((8,26),'red baseline | cyan published gate | white cross-hand | magenta wrist-only',fill='white');
 for ci,cam in enumerate((2,3)):
  if with_image:im=Image.open(a.images/f'cam{cam}'/f'{frame:05d}.jpg').convert('RGB').resize((W,H))
  else:
   im=Image.new('RGB',(W,H),(22,22,22)); dd=ImageDraw.Draw(im)
   for x in range(0,W,80):dd.line((x,0,x,H),fill=(60,60,60))
   for y in range(0,H,65):dd.line((0,y,W,y),fill=(60,60,60))
  full.paste(im,(ci*W,header));d=ImageDraw.Draw(full)
  for side,sh in [('left','L'),('right','R')]:
   x=row['wrists'][side]
   def pos(z):return [z[0]+ci*W/scale,z[1]]
   if x['reference_projected_uv'] is not None:mark(d,pos(x['reference_projected_uv'][ci]),col['reference'],sh)
   mark(d,pos(x['raw_uv'][ci]),col['observation'],sh)
   for key,c in [('baseline_500','baseline'),('published_gate_500','published_gate'),('published_gate_cross_hand_500','cross_hand'),('wrist_only_500','wrist_only')]:mark(d,pos(x['fits'][key]['projected_uv'][ci]),col[c],sh)
  d.rectangle((ci*W,header+H,ci*W+W,header+H+footer),fill='black');d.text((ci*W+8,header+H+6),f'cam{cam}   grid background = geometric schematic' if not with_image else f'cam{cam}',fill='white')
 return full
thumb=[]
for row in m['per_frame']:
 local=make(row,True); local.save(a.local_output/f"frame_{row['frame']:05d}.jpg",quality=88);make(row,False).save(a.geometry_output/f"frame_{row['frame']:05d}.png")
 thumb.append(local.resize((640,316)))
sheet=Image.new('RGB',(1280,316*((len(thumb)+1)//2)),'white')
for i,x in enumerate(thumb):sheet.paste(x,((i%2)*640,(i//2)*316))
sheet.save(a.local_output/'contact.jpg',quality=85)
