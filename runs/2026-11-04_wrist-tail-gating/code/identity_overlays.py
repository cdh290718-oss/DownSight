"""Overlay detected fingertip ownership and discarded suspect points on both cameras."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
p=argparse.ArgumentParser();p.add_argument('--observations',type=Path,required=True);p.add_argument('--cross',type=Path,required=True);p.add_argument('--images',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--first10',action='store_true');a=p.parse_args();d=np.load(a.observations,allow_pickle=False);names=d['names'].tolist();cross={(int(x['frame']),x['joint']):x for x in json.loads(a.cross.read_text())['decisions']};a.output.mkdir(exist_ok=False,parents=True);W,H=800,650;header=70;out=[]
start=0 if a.first10 else 10
for i,frame in enumerate(d['frames'][start:start+10],start=start):
 frame=int(frame);canvas=Image.new('RGB',(W*2,H+header),'black');draw=ImageDraw.Draw(canvas);draw.text((8,5),f'Frame {frame:05d} | L=orange, R=cyan; lines show detector labels, not confirmed identity',fill='white');draw.text((8,25),'circle = wrist; cross = fingertip; white ring = cross-hand candidate discarded',fill='white');draw.text((8,45),'T/I/M/R/P indicate thumb, index, middle, ring, pinky; invalid observations omitted',fill='white')
 for ci,cam in enumerate((2,3)):
  image=Image.open(a.images/f'cam{cam}'/f'{frame:05d}.jpg').convert('RGB').resize((W,H));canvas.paste(image,(ci*W,header));draw=ImageDraw.Draw(canvas)
  for side,color in [('left','orange'),('right','cyan')]:
   wi=names.index(side+'_wrist');u,v=d['xy'][i,ci,wi];wx=u*.5+ci*W;wy=v*.5+header;draw.ellipse((wx-7,wy-7,wx+7,wy+7),outline=color,width=3);draw.text((wx+7,wy-8),side[0].upper()+' wrist',fill=color,stroke_width=1,stroke_fill='black')
   for finger,label in [('thumb4','T'),('forefinger4','I'),('middle_finger4','M'),('ring_finger4','R'),('pinky_finger4','P')]:
    name=side+'_'+finger;j=names.index(name)
    if not d['valid'][i,ci,j]:continue
    x,y=d['xy'][i,ci,j];x=x*.5+ci*W;y=y*.5+header;draw.line((wx,wy,x,y),fill=color,width=2);draw.line((x-5,y-5,x+5,y+5),fill=color,width=3);draw.line((x-5,y+5,x+5,y-5),fill=color,width=3);draw.text((x+7,y-7),side[0].upper()+label,fill=color,stroke_width=1,stroke_fill='black')
    if cross[(frame,name)]['dropped']:draw.ellipse((x-10,y-10,x+10,y+10),outline='white',width=3)
 canvas.save(a.output/f'frame_{frame:05d}.jpg',quality=90)
