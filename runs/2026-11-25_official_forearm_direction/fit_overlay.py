"""Compare the final projected left forearm to official pixels for val055."""
import sys
from pathlib import Path
import json
import numpy as np
from compare import FRAMES,METHODS,RUN,ROOT,OUT,SOURCE,p
from PIL import Image,ImageDraw,ImageFont

def main():
    model,cam=p.ref.common.setup_model()
    imgs=p.pair_images(p.VM)[54]
    params=dict(np.load(OUT/'selected.npz'))
    assert p.sha(OUT/'selected.npz')==p.read(RUN/'evidence/fit.json')['selected_sha256']
    item=next(x for x in p.read(SOURCE/'fit/prepared.json')['items'] if x['pair']=='val055_annotated_all')
    canvas=Image.new('RGB',(1000,770),'#14222b')
    pen=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
    pen.text((18,12),'val055   Left forearm on original images',font=font,fill='white')
    pen.text((18,48),'Orange circle = official 2D     Cyan cross = fitted joint',font=font,fill='white')
    for view,(im,cfile) in enumerate(zip(imgs,p.ref.camera_files())):
        c=cam(cfile); a=p.read(cfile)['input_adapter']; side=im['view']
        T=np.array(item['inputs']['views'][side]['camera']['left_to_view'])
        ij={v['name']:v for v in item['points']}
        official=np.array([ij['L_'+k]['xy'][view] for k in ['Elbow','Wrist']])
        official=(official-a['offset'])/a['scale']
        raw=p.VM.parent/im['image'];assert p.sha(raw)==im['image_sha256']
        bg=Image.open(raw).convert('RGB')
        size=np.array([480,278]); crop=size//2
        origin=np.clip(np.rint(official.mean(0)-crop/2),0,np.array(bg.size)-crop).astype(int)
        patch=bg.crop(tuple(origin)+tuple(origin+crop)).resize(tuple(size),Image.Resampling.LANCZOS)
        for row,method in enumerate(['official2d','direction_fixed']):
            idx=FRAMES.index('val055')*len(METHODS)+METHODS.index(method)
            native=p.ref.full_joints(params,model.layer['neutral'],idx)[[18,20]]
            projected=(c.world2camera(native@T[:3,:3].T+T[:3,3])-a['offset'])/a['scale']
            tile=patch.copy();draw=ImageDraw.Draw(tile)
            for k,pts in enumerate([official,projected]):
                points=[tuple(t) for t in ((pts-origin+.5)*2-.5)]
                col=['#ff991e','#2df2ff'][k]
                draw.line(points,fill='#182126',width=4); draw.line(points,fill=col,width=2)
                for x,y in points:
                    if k==0: draw.ellipse((x-8,y-8,x+8,y+8),outline=col,width=3)
                    else:
                        for sign in [-1,1]:
                            draw.line((x-5,y-5*sign,x+5,y+5*sign),fill=col,width=2)
            x,y=14+view*494,94+row*332
            pen.text((x,y),side+' eye   /   '+['2D only','fixed direction'][row],font=font,fill='white')
            canvas.paste(tile,(x,y+32))
    path=RUN/'assets/fits.jpg';canvas.save(path,quality=93,subsampling=0)
    p.write(RUN/'evidence/fit_view.json',{'sha256':p.sha(path),'size':list(canvas.size)})
if __name__=='__main__':
    main()
