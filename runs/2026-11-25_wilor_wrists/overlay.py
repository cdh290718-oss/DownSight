"""Display wrist center differences on matched original-image crops; draw no fallback WiLoR marker."""
from pathlib import Path
from PIL import Image,ImageDraw
import sys
from detect import EYES,FRAMES,OUT,ROOT,RUN,read

sys.path.insert(0,str(ROOT/'runs/2026-11-25_val031_point_overlay'))
from render import label,BG,DARK


def mark(draw,pos,color,kind):
    x,y=pos
    if kind=='rw':
        for d in (-1,1):
            draw.line((x-6,y-6*d,x+6,y+6*d),fill=color,width=3)
    elif kind=='official':
        draw.ellipse((x-8,y-8,x+8,y+8),outline=color,width=3)
    elif kind=='wilor':
        draw.rectangle((x-6,y-6,x+6,y+6),outline=color,width=3)
    else:
        draw.polygon([(x,y-7),(x-7,y+6),(x+7,y+6)],outline=color,width=3)


def main():
    metrics=read(RUN/'evidence/scores.json')['frames']
    sel=read(RUN/'evidence/detections.json')['selected']
    photos=read(OUT/'images.json')['frames']
    colors=dict(rw='#34e3ff',official='#ffae3c',wilor='#5ffa7e',guided='#ff69c6')
    (RUN/'assets').mkdir(exist_ok=True)
    for n,f in enumerate(FRAMES):
        canvas=Image.new('RGB',(1024,750),BG)
        d=ImageDraw.Draw(canvas)
        label(d,(18,6),f+'　腕点位置（人物左右）',26)
        for i,(src,name) in enumerate([('official','官方'),('rw','RW20000'),('wilor','WiLoR 自动框'),('guided','WiLoR 引导框')]):
            mark(d,(28+i*249,66),colors[src],src)
            label(d,(46+i*249,50),name,23)
        for v,eye in enumerate(EYES):
            path=Path(photos[2*n+v]['image'])
            im=Image.open(path).convert('RGB')
            for a,side in enumerate(('L','R')):
                pts={src:metrics[f][src]['wrist'][side]['uv_px'][v] for src in ('rw','wilor','guided')}
                pts['official']=metrics[f]['rw']['wrist'][side]['official_px'][v]
                missing=sel[path.stem][side]['uv'] is None
                if missing:
                    del pts['wilor']
                x0=max(0,min(im.width-240,int((max(x for x,y in pts.values())+min(x for x,y in pts.values()))/2)-120))
                y0=max(0,min(im.height-144,int((max(y for x,y in pts.values())+min(y for x,y in pts.values()))/2)-72))
                patch=im.crop((x0,y0,x0+240,y0+144)).resize((480,288),Image.Resampling.LANCZOS)
                for src in ('official','wilor','rw','guided'):
                    if src in pts:
                        x,y=pts[src]
                        mark(ImageDraw.Draw(patch),((x-x0+.5)*2-.5,(y-y0+.5)*2-.5),colors[src],src)
                x,y=12+v*512,92+a*330
                label(d,(x,y),['左目','右目'][v]+' / '+['左腕','右腕'][a]+('　自动框漏检' if missing else ''),22)
                canvas.paste(patch,(x,y+33))
        canvas.save(RUN/'assets'/(f+'.jpg'),quality=90,optimize=True)


if __name__=='__main__':
    main()
