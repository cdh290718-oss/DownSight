"""Overlay val055 with the same symbols and view conventions as val031."""
import sys
import json
import math
from pathlib import Path
from PIL import Image, ImageDraw

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
sys.path.insert(0,str(ROOT/'runs/2026-11-25_val031_point_overlay'))
from render import PREV, OUT, BG, DARK, EYES, JOINTS, read, digest, label, header, mark

def main():
    evidence=read(PREV/'evidence/input_errors.json')
    input_file=OUT/'fit/prepared.json'
    assert digest(input_file)==evidence['prepared_sha256']
    rows={r['pair']:r for r in read(input_file)['items']}
    det=rows['val055_detector']['points']
    gt=rows['val055_projected_all']['points']
    photos=read(OUT/'image_inputs.json')['val055']
    manifest=ROOT/'outputs/rw_a20000_dataset_20261117/validation.json'
    assert digest(manifest)==evidence['manifest_sha256']
    records={r['image_sha256']:r for r in read(manifest)['records']}
    overview=Image.new('RGB',(1100,696),BG)
    detail=Image.new('RGB',(1100,904),BG)
    header(overview,'S5 / val055   双目全身 12 点')
    header(detail,'S5 / val055   肩、肘、腕局部对照')
    stats={}
    for v,eye in enumerate(EYES):
        info=photos[v]; raw=manifest.parent/info['image']
        rec=records[info['sha256']]
        assert (rec['view'],rec['subject'],rec['frame'])==(eye,'S5',0)
        assert digest(raw)==info['sha256'] and rec['image']==info['image']
        im=Image.open(raw).convert('RGB')
        assert im.size==(872,872)
        cal=ROOT/'outputs/rw_frozen200_golden_d_20261118'/('camera_'+eye+'.json')
        assert digest(cal)==evidence['camera_sha256'][v]
        a=read(cal)['input_adapter']
        def orig(xy):
            return tuple((x-off)/a['scale'] for x,off in zip(xy,a['offset']))
        pts={}
        stats[eye]=dict(image_sha256=digest(raw),errors_px={},detector_minus_projection_xy_px={})
        for p,q in zip(det,gt):
            assert p['name']==q['name'] and p['valid']==q['valid']
            if not p['valid'][v]:
                continue
            name=p['name']
            pair=orig(p['xy'][v]),orig(q['xy'][v])
            error=math.hypot(*(x-y for x,y in zip(*pair)))
            assert abs(error-evidence['frames']['val055'][eye]['det_vs_projection'][name])<1e-8
            stats[eye]['errors_px'][name]=error
            stats[eye]['detector_minus_projection_xy_px'][name]=[x-y for x,y in zip(*pair)]
            pts[name]=pair
        assert len(pts)==12
        x0=14+548*v
        word=['左目','右目'][v]
        label(ImageDraw.Draw(overview),(x0,104),word,23)
        panel=im.resize((524,524),Image.Resampling.LANCZOS)
        draw=ImageDraw.Draw(panel)
        for ds,gs in pts.values():
            mark(draw,*[tuple((x+.5)*524/872-.5 for x in xy) for xy in [ds,gs]])
        overview.paste(panel,(x0,145))
        for row,(arm,body_side) in enumerate([('L','左臂'),('R','右臂')]):
            points=[pts[arm+'_'+j] for j in JOINTS]
            coords=[xy for pair in points for xy in pair]
            width,height=524,322
            xlo,xhi=min(x for x,y in coords),max(x for x,y in coords)
            ylo,yhi=min(y for x,y in coords),max(y for x,y in coords)
            ch=math.ceil(max(yhi-ylo+68,(xhi-xlo+72)*height/width))
            cw=round(ch*width/height)
            x=int(max(0,min(im.width-cw,(xlo+xhi-cw)/2)))
            y=int(max(0,min(im.height-ch,(ylo+yhi-ch)/2)))
            panel=im.crop((x,y,x+cw,y+ch)).resize((width,height),Image.Resampling.LANCZOS)
            draw=ImageDraw.Draw(panel)
            for part,(d,g) in zip(['肩','肘','腕'],points):
                ds,gs=[((sx-x+.5)*width/cw-.5,(sy-y+.5)*height/ch-.5) for sx,sy in (d,g)]
                mark(draw,ds,gs)
                tx=max(2,min(width-24,min(ds[0],gs[0])-23))
                ty=max(1,min(height-25,min(ds[1],gs[1])-23))
                label(draw,(tx,ty),part,20,stroke_width=2,stroke_fill=DARK)
            top=108+row*390
            label(ImageDraw.Draw(detail),(x0,top),word+'  /  人物'+body_side,23)
            detail.paste(panel,(x0,top+36))
    files={}
    for image,name in [(overview,'stereo'),(detail,'arms')]:
        path=RUN/'assets'/(name+'.jpg')
        image.save(path,quality=91,subsampling=0,optimize=True)
        files[name]=dict(sha256=digest(path),dimensions=image.size)
    (RUN/'evidence/render.json').write_text(json.dumps(dict(frame=0,subject='S5',pair='val055',views=stats,assets=files,prepared_sha256=digest(input_file),max_check_tolerance_px=1e-8,exit_status=0),indent=2)+'\n')


if __name__=='__main__':
    main()
