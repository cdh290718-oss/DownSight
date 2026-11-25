"""Full, unresampled stereo views in native coordinates with separate observation/model overlays."""
from pathlib import Path
import sys
import numpy as np
from PIL import Image,ImageDraw

RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[1]
sys.path.insert(0,str(ROOT/'runs/2026-11-25_val031_point_overlay'))
from render import BG,DARK,label,symbol
sys.path.insert(0,str(ROOT/'runs/2026-11-25_frozen100_point_errors'))
from refine import prev, BRANCHES, OUT, RUN as SOURCE
from pixels import name
FRAMES=['pair024','pair010','pair081','pair028','pair004']
MODES={'detector':'RW20000 检测点','fit':'最终人体重投影点'}


def cross(draw,point,mode):
    if mode=='detector':
        symbol(draw,point,True,5)
        return
    x,y=point
    for color,width in [(DARK,5),('#ff66dd',3)]:
        draw.line((x-6,y,x+6,y),fill=color,width=width)
        draw.line((x,y-6,x,y+6),fill=color,width=width)


def render(pair,mode,panels):
    sheet=Image.new('RGB',(1760,1036),BG)
    pen=ImageDraw.Draw(sheet)
    label(pen,(12,8),pair+'    '+MODES[mode]+'与官方二维点',30)
    cross(pen,(32,72),mode); label(pen,(52,57),MODES[mode],23)
    symbol(pen,(332,72),False,9); label(pen,(352,57),'官方二维点',23)
    for i,(im,gt,coords,names) in enumerate(panels):
        panel=im.copy()
        draw=ImageDraw.Draw(panel)
        for a,b in zip(gt,coords):
            draw.line((tuple(a),tuple(b)),fill=DARK,width=3)
            draw.line((tuple(a),tuple(b)),fill='white',width=1)
        for a in gt:
            symbol(draw,tuple(a),False,8)
        for b in coords:
            cross(draw,tuple(b),mode)
        start=i*888
        label(pen,(start+10,95),('左目','右目')[i],24)
        sheet.paste(panel,(start,132))
        dist=np.linalg.norm(coords-gt,axis=1)
        k=int(dist.argmax())
        label(pen,(start+10,1007),f'最大距离  {name(names[k])}  {dist[k]:.1f} px',19)
    path=RUN/'assets'/f'{pair}_{mode}.jpg'
    sheet.save(path,quality=93,subsampling=0,optimize=True)
    return path


def main():
    ref=prev.ref
    order=prev.read(SOURCE/'evidence/selection.json')
    pixels=prev.read(SOURCE/'evidence/pixels.json')
    scoring=prev.read(SOURCE/'evidence/refinement_scores.json')
    images,jids=ref.reference_images()
    manifest=prev.TM
    assert prev.sha(manifest)==pixels['manifest_sha256']
    prepared=ref.OUT/'prepared.json'
    assert prev.sha(prepared)==pixels['inputs'][prepared.name]
    items={x['pair']:x for x in prev.read(prepared)['items'] if x['condition']=='rw20000'}
    path=OUT/'checkpoints.npz'
    assert prev.sha(path)==order['checkpoints_sha256']
    pack=dict(np.load(path))
    model,cls=ref.common.setup_model()
    cameras=[cls(q) for q in ref.camera_files()]
    adapters=[prev.read(q)['input_adapter'] for q in ref.camera_files()]
    assert all(prev.sha(q)==pixels['inputs'][q.name] for q in ref.camera_files())
    truth=prev.read(ref.SOURCE/'reference.json')
    assert prev.sha(ref.SOURCE/'reference.json')==scoring['truth_sha256']
    names=[ref.common.NAMES[i] for i in ref.common.OBS]
    ids=np.array(ref.common.NATIVE_IDS)[ref.common.OBS]
    cases={};assets={}
    for pair in FRAMES:
        row=items[pair];winner=order['frames'][pair]['winner']
        assert winner==scoring['records'][pair]['selected'] and winner in BRANCHES
        idx=sorted(order['frames']).index(pair)*len(BRANCHES)+BRANCHES.index(winner)
        xyz=ref.full_joints(pack,model.layer['neutral'],idx)[ids]
        mpjpe=float(1000*np.linalg.norm(xyz-np.array(truth[pair]['gt'])[ref.common.OBS],axis=1).mean())
        assert abs(mpjpe-scoring['records'][pair]['scores'][1][0])<.001
        audit=cases[pair]=dict(winner=winner,mpjpe_mm=mpjpe,views={})
        panels={m:[] for m in MODES}
        for eye,(rec,camera,ad) in enumerate(zip(images[pair],cameras,adapters)):
            side=('left','right')[eye]
            assert rec['view']==side
            photo=manifest.parent/rec['image']
            assert prev.sha(photo)==rec['image_sha256']
            im=Image.open(photo).convert('RGB')
            assert im.size==tuple(rec['size'])==(872,872)
            official=np.array(rec['keypoints308_xy'])[jids]
            points=row['points']
            assert names==[q['name'] for q in points]
            mask=np.asarray(rec['keypoints308_supervision_mask'])[jids].astype(bool)
            assert np.array_equal(mask,[q['valid'][eye] for q in points]) and mask.all()
            detect=(np.array([q['xy'][eye] for q in points])-ad['offset'])/ad['scale']
            T=np.asarray(row['inputs']['views'][side]['camera']['left_to_view'])
            fit=(camera.world2camera(xyz@T[:3,:3].T+T[:3,3])-ad['offset'])/ad['scale']
            errors=np.linalg.norm(detect-official,axis=1)
            expected=[pixels['cases'][pair]['errors'][j][eye] for j in names]
            assert np.allclose(errors,expected,atol=1e-7,rtol=0)
            view=audit['views'][side]=dict(image_sha256=rec['image_sha256'],native_size=list(im.size))
            for mode,xy in [('detector',detect),('fit',fit)]:
                panels[mode].append((im,official,xy,names))
                view[mode+'_errors_px']=dict(zip(names,np.linalg.norm(xy-official,axis=1).tolist()))
                view[mode+'_outside']=[n for n,p in zip(names,xy) if (p<0).any() or (p>=im.size).any()]
        for m,data in panels.items():
            file=render(pair,m,data)
            assets[file.name]=prev.sha(file)
    prev.write(RUN/'evidence/gallery.json',dict(cases=cases,assets=assets,checkpoint_sha256=prev.sha(path),
                 manifest_sha256=prev.sha(manifest),canvas=[1760,1036],native_panel=[872,872],
                 command='<BODY_PYTHON> -B gallery.py',exit_status=0))


if __name__=='__main__':
    main()
