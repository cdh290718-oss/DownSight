from pathlib import Path
import sys,hashlib,zipfile,types
import numpy as np,cv2,torch
RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[1]
OLD=RUN.parent/'2026-11-27_video34_uniform_v5'
OUT=ROOT/'outputs/video003_equal_v5_20261127'
SOURCE=ROOT/'outputs/video34_uniform_v5_20261127'
sys.path.insert(0,str(OLD))
import render_pairs as renderer
from base import read,write,sha

def main(index):
    key=f'v003_{int(index)+1:02}'
    # Reuse the independently verified paired comparison, including its exact
    # reproduction assertion for the original wrist weight 4.
    trial=RUN.parent/'2026-11-27_v003_04_equal_wrist/trial.py'
    script=trial.read_text().replace("RUN=Path(__file__).resolve().parent",f"RUN=Path({str(RUN)!r})")
    script=script.replace("key='v003_04'",f"key={key!r}")
    script=script.replace("dest=RUN.parents[1]/'outputs/v003_04_equal_wrist_20261127'",f"dest=Path({str(OUT)!r})")
    script=script.replace("dest/'final.npz'",f"dest/'poses'/'{key}_v5.npz'")
    script=script.replace("RUN/'assets/side.jpg'",f"RUN/'assets/{key}_side.jpg'")
    script=script.replace("RUN/'evidence/results.json'",f"RUN/'evidence/{key}_comparison.json'")
    exec(compile(script,str(trial),'exec'),{'__name__':'__main__'})
    rec=read(SOURCE/(key+'_hands.json'))
    obj=OUT/'export'/(key+'_equal_cam3_m.obj')
    assert sha(obj)==read(RUN/'evidence'/(key+'_comparison.json'))['1']['obj_sha256']
    pack=dict(np.load(OUT/'poses'/(key+'_v5.npz')))
    model,_=renderer.w.ref.prev.ref.common.setup_model()
    _,j=renderer.w.old.mesh(model.layer['neutral'].eval(),pack['locked'],pack)
    row=read(SOURCE/'fit/0/prepared.json')['items'][int(index)]
    transforms=[np.array(row['inputs']['views'][e]['camera']['left_to_view']) for e in ('left','right')]
    cams=list(map(read,renderer.cameras()))
    for i,side in enumerate(('left','right')):
        rec['hands'][side]['uv']=np.stack([renderer.projection(j[renderer.w.prior.IDS[i]]@t[:3,:3].T+t[:3,3],c) for t,c in zip(transforms,cams)]).tolist()
        rec['hands'][side]['source']=read(RUN/'evidence'/(key+'_comparison.json'))['1']['hands'][side]['source']
    rec.update(obj=obj.name,obj_sha256=sha(obj))
    write(OUT/(key+'_hands.json'),rec)
    renderer.RUN=RUN;renderer.OUT=OUT
    renderer.main(index)

if __name__=='__main__':main(sys.argv[1])
