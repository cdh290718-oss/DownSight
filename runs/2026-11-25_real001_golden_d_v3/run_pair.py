"""Adapt the hash-locked Golden D v3 runner to the native real001 camera pixels."""
import sys
import shutil
from pathlib import Path
import numpy as np
import torch

RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[1]
OUT=ROOT/'outputs/real001_golden_d_v3_20261125'
OLD=ROOT/'outputs/real001_golden_d_four_models_20261118'
SOURCE=ROOT/'runs/2026-11-18_real001_golden_d_four_models'
sys.path.insert(0,str(ROOT/'runs/2026-11-23_test500_joint'))
import body as v3
read,write,sha=v3.read,v3.write,v3.sha
KEY='real001'
sides=['left','right']
def cameras():
    return [OUT/('camera_'+s+'.json') for s in sides]

def prepare():
    (OUT/'images').mkdir(parents=True,exist_ok=True)
    v3.shard(0).mkdir(parents=True,exist_ok=True)
    manifest=read(ROOT/'configs/golden_d.current.json')
    assert manifest['method_id']=='golden-d-2d-v3'
    for locked in manifest['locked_files']:
        assert sha(ROOT/locked['path'])==locked['sha256']
    logs=read(SOURCE/'evidence/preparation.json')
    detector=read(SOURCE/'evidence/detection.json')
    digests={}
    rows=read(OLD/'records.json')
    for i,(rec,path) in enumerate(zip(rows,cameras())):
        assert rec['camera']==['cam3','cam4'][i]
        im,cal=[Path(rec['row'][k]) for k in ['image','calibration']]
        assert sha(im)==detector['images'][i]['sha256'] and sha(cal)==logs['projection_checks'][i]['calibration_sha256']
        digests[rec['camera']]=dict(image=sha(im),calibration=sha(cal),id=rec['id'])
        c=read(cal); c['input_adapter']=dict(scale=1.,offset=[0.,0.]);write(path,c)
        dst=OUT/'images'/(KEY+'_'+sides[i]+'.png')
        if not dst.exists(): dst.symlink_to(im)
        assert sha(dst)==sha(im)
    old=next(r for r in read(OLD/'prepared.json')['items'] if r['detector']=='rw20000')
    predictions=read(OLD/'detections.json')['rw20000']
    for i,pt in enumerate(old['points']):
        assert pt['valid']==logs['common_joint_view_mask'][i]
        for eye in range(2):
            assert np.allclose(pt['xy'][eye],predictions[eye]['points'][i]['xy'])
    assert old['anchor']==logs['head_anchor'] and old['inputs']==logs['camera_transforms']
    template=read(v3.base.OUT/'prepared.json')['items'][0]['points']
    by_id={p['native_index']:p for p in old['points']}
    points=[]
    for t in template:
        q=by_id[t['native_index']]
        assert (t['index'],t['group'])==(q['index'],q['group'])
        points.append(dict(t,xy=q['xy'],valid=q['valid']))
    assert len(points)==12 and sum(sum(q['valid']) for q in points)==22 and not any(q['accepted'] for q in points)
    model,cam=v3.ref.common.setup_model()
    cs=[cam(p) for p in cameras()]
    row=dict(pair=KEY,condition='baseline',inputs=old['inputs'],points=points,head_position=old['anchor']['position'])
    rotation,reason=v3.base.trial.body_seed(row,cs)
    if reason:
        layer=model.layer['neutral']
        rotation=v3.base.shoulder_hip_recovery(row,cs,(layer.J_regressor@layer.v_template).detach().numpy())
    row['initial_body_rotation']=rotation.tolist()
    dest=v3.shard(0)
    write(dest/'prepared.json',dict(items=[row]))
    cfg=v3.base.OUT/'protocol.json'
    assert sha(cfg)==read(v3.base.RUN/'evidence/inputs.json')['protocol_sha256']
    shutil.copyfile(cfg,dest/'protocol.json')
    write(RUN/'evidence/inputs.json',dict(inputs=digests,shards={'0':sha(dest/'prepared.json')},protocol_sha256=sha(cfg),
        detector_sha256=detector['weights_sha256']['rw20000'],detection_cache_sha256=sha(OLD/'detections.json'),
        head_source_sha256=logs['head_source_sha256'],seed_fallback_reason=reason,
        available={p['name']:p['valid'] for p in points},views_retained=22,method=manifest['method_id']))

if __name__=='__main__':
    torch.set_num_threads(1)
    v3.cv2.setNumThreads(1)
    v3.RUN,v3.OUT,v3.FIT=RUN,OUT,OUT/'fit'
    v3.ref.camera_files=cameras
    if sys.argv[1]=='prepare':
        prepare()
    else:
        sys.addaudithook(v3.base.fit.forbid_gt)
        (v3.fit if sys.argv[1]=='fit' else v3.candidates)(0)
