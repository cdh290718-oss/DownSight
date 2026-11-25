"""Real-camera inputs for the default v3 solver and its original candidate search."""
import sys
import shutil
import cv2
import numpy as np
import torch
from fileset import ROOT,RUN,OUT,SIDES,cameras,read,write,sha
from calibrate import rays,projection
sys.path.insert(0,str(ROOT/'runs/2026-11-25_real001_golden_d_v3'))
from run_pair import v3

def unit(v):
    return v/np.linalg.norm(v)

def prepare():
    manifest=read(ROOT/'configs/golden_d.2d-v3.json')
    assert manifest['method_id']=='golden-d-2d-v3'
    assert all(sha(ROOT/k['path'])==k['sha256'] for k in manifest['locked_files'])
    calibration=read(RUN/'evidence/calibration.json')
    T=np.asarray(calibration['T'])
    R,t=T[:3,:3],T[:3,3]
    center=-R.T@t
    right=unit(center)
    down=unit(np.array([0.,0.,1.])+R.T@[0.,0.,1.])
    up=-unit(down-right*(down@right))
    forward=unit(np.cross(-right,up))
    assert right[0]>0 and up[2]<0 and forward[1]<0
    basis=np.stack([-right,up,forward],axis=1)
    brow=center/2+basis@[0.,.05,-.05]
    previous=read(ROOT/'runs/2026-11-25_real001_head_mount/evidence/inputs.json')
    head=brow+basis@previous['brow_to_head_template_m']
    model,Camera=v3.ref.common.setup_model()
    layer=model.layer['neutral']
    rest=(layer.J_regressor@layer.v_template).detach().numpy()
    paths=cameras()
    cams=[Camera(p) for p in paths]
    # Confirm the actual backend (not just our converter) implements the native image model.
    uv=np.stack(np.meshgrid(np.linspace(0,1279,37),np.linspace(0,719,23)),-1).reshape(-1,2)
    backend_error=[]
    for p,c in zip(paths,cams):
        source=read(p)
        q=rays(uv,source)
        q/=np.linalg.norm(q,axis=1)[:,None]
        assert np.max(np.linalg.norm(c.camera2world_ray(uv)-q,axis=1))<1e-5
        z=c.world2camera(q.copy())
        err=np.max(np.linalg.norm(z-uv,axis=1))
        assert err<.01 and np.max(abs(z-projection(q,source)))<.01
        backend_error.append(float(err))
    template=read(v3.base.OUT/'prepared.json')['items'][0]['points']
    mapping=v3.prev.mapping()
    protocol=v3.base.OUT/'protocol.json'
    assert sha(protocol)==previous['protocol']
    det=read(OUT/'predictions.json');assert sha(OUT/'predictions.json')==read(RUN/'evidence/detection.json')['predictions_sha256']
    records=read(RUN/'evidence/frames.json')
    views=dict(left=dict(camera=dict(left_to_view=np.eye(4).tolist())),right=dict(camera=dict(left_to_view=T.tolist())))
    rows=[];summary={}
    for rec in records:
        key=rec['pair']
        samples=[det[key+'_'+s+'.png'] for s in SIDES]
        assert all(a['image_sha256']==rec['views'][s]['frame'] for a,s in zip(samples,SIDES))
        points=[]
        for proto,index in zip(template,mapping):
            xy=np.array([a['xy'][index] for a in samples])
            valid=np.all((xy>=0)&(xy<[1280,720]),axis=1)
            points.append(dict(name=proto['name'],index=proto['index'],native_index=proto['native_index'],group=proto['group'],
                               accepted=False,xy=xy.tolist(),valid=valid.tolist()))
        row=dict(pair=key,condition='baseline',inputs=dict(views=views),points=points,head_position=head.tolist())
        seed,reason=v3.base.trial.body_seed(row,cams)
        if reason:
            seed=v3.base.shoulder_hip_recovery(row,cams,rest)
        row['initial_body_rotation']=seed.tolist()
        summary[key]=dict(valid_joint_views=sum(sum(p['valid']) for p in points),fallback_reason=reason)
        rows.append(row)
    folder=v3.shard(0)
    folder.mkdir(parents=True,exist_ok=True)
    write(folder/'prepared.json',dict(items=rows))
    shutil.copyfile(protocol,folder/'protocol.json')
    write(RUN/'evidence/inputs.json',dict(shards={'0':sha(folder/'prepared.json')},protocol=sha(protocol),method=manifest['method_id'],
        backend_native_roundtrip_max_px=backend_error,baseline_m=float(np.linalg.norm(center)),brow_cam3_m=brow.tolist(),
        head_cam3_m=head.tolist(),template_brow_to_head_m=previous['brow_to_head_template_m'],summary=summary,exit_status=0))

if __name__=='__main__':
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    v3.RUN,v3.OUT,v3.FIT=RUN,OUT,OUT/'fit'
    v3.ref.camera_files=cameras
    mode=sys.argv[1]
    if mode=='prepare':
        prepare()
    else:
        sys.addaudithook(v3.base.fit.forbid_gt)
        {'fit':v3.fit,'candidates':v3.candidates}[mode](0)
