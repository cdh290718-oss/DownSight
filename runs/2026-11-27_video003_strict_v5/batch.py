"""Use v4 directly as the seed for the frozen v5 wrist solver."""
from pathlib import Path
import sys
import numpy as np
import torch
RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[1]
PRIOR=RUN.parent/'2026-11-27_video34_uniform_v5'
SOURCE=ROOT/'outputs/video34_uniform_v5_20261127'
OUT=ROOT/'outputs/video003_strict_v5_20261127'
sys.path.insert(0,str(PRIOR))
import finish_frame as f
import render_pairs as render
from base import read,write,sha

def main(index):
    index=int(index)
    torch.set_num_threads(1)
    key=f'v003_{index+1:02}'
    manifest=read(ROOT/'configs/golden_d.current.json')
    assert manifest['method_id']=='golden-d-2d-v5'
    assert all(sha(ROOT/v['path'])==v['sha256'] for v in manifest['locked_files'])
    pose=SOURCE/'poses'/(key+'.npz')
    assert sha(pose)==read(PRIOR/'evidence'/(key+'.json'))['pose_hash']
    pack=dict(np.load(pose));x=pack['locked']
    rows=SOURCE/'fit/0/prepared.json'
    assert sha(rows)==read(PRIOR/'evidence/inputs.json')['shards']['0']
    row=read(rows)['items'][index]
    assert row['pair']==key
    detect=SOURCE/'predictions.json'
    assert sha(detect)==read(PRIOR/'evidence/detection.json')['predictions_sha256']
    pred=read(detect)
    w=f.w
    model,_=w.ref.prev.ref.common.setup_model()
    layer=model.layer['neutral'].eval()
    means=layer.pose_mean.detach().numpy().reshape(-1,3)[25:]
    neutral=np.zeros_like(x);neutral[23:]=-means
    _,rest=w.old.mesh(layer,neutral,dict(pack,global0=np.zeros(3)))
    world=[w.old.mat(x[21])@w.old.mat(pack['global0'])]
    for n in range(1,22):world.append(world[int(layer.parents[n])]@w.old.mat(x[n-1]))
    trans=[np.array(row['inputs']['views'][e]['camera']['left_to_view']) for e in f.SIDES]
    cams=list(map(read,f.cameras()))
    def project(points):
        return np.stack([w.old.projection(points@t[:3,:3].T+t[:3,3],c) for t,c in zip(trans,cams)])
    order=np.array([41]+[n for a in range(21,41,4) for n in range(a+3,a-1,-1)])
    final=x.copy();hands={}
    w.prior.RUN=PRIOR;w.prior.OUT=SOURCE;w.prior.KEYS=(key,)
    for i,side in enumerate(f.SIDES):
        obs=np.stack([np.array(pred[key+'_'+eye+'.png']['xy'])[order+(21 if i==0 else 0)] for eye in f.SIDES])
        distance=np.linalg.norm(obs-obs[:,:1],axis=2)
        inside=((obs>=0)&(obs<[1280,720])).all(-1)
        used=(inside&(distance<3.2*np.median(distance[:,w.ROOTS],axis=1)[:,None]))[:,w.FINGER_JOINTS]
        alt,choice=w.prior.choose_hand(key,i,layer,pack,means,rest,world,project,obs,used)
        if alt is not None:
            final[23+15*i:38+15*i]=alt[23+15*i:38+15*i]
            final[19+i]=alt[19+i]
        hands[side]=dict(choice,obs=obs.tolist(),valid=inside.tolist(),fit_mask=used.tolist())
    v,j=w.old.mesh(layer,final,pack)
    _,before=w.old.mesh(layer,x,pack)
    drift=float(1000*np.linalg.norm(j[:22]-before[:22],axis=1).max())
    assert drift<.001
    assert np.array_equal(final[:19],x[:19]) and np.array_equal(final[21:23],x[21:23])
    w.prior.stereo.OUT=OUT
    _,obj=w.prior.stereo.export(key,'v5',v,np.asarray(model.face,np.int32))
    np.savez_compressed(OUT/'poses'/(key+'_v5.npz'),**dict(pack,locked=final))
    for i,side in enumerate(f.SIDES):
        hands[side]['uv']=project(j[w.prior.IDS[i]]).tolist()
    write(OUT/(key+'_hands.json'),dict(source={'v4_sha256':sha(pose)},hands=hands,drift_mm=drift,obj=obj.name,obj_sha256=sha(obj),exit_status=0))
    render.RUN=RUN;render.OUT=OUT
    render.main(index)
    print(key,[(s,hands[s]['source']) for s in f.SIDES],drift,flush=True)
if __name__=='__main__':main(sys.argv[1])
