"""Recover the native hand template in the source shape for rest-frame calibration."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[1]
OUT=ROOT/'outputs/hand_pose_transfer_20261127'
NATIVE=ROOT/'outputs/wilor_stereo_fingers_20261126'
read=lambda path: json.loads(path.read_text())

def recover():
    import torch
    import numpy as np
    from wilor.models import load_wilor
    cfg=read(ROOT/'configs/pipeline.head01.developer.json')
    paths=cfg['paths']
    model,_=load_wilor(paths['wilor_checkpoint'],paths['wilor_config'])
    mano=model.mano.cpu().eval()
    native=read(RUN.parent/'2026-11-27_wilor_native_hands/evidence/results.json')['cases']
    guide=read(ROOT/'outputs/realvideo4_palm_20261126/guided.json')['selected']
    archive={}
    for key,receipt in native.items():
        path=NATIVE/(key+'.npz')
        assert hashlib.sha256(path.read_bytes()).hexdigest()==receipt['input_pack_sha256']
        p=np.load(path)
        params={k:torch.from_numpy(p[k])[None] for k in ('hand_pose','global_orient','betas')}
        side=key.split('_')[-1]
        mirror=np.array([-1,1,1]) if side=='L' else np.ones(3)
        with torch.no_grad():
            j=mano(**params,pose2rot=False).joints[0,:21].numpy()*mirror
            identity=torch.eye(3).reshape(1,1,3,3)
            rest=mano(global_orient=identity,hand_pose=identity.expand(1,15,3,3),betas=params['betas'],pose2rot=False).joints[0,:21].numpy()*mirror
        gpu=np.asarray(guide[key.rsplit('_',1)[0]][side]['local'])
        diff=np.max(np.linalg.norm(gpu-j,axis=1))*1000
        assert diff<.5,(key,diff)
        archive[key]=dict(rest=(rest-rest[:1]).tolist(),posed=(gpu-gpu[:1]).tolist(),cpu_gpu_mm=diff)
    (OUT/'native.json').write_text(json.dumps(archive))

if __name__=='__main__':
    if sys.argv[1:]==['recover']:
        recover()
    else:
        paths=read(ROOT/'configs/pipeline.head01.developer.json')['paths']
        mano=OUT/'mano_data'
        OUT.mkdir(parents=True,exist_ok=True)
        (OUT/'runtime').mkdir(exist_ok=True)
        if not mano.exists(): mano.symlink_to(Path(paths['wilor_repo'])/'mano_data',target_is_directory=True)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=os.pathsep.join((str(ROOT),paths['wilor_repo'])))
        for key in ('XDG_CACHE_HOME','MPLCONFIGDIR','TMPDIR','TORCH_HOME','YOLO_CONFIG_DIR'):
            path=OUT/'runtime'/key
            path.mkdir(exist_ok=True)
            env[key]=str(path)
        subprocess.run([paths['wilor_python'],'-Bu',str(Path(__file__).resolve()),'recover'],env=env,cwd=OUT,check=True)
