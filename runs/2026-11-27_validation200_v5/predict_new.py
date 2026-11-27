from pathlib import Path
import json,sys
import numpy as np
import cv2
RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[1]
OUT=ROOT/'outputs/validation200_v5_20261127'
sys.path.insert(0,str(RUN.parent/'2026-11-27_vidihand2_comparison'))
import predict
sys.path.insert(0,str(RUN.parent/'2026-11-25_realvideo4_dv3'))
from calibrate import projection

def inputs():
    folder=ROOT/'outputs/rw_a20000_dataset_20261117'
    pairs={}
    for row in predict.read(folder/'validation.json')['records']:
        pairs.setdefault(row['pair_id'],{})[row['view']]=row
    maps={}
    K=predict.videos['clip003_left']['K']
    y,x=np.indices((480,640))
    rays=np.stack([x,y,np.ones_like(x)],-1).reshape(-1,3)@np.linalg.inv(K).T
    for eye in ('left','right'):
        cam=predict.read(ROOT/'outputs/rw_frozen200_golden_d_20261118'/('camera_'+eye+'.json'))
        a=cam['input_adapter']
        maps[eye]=((projection(rays,cam)-a['offset'])/a['scale']).reshape(480,640,2).astype('f')
    assert len(pairs)==200
    for n,k in enumerate(sorted(pairs),1):
        for eye in ('left','right'):
            row=pairs[k][eye]
            path=folder/row['image']
            assert predict.digest(path)==row['image_sha256']
            img=cv2.remap(cv2.imread(str(path)),maps[eye],None,cv2.INTER_LINEAR)
            yield f'val{n:03d}_{eye}',dict(K=K,center_index=0),img[None,:,:,::-1]

predict.inputs=inputs
predict.OUT=OUT
(RUN/'evidence/predictions.json').write_text(json.dumps(predict.wilor()))
