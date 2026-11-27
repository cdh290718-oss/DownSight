"""Use the same rectification, model weights, detector and box selection as the previous comparison."""
from pathlib import Path
import sys
import cv2
import numpy as np
RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
OUT = ROOT/'outputs/perspective_wrist_gallery_20261127'
sys.path.insert(0,str(RUN.parent/'2026-11-27_vidihand2_comparison'))
import predict
sys.path.insert(0,str(RUN.parent/'2026-11-27_vidihand2_wrist'))
import video_inputs as vi

def inputs():
    archive = vi.read(vi.RUN/'evidence/video.json')
    meta = vi.read(vi.RUN.parent/'2026-11-25_realvideo4_dv3/evidence/frames.json')
    for eye,cal in zip(vi.SIDES,vi.cameras()):
        ref = archive['clip003_'+eye]
        w,h = ref['size']
        y,x = np.indices((h,w))
        unit = np.stack([x,y,np.ones_like(x)],axis=-1).reshape(-1,3)@np.linalg.inv(ref['K']).T
        remap = vi.projection(unit,vi.read(cal)).reshape(h,w,2).astype('f')
        for item in meta[:2]:
            key = item['pair']+'_'+eye
            photo = vi.OLD/'images'/(key+'.png')
            assert vi.sha(photo)==item['views'][eye]['frame']
            image = cv2.remap(cv2.imread(str(photo)),remap,None,cv2.INTER_LINEAR)
            yield key,dict(center_index=0,K=ref['K']),image[None,:,:,::-1]

if __name__=='__main__':
    predict.inputs = inputs
    predict.OUT = OUT
    vi.write(RUN/'evidence/predictions.json',predict.wilor())
