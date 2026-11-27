from base import *
import cv2,numpy as np
sys.path.insert(0,str(RUN.parent/'2026-11-27_vidihand2_comparison'))
import predict
from calibrate import projection

def inputs():
    for side,cal in zip(SIDES,cameras()):
        info=predict.videos['clip003_'+side]
        w,h=info['size']
        y,x=np.indices((h,w))
        unit=np.stack([x,y,np.ones_like(x)],-1).reshape(-1,3)@np.linalg.inv(info['K']).T
        remap=projection(unit,read(cal)).reshape(h,w,2).astype('f')
        for row in read(RUN/'evidence/frames.json'):
            key=row['pair']+'_'+side
            file=OUT/'images'/(key+'.png')
            assert sha(file)==row['views'][side]['frame']
            img=cv2.remap(cv2.imread(str(file)),remap,None,cv2.INTER_LINEAR)
            yield key,dict(K=info['K'],center_index=0),img[None,:,:,::-1]
predict.inputs=inputs
predict.OUT=OUT
write(RUN/'evidence/predictions.json',predict.wilor())
