import json
from pathlib import Path
import numpy as np
RUN = Path(__file__).resolve().parent

def load(run, name):
    return json.loads((run/'evidence'/(name+'.json')).read_text())

def rmse(diff):
    return float(np.sqrt(np.mean(np.sum(diff**2,axis=-1))))

def hand(run,key):
    hands = load(run,key)['results']['control']['hands']
    diff = np.array([np.array(h['uv'])-h['target'] for h in (hands['left'],hands['right'])])
    return dict(wrist=rmse(diff[:,:,0]),tips=rmse(diff[:,:,5:]),aligned=rmse(diff[:,:,5:]-diff[:,:,:1]),
                roots=rmse(diff[:,:,1:5]))

result={}
for key in ['clip003','clip004']:
    arm=load(RUN,key+'_arm')
    result[key] = {name:hand(folder,key) for name,folder in [('orig', RUN.parent/'2026-11-26_realvideo4_wilor_outline'),
        ('purple',RUN.parent/'2026-11-26_realvideo34_purple_hand_a'),('refit',RUN)]}
    for step in ['initial','fitted']:
        diff = np.array(arm[step])-arm['target']
        result[key][step+'_elbow']=rmse(diff[:,:2])
    result[key]['bone_mm']=arm['max_bone_drift_mm']
    result[key]['head_mm']=arm['head_drift_mm']
print(json.dumps(result,indent=2))
