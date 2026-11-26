"""Report paired fits separately from direct measurements of the two contour targets."""
from pathlib import Path
import json
import numpy as np
import cv2
from scipy.spatial import cKDTree

RUN=Path(__file__).resolve().parent
PREVIOUS=RUN.parent/'2026-11-26_realvideo4_wilor_outline'
OUTPUTS=RUN.parents[1]/'outputs'
SIDES=['left','right']
def read(path):
    return json.loads(path.read_text())

def boundary(mask):
    return mask & ~cv2.erode(mask.astype('uint8'),np.ones((3,3),np.uint8)).astype(bool)

def distance(a,b):
    u,v=[np.stack(np.nonzero(boundary(m)),axis=-1)*4 for m in (a,b)]
    return float(np.mean([np.minimum(cKDTree(s).query(t)[0],64).mean() for s,t in ((u,v),(v,u))]))

def rms(values):
    return float(np.sqrt(np.mean(np.square(values))))

def main():
    targets=read(RUN/'evidence/masks.json')['views']
    fields=['root_px','tips_px','thumbtip_px','pinkytip_px','wrist_px']
    by_hand={}
    by_view={}
    for num in range(1,5):
        key=f'clip{num:03}'
        arch=read(PREVIOUS/'evidence'/(key+'.json'))['results']
        fit=read(RUN/'evidence'/(key+'.json'))['results']['wilor']
        for side in SIDES:
            hand=key+'_'+side
            by_hand[hand]={}
            for mode,rec in [('A',arch['control']),('B',arch['wilor']),('C',fit)]:
                data=rec['hands'][side]
                by_hand[hand][mode]=dict({k:data[k] for k in fields},normal_deg=data['normal_vs_stereo'],
                        seg_px=float(np.mean(data['segment_edge_px'])),max_joint_mm=rec['body_joint_max_mm'])
            for i,eye in enumerate(SIDES):
                name=key+'_'+eye+'_'+side[0].upper()
                item=targets[name]
                segpath=OUTPUTS/'realvideo4_dv4_20261125/seg'/(key+'_'+eye+'.npy')
                seg=np.load(segpath).argmax(0)==(6 if side=='left' else 15)
                record={}
                for method,folder in [('native','realvideo4_wilor_outline_20261126'),('anchored','realvideo4_wilor_wrist_20261126')]:
                    raw=cv2.imread(str(OUTPUTS/folder/'masks'/(name+'.png')),0)
                    mask=raw[2::4,2::4]>0
                    pts=np.asarray(item[method+'_tips_px'])
                    obs=np.asarray(data['target'][i])
                    record[method]=dict(wrist_to_auto_px=float(np.linalg.norm(np.asarray(item[method+'_wrist_px'])-obs[0])),
                        tips_to_auto_px=rms(np.linalg.norm(pts-obs[5:],axis=1)),
                        target_to_seg_px=distance(mask,seg))
                by_view[name]=record
    summary={}
    for mode in ('A','B','C'):
        hands=[row[mode] for row in by_hand.values()]
        summary[mode]={k:rms([h[k] for h in hands]) for k in fields}
        summary[mode].update(seg_px=float(np.mean([h['seg_px'] for h in hands])),normal_deg=float(np.mean([h['normal_deg'] for h in hands])),
                             max_joint_mm=max(h['max_joint_mm'] for h in hands))
    direct={m:dict(tips_to_auto_px=rms([v[m]['tips_to_auto_px'] for v in by_view.values()]),
                    wrist_to_auto_px=rms([v[m]['wrist_to_auto_px'] for v in by_view.values()]),
                    target_to_seg_px=float(np.mean([v[m]['target_to_seg_px'] for v in by_view.values()]))) for m in ('native','anchored')}
    improved={m:{k:sum(bool(h['C'][k]<h[m][k]) for h in by_hand.values()) for k in fields+['seg_px','normal_deg']} for m in ('A','B')}
    result=dict(summary=summary,C_improved_hands=improved,by_hand=by_hand,target_metrics=direct,target_per_eye=by_view,exit_status=0)
    (RUN/'evidence/summary.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':
    main()
