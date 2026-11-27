"""Aggregate squared Euclidean residuals using a common in-frame fingertip mask."""
from pathlib import Path
import json
import numpy as np

RUN=Path(__file__).resolve().parent
MANO=RUN.parent/'2026-11-26_wilor_stereo_fingers'
TIPS=[4,8,12,16,20]
NAMES=['thumb','index','middle','ring','pinky']
read=lambda p:json.loads(p.read_text())
def rms(v):
    return float(np.sqrt(np.mean(np.square(v)))) if len(v) else None

def main():
    reference=read(MANO/'evidence/metrics.json')
    report={}
    pool={name:[] for name in ['A','B','MANO']}
    edges={name:[] for name in ['A','B']}
    for key in ('clip003','clip004'):
        trial=read(RUN/'evidence'/(key+'.json'))['results']
        for side in ('left','right'):
            row={}
            for mode in ('A','B'):
                h=trial[mode]['hands'][side]
                obs=np.array(h['observation'])
                errors=np.linalg.norm(np.array(h['uv'])-obs,axis=-1)
                mask=np.array(h['inside'])[:,TIPS]
                tips=errors[:,TIPS]
                edge=h['hand_edge_px']
                pool[mode].extend(tips[mask])
                edges[mode].extend(edge)
                row[mode]=dict(tip_px=rms(tips[mask]),wrist_px=rms(errors[:,0]),edge_px=float(np.mean(edge)),
                    digit_px={d:rms(tips[:,n][mask[:,n]]) for n,d in enumerate(NAMES)},counts=mask.sum(axis=1).tolist(),
                    tip_px_per_eye=[rms(e[m]) for e,m in zip(tips,mask)],train_counts=np.sum(h['used'],axis=1).tolist())
            mano=reference[key][side]
            val=np.array(mano['constrained']['per_eye_tips'])[mask]
            assert len(val)==mano['constrained']['in_frame_count']
            assert abs(row['A']['tip_px']-mano['smplx']['visible_frame_tip_rmse'])<.02
            pool['MANO'].extend(val)
            row['MANO']=dict(tip_px=rms(val))
            report[key+'_'+side]=row
    assert all(len(x)==33 for x in pool.values())
    results=dict(rows=report,pooled_px={k:rms(v) for k,v in pool.items()},
                 mean_hand_edge_px={k:float(np.mean(v)) for k,v in edges.items()},tip_count=33,exit_status=0)
    (RUN/'evidence/metrics.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results['pooled_px']))
if __name__=='__main__':
    main()
