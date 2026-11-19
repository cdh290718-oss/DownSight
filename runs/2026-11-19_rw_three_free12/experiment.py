"""Run the three stored detectors on the named ten-pair comparison."""
import sys
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import least_squares

RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[1]
sys.path.insert(0,str(ROOT/'runs/2026-11-19_rw_position_only'))
import trial
from trial import common,read,write,sha,fit
OUT=ROOT/'outputs/rw_three_free12_20261119'
REF=ROOT/'runs/2026-11-18_rw-a20000-evaluation'
MODELS={'baseline':'original','rw5000':'rw5000','rw20000':'rw20000'}


def shoulder_hip_recovery(item,cams,rest):
    # Use the same four landmarks when independent stereo intersections are ill-conditioned.
    js={'L_Shoulder':16,'R_Shoulder':17,'L_Hip':1,'R_Hip':2}
    pts=[p for p in item['points'] if p['name'] in js]
    pos=rest[[js[p['name']] for p in pts]]
    pos=pos-pos.mean(0)
    def residual(v):
        q=pos@cv2.Rodrigues(v[:3])[0].T+v[3:]
        err=[]
        for side,(eye,cam) in enumerate(zip(['left','right'],cams)):
            T=np.asarray(item['inputs']['views'][eye]['camera']['left_to_view'])
            pred=cam.world2camera(q@T[:3,:3].T+T[:3,3])
            err.extend((pred[j]-p['xy'][side]) for j,p in enumerate(pts) if p['valid'][side])
        return np.asarray(err).ravel()
    init=np.r_[cv2.Rodrigues(trial.FIXED)[0].ravel(),[0.,0.,.4]]
    sol=least_squares(residual,init,loss='soft_l1',f_scale=8,max_nfev=400)
    assert sol.success and np.isfinite(sol.x).all()
    return cv2.Rodrigues(sol.x[:3])[0]


def main():
    source=ROOT/'outputs/rw_a20000_evaluation_20261118/d_fit/prepared.json'
    assert sha(source)==read(REF/'evidence/fit_preparation.json')['prepared_sha256']
    cfg=read(REF/'protocol.json')
    cfg.update(head_rotation_weight=0.,landmark_ratios=read(trial.PREV/'evidence/calibration.json')['ratios'])
    model,cam_cls=common.setup_model()
    layer=model.layer['neutral']
    rest=(layer.J_regressor@layer.v_template).detach().numpy()
    cams=[cam_cls(p) for p in common.camera_files()]
    items=[];audit=[];nuisance={}
    for row in read(source)['items']:
        pair=row['pair'];key=row['condition']
        seed,reason=trial.body_seed(row,cams)
        if reason:seed=shoulder_hip_recovery(row,cams,rest)
        pts=[dict({k:p[k] for k in ['name','index','native_index','group','xy','valid']},accepted=False) for p in row['points']]
        assert len(pts)==12 and not any(p['accepted'] for p in row['points'])
        shared=[row['inputs'],row['anchor']['position'],[(p['name'],p['valid']) for p in pts]]
        assert nuisance.setdefault(pair,shared)==shared
        items.append(dict(pair=pair,condition=key,points=pts,head_position=row['anchor']['position'],inputs=row['inputs'],initial_body_rotation=seed.tolist()))
        audit.append(dict(pair=pair,model=MODELS[key],stereo_seed_issue=reason,valid_view_points=sum(sum(p['valid']) for p in pts)))
    assert len(items)==30 and len(nuisance)==10
    write(OUT/'prepared.json',dict(items=items));write(OUT/'protocol.json',cfg)
    write(RUN/'evidence/inputs.json',dict(source_sha256=sha(source),solver_input_sha256=sha(OUT/'prepared.json'),protocol_sha256=sha(OUT/'protocol.json'),rows=audit))
    fit.Problem=trial.TorsoOnly
    fit.RUN=fit.OUT=OUT
    sys.argv=[__file__,'--threads','1']
    fit.main()

if __name__=='__main__': main()
