"""Change only the initial body rotation in the frozen auto-12, torso-locked solver."""
import sys
from pathlib import Path
import cv2
import numpy as np
RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[1]
PREV=ROOT/'runs/2026-11-19_rw_position_only'
sys.path.insert(0,str(PREV))
import trial
from trial import fit,common,read,write,sha
OUT=ROOT/'outputs/rw_auto_history_20261119'
E=RUN/'evidence'
KEYS={'baseline':'fixed','body':'body','proxy':'proxy','history':'history'}


def head_frame(p):
    r=cv2.Rodrigues(p['global_orient'].astype(float).reshape(3))[0]
    body=p['body_pose'].astype(float).reshape(21,3)
    for j in (3,6,9,12,15):
        r=r@cv2.Rodrigues(body[j-1])[0]
    return r


def prepare():
    refs=read(trial.OUT/'prepared.json')['items']
    proxy=read(trial.PRIOR/'protocol.json')['camera_R_head_proxy']
    cfg=read(trial.OUT/'protocol.json')
    cfg.update(conditions=list(KEYS),scope='auto12 torso lock, four body starts')
    rows=[]
    sources=[]
    for ref in refs:
        if ref['condition']!='baseline':continue
        pair=ref['pair']
        frame=common.ARCH/'body_work_full_image'/(pair+'_left')/'frames'/(pair+'_left')
        params=frame/'fused/body_params.npz'
        if not params.is_file():params=frame/'body/params.npz'
        with np.load(params) as p:
            history=head_frame(p)
        cached=ROOT/'outputs/unrealego_gt2d_ideal_20261115'/pair/'anchors/oracle.json'
        anchor=read(cached)
        diff=np.max(np.abs(history-anchor['rotation']))
        assert diff<1e-8 and ref['head_position']==anchor['position']
        # Use the exact historical matrix consumed by the official-15 run.
        histref=next(r for r in read(trial.PRIOR/'prepared.json')['items'] if r['pair']==pair and r['condition']=='15torso')
        assert histref['anchor']==anchor
        body=next(r for r in refs if r['pair']==pair and r['condition']=='auto_body')
        assert all(ref[k]==body[k] for k in ['inputs','points','head_position'])
        seeds=[ref['initial_body_rotation'],body['initial_body_rotation'],proxy,anchor['rotation']]
        for key,seed in zip(KEYS,seeds):
            rows.append(dict(ref,condition=key,initial_body_rotation=seed))
        assert sum(any(p['valid']) for p in ref['points'])==12
        sources.append(dict(pair=pair,archive_stage=str(params.relative_to(frame)),archive_sha256=sha(params),cached_anchor_sha256=sha(cached),rotation_max_element_delta=float(diff)))
    assert len(rows)==16
    write(OUT/'prepared.json',dict(items=rows))
    write(OUT/'protocol.json',cfg)
    write(E/'inputs.json',dict(sources=sources,position_only_input_sha256=sha(trial.OUT/'prepared.json'),
          controls_identical_except_condition_and_seed=True,solver_input_sha256=sha(OUT/'prepared.json'),
          protocol_sha256=sha(OUT/'protocol.json'),head_rotation_weight=cfg['head_rotation_weight']))


if __name__=='__main__':
    prepare()
    fit.RUN=fit.OUT=OUT
    fit.Problem=trial.TorsoOnly
    sys.argv=[__file__,'--threads','1']
    fit.main()
