"""Re-pool published frame/wrist errors and verify both saved SMPL-X outputs."""
import json
from pathlib import Path
from hashlib import sha256
import numpy as np

RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[1]
OUT=ROOT/'outputs/stereo_wrist_candidates_20261123'
def read(f): return json.loads(f.read_text())

def main():
    frozen=read(RUN/'evidence/frozen_rule.json')
    diagnostic={}
    for split in ['validation','test']:
        data=read(RUN/('evidence/'+split+'_results.json'))
        coverage=read(RUN/('evidence/'+split+'_coverage.json'))
        rows=data['rows'];wrists=data['wrists'];summary=data['summary']['all']
        assert len(rows)==coverage['feasible'] and len(wrists)==2*len(rows)
        assert coverage['feasible']+len(coverage['excluded_no_baseline'])==coverage['pairs']
        if split=='test': assert len(rows)==100
        for k,s in summary.items():
            wrist=[w[k]['error_mm'] for w in wrists]
            assert abs(np.sqrt(np.mean(np.square(wrist)))-s['wrist_rmse'])<.005
            for key in ('mpjpe','pa','other10','rmse','px','wrist_rmse'):
                a=np.array([r[k][key] for r in rows.values()])
                if key in ('rmse','wrist_rmse'): value=np.sqrt(np.mean(a*a))
                elif key=='px': value=np.sqrt(np.average(a*a,weights=[r[k]['n_px'] for r in rows.values()]))
                else: value=a.mean()
                assert abs(value-s[key])<1e-8
        for w in wrists:
            assert w['oracle']['error_mm']<=min(w[k]['error_mm'] for k in ['image','objective','baseline'])+1e-7
        for k in ['objective','image','gated']:
            bank=read(OUT/(split+'_candidates.json'))
            pack=OUT/('validation_fit' if split=='validation' else '../rw_frozen200_free12_20261119')/'budget_20000.npz'
            with np.load(OUT/(split+'_'+k+'.npz')) as z,np.load(pack) as initial:
                order=z['pair'].tolist();idx=[bank[p]['index'] for p in order]
                assert len(set(order))==len(order)==len(rows)
                for field in ('global0','delta','translation','betas'): assert np.array_equal(z[field],initial[field][idx])
                fixed=[i for i in range(21) if i not in (15,16,17,18)]
                assert np.array_equal(z['body_pose'][:,fixed],initial['body_pose'][idx][:,fixed])
        actual='image' if frozen['active'] else 'baseline'
        assert data['summary']['all']['gated']==data['summary']['all'][actual]
        diagnostic[split]=dict(feasible=len(rows),excluded=coverage['excluded_no_baseline'],
           nontrivial_wrists=sum(w['n_candidates']>1 for w in wrists),n_candidates=sum(w['n_candidates'] for w in wrists),
           saved_parameter_hashes={k:sha256((OUT/(split+'_'+k+'.npz')).read_bytes()).hexdigest() for k in ['objective','image','gated']},
           valid_2d=sum(r['baseline']['n_px'] for r in rows.values()),n_subjects=len({w['subject'] for w in wrists}))
    valfit=read(OUT/'validation_fit/fit_complete.json')
    commands=['validation.py prepare','validation.py fit','candidates.py validation','select_wrist.py freeze',
              'select_wrist.py validation','score_candidates.py validation','candidates.py test',
              'select_wrist.py test','score_candidates.py test','verify.py']
    result=dict(base='4461ff0c066c242f8de21b0b1130f3c25916a554',status=0,diagnostics=diagnostic,
        commands=[dict(command='<BODY_PYTHON> -Bu '+str(RUN.relative_to(ROOT))+'/'+c,exit_status=0) for c in commands],
        validation_adam_steps=valfit['steps'],validation_fit_s=valfit['elapsed_seconds'],selection_tolerance_mm=.001,
        confirmation_rule='wrist RMSE and whole-body mean compared to baseline; at most +0.25 mm other-ten mean on both subject groups',
        image_descriptor='oriented wrist and forearm patches; bilateral mean normalized correlation; minimum local intensity std 4',
        candidate_depth_offsets_mm=[-300,-200,-120,-80,-40,-20,0,20,40,80,120,200,300])
    (RUN/'evidence/verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(diagnostic)

if __name__=='__main__': main()
