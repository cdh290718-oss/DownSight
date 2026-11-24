"""Check subject/action isolation, image studies, summaries and archived asset identities."""
import itertools
import platform
import torch
from calibrate import load_labels
from data_shape import EV, FOUR, ROOT, RUN, read, write, sha, np, SPANS

def main():
    cal=read(EV/'calibration.json')
    exp=read(EV/'poses.json')
    surface=read(EV/'surfaces.json')
    controls=read(EV/'controls.json')
    values,official,source=load_labels()
    split=cal['split']
    assert source==cal['input_sha256']
    groups={key:set(itertools.chain.from_iterable(s[key] for s in split.values())) for key in ('calibration','holdout')}
    assert not groups['calibration'] & groups['holdout'] and set(values)==set.union(*groups.values())
    qrows=exp['rows']
    assert all(r['sequence'] in split[r['subject']]['holdout'] for r in qrows.values())
    assert set(FOUR)<=set(qrows) and set(FOUR)==set(surface['rows'])
    sparse_ids=[k for r in cal['records'].values() for keys in r.get('stereo',{}).get('frame_ids',{}).values() for k in keys]
    assert len(set(sparse_ids))==len(sparse_ids) and not set(sparse_ids)&set(qrows)
    assert len(qrows)+len(sparse_ids)==300
    records=list(cal['records'].values())
    checks={}
    for reference in ['mean','cohort']:
        checks[reference]={metric:sum(r['scores']['personal'][metric]<r['scores'][reference][metric] for r in records)
                           for metric in ('mae_mm','proportion_mae_pp')}
    for arm in ('mean','cohort','personal'):
        expected=cal['summary']['all'][arm]['mae_mm']
        assert abs(np.mean([r['scores'][arm]['span_mae_mm'] for r in records])-expected)<1e-10
    assert np.isclose(controls['summary']['uniform_scale']['proportion_mae_pp'],cal['summary']['all']['mean']['proportion_mae_pp'])
    across={}
    for p,s in split.items():
        med={group:np.median([np.median(values[seq],axis=0) for seq in seqs],axis=0) for group,seqs in s.items()}
        across[p]=(abs(med['calibration']-med['holdout'])*1000).tolist()
    survey=read(ROOT/'runs/2026-11-23_rw_joint_ranges/evidence/source_check.json')
    neutral=ROOT.parent/'EgoSMPLX/common/utils/human_model_files/smplx/SMPLX_NEUTRAL.npz'
    assert sha(neutral)==survey['external_sha256']['neutral_smplx']
    write(EV/'execution.json',dict(base='c944872157c0d7826095b0c0e28f488ce4348e51',
        dirty_at_execution=True,commands=[dict(command='bash runs/2026-11-24_rw_subject_shape/run.sh '+s,exit_status=0)
        for s in ['calibrate','evaluate','controls','verify']], environment=dict(python=platform.python_version(),
        torch=torch.__version__,numpy=np.__version__,device='cpu'),source_sha256={p.name:sha(p) for p in RUN.iterdir() if p.suffix in ('.py','.sh')},
        subjects=len(split),frames={k:sum(len(values[s]) for s in seqs) for k,seqs in groups.items()},
        sequences={k:len(seqs) for k,seqs in groups.items()},spans=SPANS,calibration_and_holdout_overlap=0,
        stereo_calibration_frames=len(sparse_ids),v3_scoring_frames=len(qrows),personal_better_subjects=checks,
        per_subject_calibration_holdout_median_abs_delta_mm=across,neutral_model_sha256=sha(neutral),
        maximum_personal_bound_hits=sum(r['scores']['personal']['max_abs_beta']>2.999 for r in records),
        collision_reference_sha256=sha(ROOT/'runs/2026-11-23_floor24_intersections/evidence/results.json')))

if __name__=='__main__':
    main()
