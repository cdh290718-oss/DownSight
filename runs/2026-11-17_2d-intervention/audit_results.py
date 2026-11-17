"""Final input, asset and paired-output consistency checks, without new fitting."""
import sys

import numpy as np

from common import EVIDENCE, OUT, ROOT, RUN, camera_files, read, sha, write


def main():
    cfg=read(RUN/'protocol.json');prep=read(OUT/'prepared.json')
    assert sha(OUT/'prepared.json')==read(EVIDENCE/'preparation.json')['prepared_local_sha256']
    assert sha(RUN/'protocol.json')==prep['protocol_sha256']
    resources=read(EVIDENCE/'input_hashes.json')['resources']
    for r in resources:
        assert sha(ROOT.parent/r['resource_id'])==r['sha256'],r['resource_id']
    # When the baseline has no accepted triangulated targets, the two methods
    # must yield identical complete trajectories and selected parameters.
    complete=read(OUT/'fit_complete.json');ids=[tuple(x) for x in complete['ids']]
    no_route=[i['pair'] for i in prep['items'] if i['condition']=='baseline' and not any(p['accepted'] for p in i['points'])]
    equivalent=[]
    for budget in cfg['budgets']:
        with np.load(OUT/('budget_'+str(budget)+'.npz')) as data:
            for pair in no_route:
                for condition in cfg['conditions']:
                    a,b=ids.index((pair,condition,'2d')),ids.index((pair,condition,'fixed_mixed'))
                    errors={k:float(np.max(np.abs(data[k][a]-data[k][b]))) for k in data.files}
                    assert max(errors.values())==0,errors
                    equivalent.append(dict(pair=pair,condition=condition,budget=budget,max_difference=max(errors.values())))
    sys.path.insert(0,str(ROOT/'runs/2026-11-04_session-hand6-v2/code'))
    from prepare import Camera,triangulate
    cams=[Camera(p) for p in camera_files()]
    diagnostic=[]
    for item in prep['items']:
        transform=np.array(item['inputs']['views']['right']['camera']['left_to_view'])
        for p in item['points']:
            if not p['accepted']:continue
            q=triangulate(cams,transform[:3,:3],transform[:3,3],np.array(p['xy']))
            diagnostic.append(dict(pair=item['pair'],condition=item['condition'],joint=p['name'],
                                   recomputed_geometry_accepted=bool(q['accepted']),
                                   reason=q.get('reason'),ray_gap_m=q.get('ray_gap_m'),
                                   baseline_covariance_kept=True))
    output=dict(command='<BODY_PYTHON> runs/2026-11-17_2d-intervention/audit_results.py',exit_status=0,
                protocol_and_prepared_unchanged=True,external_resource_hashes_unchanged=True,
                zero_3d_route_method_equivalence=equivalent,
                fixed_route_recomputed_geometry=diagnostic,
                geometry_gate_failures=sum(not q['recomputed_geometry_accepted'] for q in diagnostic))
    write(EVIDENCE/'consistency.json',output)
    print('Consistency audit passed; recomputed geometry gate failures:',output['geometry_gate_failures'])


if __name__=='__main__':
    main()
