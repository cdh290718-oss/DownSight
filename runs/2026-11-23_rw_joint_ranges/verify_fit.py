"""Recover the paired decisions and compare fresh controls to archived summary metrics."""
import json
import numpy as np
from angles import EVIDENCE, ROOT, RUN, sha, write


def read(path):
    return json.loads(path.read_text())


def main():
    results = {}
    for split in ('validation','test'):
        result = read(EVIDENCE/(split+'_evaluation.json'))
        archive = read(ROOT/'runs/2026-11-23_stereo_wrist_candidates/evidence'/(split+'_results.json'))
        baseline = result['summary']['all']['baseline']
        metrics = {'mpjpe_mm':'mpjpe', 'rmse_mm':'rmse', 'pa_mm':'pa', 'px_rmse':'px'}
        previous = archive['summary']['all']['baseline']
        drift = {name:baseline[name]-previous[old] for name,old in metrics.items()}
        rows = list(result['rows'].values())
        assert len(rows) == (100 if split == 'test' else 200)
        assert set(result['rows']) == set(archive['rows'])
        shifts = [np.mean(r['baseline']['mm'])-archive['rows'][pair]['baseline']['mpjpe'] for pair,r in result['rows'].items()]
        diagnostics = {}
        for arm in ('baseline','rw_prior'):
            vals = np.array([r[arm]['mm'] for r in rows])
            mean = vals.mean()
            assert abs(mean-result['summary']['all'][arm]['mpjpe_mm'])<1e-9
            diagnostics[arm] = {str(j):m for j,m in zip(result['joint_order'],vals.mean(0).tolist())}
        results[split] = dict(archived_control_mean_differences=drift,
                              maximum_paired_control_drift_mm=float(np.max(np.abs(shifts))),
                              per_joint_mpjpe_mm=diagnostics)
    write(EVIDENCE/'fit_check.json',dict(command='bash run.sh verify_fit', exit_status=0,
        source_sha256={p.name:sha(p) for p in sorted(RUN.iterdir()) if p.suffix in ('.py','.sh')},
        results=results))


if __name__ == '__main__':
    main()
