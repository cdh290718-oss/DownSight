"""Recompute published aggregates and verify local execution fingerprints."""
import hashlib
import json
import math
import statistics
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parents[1]


def read(path):
    return json.loads(path.read_text())


def close(a, b):
    assert math.isfinite(a) and math.isfinite(b)
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10), (a, b)


def main():
    evidence = RUN / 'evidence'
    summary = read(evidence / '2d_summary.json')['summary']
    errors = read(evidence / '2d_errors.json')['records']
    groups = 0
    for version, datasets in summary.items():
        for dataset, expected in datasets.items():
            rows = [r for r in errors if r['version'] == version and r['dataset'] == dataset]
            values = [v for r in rows for v in r['per_joint_error_px'] if v is not None]
            assert len(rows) == expected['images']
            assert len(values) == expected['valid_points']
            assert all(math.isfinite(v) for v in values)
            close(statistics.mean(values), expected['mean_px'])
            close(100 * sum(v <= 20 for v in values) / len(values), expected['pck20_percent'])
            assert expected['failed_images'] == 0
            groups += 1
    for dataset in summary['original']:
        maps = [{r['id']: [v is None for v in r['per_joint_error_px']]
                 for r in errors if r['dataset'] == dataset and r['version'] == version}
                for version in summary]
        assert maps[0] == maps[1] == maps[2]

    results = read(evidence / 'results.json')
    body = read(evidence / 'summary.json')['body3d']
    assert len(results['records']) == 30
    for condition, expected in body.items():
        rows = [r for r in results['records'] if r['condition'] == condition]
        assert len(rows) == 10
        assert all(r['status'] == 'ok' and r['selected_feasible'] for r in rows)
        for field, aggregate in [('absolute_per_joint_mm', 'mpjpe_mm'), ('pa_per_joint_mm', 'pa_mpjpe_mm')]:
            close(statistics.mean(v for r in rows for v in r[field]), expected[aggregate])
        assert sum(r['convergence_pass'] for r in rows) == expected['convergence_pass']
    equivalence = max(r['body_joint_max_difference_m'] for r in results['full_model_body_joint_equivalence'])
    assert equivalence < 3e-6
    previous = ROOT / 'runs/2026-11-17_newdetector_d10'
    for name in ['fit.py', 'geometry.py']:
        assert (RUN / name).read_bytes() == (previous / name).read_bytes()
    for name, expected in read(evidence / 'source.json')['code_sha256'].items():
        assert hashlib.sha256((RUN / name).read_bytes()).hexdigest() == expected
    fit = read(evidence / 'fit_execution.json')
    assert fit['exit_status'] == 0 and not fit['completion']['body_gt_read']
    init = read(ROOT / 'outputs/rw_a20000_evaluation_20261118/d_fit/initialization.json')['initialization']
    assert len(init) == 10 and all(v['rank'] == 0 for v in init.values())
    result = dict(command='python code/verify_evidence.py', exit_status=0,
                  paired_2d_groups_recomputed=groups, successful_3d_records=30,
                  same_image_ids_and_validity_masks=True, source_fingerprints_match=True,
                  fit_and_geometry_match_previous=True, body_gt_not_read_during_fit=True,
                  fixed_shape_rank_zero_pairs=len(init), body_model_max_difference_m=equivalence,
                  convergence_pass_per_condition={k: v['convergence_pass'] for k, v in body.items()})
    (evidence / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
