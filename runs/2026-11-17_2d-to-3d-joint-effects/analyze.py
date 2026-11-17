"""Reanalyze published derived residuals; no models, raw data or GPU required."""
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
EVIDENCE = RUN / 'evidence'
SOURCES = {
    'paired': 'runs/2026-11-15_gt2d-vs-sapiens/evidence/results.json',
    'preparation': 'runs/2026-11-15_gt2d-vs-sapiens/evidence/preparation.json',
    'noise': 'runs/2026-11-16_unrealego-official-noise/evidence/results.json',
    'noise_prepare': 'runs/2026-11-16_unrealego-official-noise/prepare.py',
    'noise_protocol': 'runs/2026-11-16_unrealego-official-noise/protocol.json',
}


def read(key):
    return json.loads((ROOT / SOURCES[key]).read_text())


def write(name, value):
    (EVIDENCE / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sign_count(values, tolerance=1.0):
    values = tuple(values)
    return {
        'improved_gt_1mm': sum(v > tolerance for v in values),
        'changed_within_1mm': sum(abs(v) <= tolerance for v in values),
        'worsened_gt_1mm': sum(v < -tolerance for v in values),
        'strict_positive': sum(v > 0 for v in values),
        'strict_negative': sum(v < 0 for v in values),
    }


def main():
    EVIDENCE.mkdir(exist_ok=True)
    rows = read('paired')['records']
    prep = read('preparation')
    pairs = []
    source_rows = []
    for p in prep['pairs']:
        records = [r for r in rows if r['pair'] == p['pair']]
        if not p['matched']:
            assert len(records) == 1 and records[0]['status'] == 'not_run'
            continue
        by_mode = {r['mode']: r for r in records}
        assert set(by_mode) == {'auto', 'official'}
        auto, official = by_mode['auto'], by_mode['official']
        assert all(r['status'] == 'ok' and r['exit_status'] == 0 for r in records)
        names = list(auto['absolute15_per_joint_mm'])
        assert len(names) == 15 and names == list(official['absolute15_per_joint_mm'])
        for r in records:
            for metric, mean_key in [('absolute15_per_joint_mm', 'absolute15_mean_mm'),
                                     ('pa15_per_joint_mm', 'pa15_mean_mm')]:
                assert math.isclose(statistics.mean(r[metric].values()), r[mean_key], abs_tol=1e-9)
            source_rows.append({k: r[k] for k in [
                'pair', 'mode', 'status', 'exit_status', 'absolute15_mean_mm',
                'absolute15_per_joint_mm', 'pa15_mean_mm', 'pa15_per_joint_mm',
                'convergence', 'shape_rank', 'final_sha256']})
        delta = {j: auto['absolute15_per_joint_mm'][j] - official['absolute15_per_joint_mm'][j]
                 for j in names}
        pairs.append({
            'pair': p['pair'], 'replaced_2d_joints': p['names'],
            'automatic_absolute_mean_mm': auto['absolute15_mean_mm'],
            'official_absolute_mean_mm': official['absolute15_mean_mm'],
            'absolute_delta_mm': delta, 'counts': sign_count(delta.values()),
            'both_convergence_pass': all(r['convergence']['diagnostic_pass'] for r in records),
        })
    assert len(pairs) == 3
    groups = {'Neck': ['Neck']}
    groups.update({part: [f'L_{part}', f'R_{part}']
                   for part in ['Shoulder', 'Elbow', 'Wrist', 'Hip', 'Knee', 'Ankle', 'Toe']})
    index = {(r['pair'], r['mode']): r for r in source_rows}
    grouped = []
    joints = []
    for label, members in list(groups.items()) + [(j, [j]) for j in names]:
        result = {'name': label, 'members': members}
        for metric in ['absolute15_per_joint_mm', 'pa15_per_joint_mm']:
            values = {mode: [statistics.mean(index[p['pair'], mode][metric][j] for j in members)
                             for p in pairs] for mode in ['auto', 'official']}
            delta = [a-b for a, b in zip(values['auto'], values['official'])]
            result[metric] = {
                'automatic_mean_mm': statistics.mean(values['auto']),
                'official_mean_mm': statistics.mean(values['official']),
                'improvement_mm': statistics.mean(delta),
                'paired_deltas_mm': delta, 'counts': sign_count(delta),
            }
        (grouped if len(grouped) < len(groups) else joints).append(result)
    write('paired_source_metrics.json', {'source': SOURCES['paired'], 'records': source_rows,
                                        'preparation': prep, 'not_run': rows[-1]})
    write('paired_analysis.json', {
        'delta_definition': 'prediction-2D fit error minus official-2D fit error; positive is improvement',
        'tolerance_note': '1mm is a descriptive threshold, not a significance test or equivalence margin',
        'pair_order': [p['pair'] for p in pairs], 'pairs': pairs,
        'per_group': grouped, 'per_joint': joints,
        'overall_absolute_automatic_mm': statistics.mean(p['automatic_absolute_mean_mm'] for p in pairs),
        'overall_absolute_official_mm': statistics.mean(p['official_absolute_mean_mm'] for p in pairs),
        'pooled_joint_sample_counts': sign_count(v for p in pairs for v in p['absolute_delta_mm'].values()),
    })

    # Descriptive noise-level comparisons only. Sigma is included in SeedSequence,
    # so a shared seed label does not mean a shared noise direction across sigmas.
    noise = read('noise')['records']
    assert len(noise) == 144 and all(r['exit_status'] == 0 for r in noise)
    noise_summary = []
    for method in ['2d', '3d', 'mixed']:
        levels = []
        for sigma in [1, 3, 5, 8]:
            chosen = [r for r in noise if r['method'] == method and
                      re.fullmatch(rf'gt_s{sigma}_seed\d+', r['condition'])]
            assert len(chosen) == 12
            per_joint = {j: statistics.mean(r['absolute_per_joint_mm'][i] for r in chosen)
                         for i, j in enumerate(names)}
            assert math.isclose(statistics.mean(per_joint.values()),
                                statistics.mean(r['absolute15_mm'] for r in chosen), abs_tol=1e-9)
            levels.append({'sigma_px': sigma, 'records': len(chosen),
                           'absolute_mean_mm': statistics.mean(per_joint.values()),
                           'absolute_per_joint_mm': per_joint,
                           'convergence_pass': sum(r['convergence']['diagnostic_pass'] for r in chosen)})
        delta = {j: levels[-1]['absolute_per_joint_mm'][j] - levels[0]['absolute_per_joint_mm'][j]
                 for j in names}
        noise_summary.append({'method': method, 'levels': levels,
                              'sigma8_minus_sigma1_per_joint_mm': delta,
                              'sigma8_to_sigma1_counts': sign_count(delta.values())})
    write('noise_descriptive.json', {
        'source': SOURCES['noise'], 'summary': noise_summary,
        'not_single_factor': [
            'Noise RNG SeedSequence includes sigma; directions differ across noise levels.',
            'Assumed observation sigma, covariances and 3D gates change with noise level.',
            'Shape subspace is derived from each condition and can change.',
            '2D method also uses condition-dependent sigma and shape subspace.',
            'Four poses of one subject and repeated noise seeds are not independent subjects.',
        ],
    })

    # Exact elementary counterexample: orthographic 2D observations of two 3D
    # joints (t,0,1), (t,1,1); t is their only unknown shared translation.
    truth_t = 0.0
    old_x, better_x = [2.0, -2.0], [0.0, -1.0]
    old_t, better_t = statistics.mean(old_x), statistics.mean(better_x)
    assert all(abs(b) < abs(a) for a, b in zip(old_x, better_x))
    assert abs(better_t-truth_t) > abs(old_t-truth_t)
    write('mathematical_counterexample.json', {
        'kind': 'mathematical construction, not a DownSight experiment',
        'model': '3D joints (t,0,1) and (t,1,1), orthographic projection (x,y), true t=0',
        'fit': 'minimize (t-u1)^2+(t-u2)^2; exact optimum t=(u1+u2)/2',
        'old_2d_x': old_x, 'better_2d_x': better_x,
        'old_2d_errors': list(map(abs, old_x)), 'better_2d_errors': list(map(abs, better_x)),
        'old_fitted_t': old_t, 'better_fitted_t': better_t,
        'old_3d_errors': [abs(old_t-truth_t)]*2,
        'better_3d_errors': [abs(better_t-truth_t)]*2,
        'units': 'arbitrary consistent coordinate units; no measurement claims',
    })
    write('execution.json', {
        'command': 'python runs/2026-11-17_2d-to-3d-joint-effects/analyze.py',
        'exit_status': 0, 'baseline_commit': 'f10ed89d875715baa069e2b9f488b41aa381fcb6',
        'new_fit_or_training': False,
        'source_sha256': {key: {'path': path, 'sha256': hashlib.sha256((ROOT/path).read_bytes()).hexdigest()}
                          for key, path in SOURCES.items()},
        'checks': ['paired completeness and successful exit', 'per-joint means match published means',
                   'all noise conditions complete', 'noise mean mapping matches published mean',
                   'counterexample improves both 2D inputs and worsens both exact-optimum 3D outputs'],
    })
    for row in grouped:
        q = row['absolute15_per_joint_mm']
        print(row['name'], *(round(q[k], 3) for k in ['automatic_mean_mm', 'official_mean_mm', 'improvement_mm']))
    print('noise 8 to 1', [(r['method'], r['sigma8_to_sigma1_counts']) for r in noise_summary])


if __name__ == '__main__':
    main()
