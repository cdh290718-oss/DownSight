import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
RUN = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'runs/2026-11-20_rw20000_floor12'))
sys.path.insert(0, str(ROOT / 'runs/2026-11-25_frozen100_fingers'))
from render_gallery import g
from geometry_inputs import prev, search


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(joints, truth):
    pred = joints[g.common.NATIVE_IDS][g.common.OBS]
    target = np.asarray(truth['gt'])[g.common.OBS]
    return {
        'pa_mm': float(np.linalg.norm(g.align(pred, target) - target, axis=1).mean() * 1000),
        'nonpa_mm': float(np.linalg.norm(pred - target, axis=1).mean() * 1000),
    }


def main():
    torch.set_num_threads(1)
    model, _ = g.common.setup_model()
    layer = model.layer['neutral'].cpu().eval()
    source_path = ROOT / 'runs/2026-11-28_fixed_skeleton_shape/evidence/results.json'
    source = g.read(source_path)
    gallery = g.read(ROOT / 'runs/2026-11-23_fixed_floor24/evidence/gallery.json')['rows']
    assert [x['pair'] for x in source['rows']] == [x['pair'] for x in gallery]
    rows = []
    hashes = {'previous_results': digest(source_path)}
    for split in ('test', 'validation'):
        bank_path = search.OUT / (split + '_bank.npz')
        choices_path = search.OUT / (split + '_choices.json')
        truth_path = g.DATA / 'reference.json' if split == 'test' else ROOT / 'outputs/stereo_wrist_candidates_20261123/validation_truth.json'
        audit = g.read(search.RUN / ('evidence/' + split + '_search.json'))
        assert digest(bank_path) == audit['parameter_sha256']
        assert digest(choices_path) == audit['choices_sha256']
        hashes[split + '_bank'] = digest(bank_path)
        hashes[split + '_choices'] = digest(choices_path)
        hashes[split + '_truth'] = digest(truth_path)
        choices = g.read(choices_path)
        truth = g.read(truth_path)
        with np.load(bank_path) as bank:
            for row in source['rows']:
                if row['split'] != split:
                    continue
                pair = row['pair']
                record = choices[pair]
                index = record['offset'] + record['fixed']
                original = {
                    'global_orient': bank['global0'][index],
                    'body_pose': bank['body_pose'][index],
                    'transl': bank['translation'][index],
                    'betas': bank['betas'][index],
                }
                delta = bank['delta'][index]
                assert not original['betas'].any()
                assert row['subject'] == truth[pair]['subject']
                updated = dict(original, betas=np.asarray(row['coefficients'], dtype=np.float32))
                _, before_joints, _, _ = g.geometry(layer, original, delta, False, None)
                _, after_joints, _, _ = g.geometry(layer, updated, delta, False, None)
                before = score(before_joints, truth[pair])
                after = score(after_joints, truth[pair])
                assert abs(before['pa_mm'] - row['pa_mm']) < 0.001
                assert abs(before['nonpa_mm'] - row['nonpa_mm']) < 0.001
                ids = np.asarray(g.common.NATIVE_IDS)[g.common.OBS]
                movement = np.linalg.norm(after_joints[ids] - before_joints[ids], axis=1) * 1000
                rows.append({'pair': pair, 'subject': row['subject'], 'split': split,
                             'baseline': before, 'standard_smplx': after,
                             'mean_joint_shift_mm': float(movement.mean()),
                             'max_joint_shift_mm': float(movement.max())})
    assert len(rows) == 24
    summary = {}
    for key in ('pa_mm', 'nonpa_mm'):
        baseline = np.array([r['baseline'][key] for r in rows])
        updated = np.array([r['standard_smplx'][key] for r in rows])
        change = updated - baseline
        summary[key] = {'baseline_mean': float(baseline.mean()),
                        'standard_mean': float(updated.mean()),
                        'mean_change': float(change.mean()),
                        'improved': int((change < -0.01).sum()),
                        'unchanged': int((np.abs(change) <= 0.01).sum()),
                        'worsened': int((change > 0.01).sum())}
    summary['mean_joint_shift_mm'] = float(np.mean([r['mean_joint_shift_mm'] for r in rows]))
    summary['max_joint_shift_mm'] = float(max(r['max_joint_shift_mm'] for r in rows))
    result = {'base_commit': '92f6b76ba23e68cc2f700bec37d8228def5cc70e',
              'dirty_worktree': True, 'command': 'bash <RUN>/run.sh',
              'input_sha256': hashes, 'summary': summary, 'rows': rows, 'exit_status': 0}
    (RUN / 'evidence/results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
