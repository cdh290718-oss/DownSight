"""Score sealed head poses against the official Head joint in the left camera."""
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from body_start import ROOT, RUN, OUT, PREP, read, sha, write

TRUTH = ROOT.parent / 'EgoSMPLX/data/UnrealEgo_RW_Test/prepared/samples.npz'
STAGES = ('mount', 'initial', 'final')


def score():
    inputs = read(RUN / 'evidence/inputs.json')
    receipts = {}
    for pair in inputs['rows']:
        folder = OUT / 'poses'
        result = read(folder / f'{pair}_head.json')
        assert result['pair'] == pair and result['exit_status'] == 0
        assert sha(folder / f'{pair}_body.npz') == result['body_pose_sha256']
        receipts[pair] = result
    assert len(receipts) == 12
    # Read official 3D positions only after checking all fitted outputs.
    archive = read(PREP)
    samples = {x['pair']: x for x in archive['pairs']}
    assert sha(TRUTH) == archive['dataset_sha256']
    with np.load(TRUTH) as data:
        heads = data['keypoints3d_fit'][:, 0]
        sequence, frames, sides = (data[k] for k in ('source_sequence', 'frame_index', 'camera_side'))
    for pair, row in receipts.items():
        sample = samples[pair]
        index = sample['source_indices'][0]
        assert (sequence[index], frames[index], sides[index]) == (sample['sequence'], sample['frame'], 'left')
        truth = heads[index]
        row['subject'] = sample['subject']
        row['official_head_m'] = truth.tolist()
        row['errors'] = {}
        for stage, field in zip(STAGES, ('mount_cam3_m', 'initial_head_m', 'final_head_m')):
            difference = (np.asarray(row[field]) - truth) * 1000
            row['errors'][stage] = dict(delta_xyz_mm=difference.tolist(), norm_mm=float(np.linalg.norm(difference)))
        row['head_movement_mm'] = float(np.linalg.norm(np.asarray(row['final_head_m']) - row['initial_head_m']) * 1000)

    grouped = {
        'all': list(receipts.values()),
        'nose_and_outline': [r for r in receipts.values() if r['method'] == 'nose_and_outline'],
        'nose_only': [r for r in receipts.values() if r['method'] == 'nose_only'],
    }
    summary = {}
    for method, group in grouped.items():
        errors = {stage: np.array([r['errors'][stage]['norm_mm'] for r in group]) for stage in STAGES}
        changes = errors['final'] - errors['initial']
        summary[method] = dict(
            pairs=len(group), mean_mm={k: float(v.mean()) for k, v in errors.items()},
            median_mm={k: float(np.median(v)) for k, v in errors.items()},
            max_mm={k: float(v.max()) for k, v in errors.items()},
            improved=int((changes < -.01).sum()), worsened=int((changes > .01).sum()),
            tied=int((np.abs(changes) <= .01).sum()),
            mean_head_movement_mm=statistics.mean(r['head_movement_mm'] for r in group),
            mean_nose_rmse_before_px=statistics.mean(r['nose_rmse_before_px'] for r in group),
            mean_nose_rmse_after_px=statistics.mean(r['nose_rmse_after_px'] for r in group))
    fit = read(OUT / 'fit/0/fit_complete.json')
    assert fit['steps'] == 20000 and fit['exit_status'] == 0
    write(RUN / 'evidence/results.json', dict(
        base_commit='8459affaffbdee353511042b95305fb4acd5de8c', dirty_worktree=True,
        reference_sha256=archive['dataset_sha256'], inputs_sha256=sha(RUN / 'evidence/inputs.json'),
        source_sha256={p.name: sha(p) for p in RUN.glob('*.py')}, fit_steps=fit['steps'],
        rows=list(receipts.values()), summary=summary, exit_status=0))
    chart(list(receipts.values()))
    print(summary, flush=True)


def chart(rows):
    fig, ax = plt.subplots(figsize=(8.5, 5.3), layout='constrained')
    y = np.arange(len(rows))
    starts = [r['errors']['initial']['norm_mm'] for r in rows]
    ends = [r['errors']['final']['norm_mm'] for r in rows]
    ax.hlines(y, np.minimum(starts, ends), np.maximum(starts, ends), color='#8a94a4', linewidth=1.2)
    ax.scatter(starts, y, facecolors='white', edgecolors='#415169', label='Before head fit', zorder=3)
    for method, marker, color, label in [
        ('nose_and_outline', 'o', '#007f7b', 'Nose + outline'), ('nose_only', 's', '#b65824', 'Nose only')
    ]:
        idx = [i for i, r in enumerate(rows) if r['method'] == method]
        ax.scatter([ends[i] for i in idx], y[idx], marker=marker, c=color, label=label, zorder=4)
    ax.set_yticks(y, [r['pair'] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel('3D Head position error in left camera (mm)')
    ax.set_xlim(left=0)
    ax.grid(axis='x', color='#eceef1')
    ax.spines[['right', 'top']].set_visible(False)
    ax.legend(loc='best', frameon=False)
    fig.savefig(RUN / 'assets/head_position.png', dpi=160)


if __name__ == '__main__':
    score()
