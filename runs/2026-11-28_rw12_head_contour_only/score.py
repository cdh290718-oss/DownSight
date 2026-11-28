"""Assess sealed contour fits in the left-camera coordinate system."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from body_only import OUT, RUN, REFERENCE, read, sha, write


def evaluate():
    inputs = read(RUN / 'evidence/inputs.json')
    poses = [read(OUT / 'poses' / (key + '.json')) for key in inputs['rows']]
    assert all(r['exit_status'] == 0 for r in poses) and len(poses) == 12
    previous_file = REFERENCE / 'evidence/results.json'
    previous = {r['pair']: r for r in read(previous_file)['rows']}
    rows = []
    for pose in poses:
        pair = pose['pair']
        old = previous[pair]
        assert pose['subject'] == old['subject']
        truth = np.asarray(old['official_head_m'])
        initial = np.asarray(pose['initial_head_m'])
        final = np.asarray(pose['final_head_m'])
        before, after = (float(np.linalg.norm(p - truth) * 1000) for p in (initial, final))
        good = [np.asarray(x['transform']) for x in pose['search_finals']
                if x['loss'] <= pose['contour_after_px'] + 1]
        ref_rot = Rotation.from_rotvec(np.deg2rad(pose['transform_delta'][:3]))
        spread = [float(np.rad2deg((Rotation.from_rotvec(np.deg2rad(x[:3])) *
                                   ref_rot.inv()).magnitude())) for x in good]
        rows.append(dict(pose, official_head_m=truth.tolist(), previous_head_error_mm=old['errors']['final']['norm_mm'],
                         body_head_error_mm=before, contour_head_error_mm=after,
                         delta_xyz_mm=((final-truth)*1000).tolist(),
                         observations=sum(n >= 8 for n in pose['silhouette_points']),
                         near_minimum_angular_spread_deg=max(spread)))
    summary = {}
    for key, group in [('all', rows),
                       ('stereo', [r for r in rows if r['observations'] == 2]),
                       ('mono', [r for r in rows if r['observations'] == 1])]:
        error = np.array([r['contour_head_error_mm'] for r in group])
        initial = np.array([r['body_head_error_mm'] for r in group])
        changes = error - initial
        summary[key] = dict(pairs=len(group), mean_final_mm=float(error.mean()),
                            median_final_mm=float(np.median(error)),
                            min_final_mm=float(error.min()), max_final_mm=float(error.max()),
                            mean_initial_mm=float(initial.mean()),
                            mean_previous_mm=float(np.mean([r['previous_head_error_mm'] for r in group])),
                            mean_contour_start_px=float(np.mean([r['contour_before_px'] for r in group])),
                            mean_contour_end_px=float(np.mean([r['contour_after_px'] for r in group])),
                            improved=int((changes < -.01).sum()), worsened=int((changes > .01).sum()),
                            tied=int((np.abs(changes) <= .01).sum()),
                            position_limit_hits=sum(r['search_limit_hit'] for r in group),
                            max_near_minimum_angular_spread_deg=max(r['near_minimum_angular_spread_deg'] for r in group))
    fit = read(OUT/'fit_complete.json')
    assert fit['exit_status'] == 0 and fit['steps'] == 20000
    evidence = dict(base_commit='9bb5130e748d31249fd873669afce70a14dfb21b',
                    dirty_worktree=True, fit_steps=fit['steps'],
                    head_fit_input_sha256=sha(OUT/'budget_20000.npz'), scoring_reference_sha256=sha(previous_file),
                    source_sha256={p.name: sha(p) for p in RUN.glob('*.py')},
                    position_limit_mm=180, outline_clip_px=128, rows=rows, summary=summary, exit_status=0)
    write(RUN / 'evidence/results.json', evidence)
    chart(rows)
    print(summary, flush=True)


def chart(rows):
    fig, ax = plt.subplots(figsize=(8.7, 5.4), layout='constrained')
    y = np.arange(len(rows))
    before = [r['body_head_error_mm'] for r in rows]
    after = [r['contour_head_error_mm'] for r in rows]
    old = [r['previous_head_error_mm'] for r in rows]
    ax.hlines(y, np.minimum(before, after), np.maximum(before, after), color='#aab5c4')
    ax.scatter(old, y, c='#747c88', marker='|', s=120, label='Previous mount-based fit')
    ax.scatter(before, y, facecolors='white', edgecolors='#32456c', label='Body keypoints only', zorder=3)
    for count, color, shape, name in [(2, '#027e7a', 'o', 'Two-eye contour'), (1, '#b65824', 's', 'One-eye contour')]:
        idx = [i for i,r in enumerate(rows) if r['observations'] == count]
        ax.scatter([after[i] for i in idx], y[idx], c=color, marker=shape, label=name, zorder=4)
    ax.set_yticks(y, [r['pair'] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel('3D Head position error in the left camera (mm)')
    ax.set_xlim(left=0)
    ax.grid(axis='x', color='#edf0f3')
    ax.spines[['right', 'top']].set_visible(False)
    ax.legend(frameon=False, loc='best')
    fig.savefig(RUN / 'assets/head_position.png', dpi=160)


if __name__ == '__main__':
    evaluate()
