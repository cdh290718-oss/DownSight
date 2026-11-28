"""Score only the twelve body joints after both Head-constrained fits are sealed."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
OUT = ROOT / 'outputs/rw12_body_head_anchors_20261128'
CONTROL = ROOT / 'outputs/rw12_head_contour_only_20261128'
METHODS = ('unanchored', 'mount_nose', 'body_contour')
sys.path.insert(0, str(ROOT / 'runs/2026-11-20_rw20000_floor12'))
from render_gallery import g

read, write, sha = g.read, g.write, g.sha


def score():
    torch.set_num_threads(1)
    model, _ = g.common.setup_model()
    layer = model.layer['neutral'].cpu().eval()
    sealed = read(OUT / 'sealed.json')
    fit = read(OUT / 'fit_complete.json')
    assert sha(OUT / 'anchored_body.npz') == sealed['sha256']
    assert sha(OUT / 'fit_complete.json') == sealed['fit_complete_sha256']
    prepared = read(OUT / 'prepared.json')['items']
    inputs = read(RUN / 'evidence/inputs.json')
    assert sha(OUT / 'prepared.json') == inputs['prepared_sha256']
    assert sha(OUT / 'protocol.json') == inputs['protocol_sha256']
    diagnostic = read(OUT / 'diagnostic_20000.json')
    assert fit['ids'] == diagnostic['ids'] and all(diagnostic['selected_feasible'])
    assert len(fit['ids']) == 24
    with np.load(OUT / 'anchored_body.npz') as archive:
        constrained = dict(archive)
    with np.load(CONTROL / 'budget_20000.npz') as archive:
        control = dict(archive)
    control_fit = read(CONTROL / 'fit_complete.json')
    assert fit['steps'] == control_fit['steps'] == 20000 and control_fit['exit_status'] == 0
    assert not constrained['betas'].any() and not control['betas'].any()
    previous = ROOT / 'runs/2026-11-28_rw12_head_contour_only/evidence/results.json'
    archive = read(previous)
    assert sha(CONTROL / 'budget_20000.npz') == archive['head_fit_input_sha256']
    old = {row['pair']: row for row in archive['rows']}
    control_by_pair = {x[0]: i for i, x in enumerate(control_fit['ids'])}
    reference_path = g.DATA / 'reference.json'
    receipt = read(ROOT / 'runs/2026-11-18_rw_frozen200_golden_d/evidence/preparation.json')
    assert sha(reference_path) == receipt['reference_local_sha256']
    reference = read(reference_path)
    obs = g.common.OBS
    ids = np.asarray(g.common.NATIVE_IDS)[obs]
    rows = []

    def reconstruct(pack, index):
        params = dict(body_pose=pack['body_pose'][index], betas=pack['betas'][index],
                      global_orient=pack['global0'][index], transl=pack['translation'][index])
        return g.geometry(layer, params, pack['delta'][index], False, None)[1]

    def metrics(native, gt):
        predicted = native[ids]
        errors = np.linalg.norm(predicted - gt, axis=1) * 1000
        aligned = np.linalg.norm(g.align(predicted, gt) - gt, axis=1) * 1000
        return dict(pa_mm=float(aligned.mean()), nonpa_mm=float(errors.mean()),
                    nonpa_joint_mm=errors.tolist(), pa_joint_mm=aligned.tolist())

    for n, pair in enumerate(inputs['rows']):
        control_index = control_by_pair[pair]
        truth = np.asarray(reference[pair]['gt'])[obs]
        assert reference[pair]['subject'] == old[pair]['subject']
        native = reconstruct(control, control_index)
        assert np.linalg.norm(native[15] - old[pair]['initial_head_m']) < 1e-6
        entry = dict(pair=pair, subject=reference[pair]['subject'], contour_eyes=old[pair]['observations'],
                     unanchored=metrics(native, truth))
        for index, method in zip((2*n, 2*n+1), METHODS[1:]):
            assert prepared[index]['pair'] == pair == fit['ids'][index][0]
            assert prepared[2*n]['points'] == prepared[2*n+1]['points']
            assert np.array_equal(constrained['global0'][index], control['global0'][control_index])
            native = reconstruct(constrained, index)
            error = np.linalg.norm(native[15] - inputs['rows'][pair][method]['position_m']) * 1000
            assert error < .001
            entry[method] = dict(metrics(native, truth), anchor_error_mm=float(error),
                                 selected_step=int(constrained['selected_step'][index]))
        rows.append(entry)

    assert len(rows) == 12
    summary = {}
    for key, group in [('all', rows), ('stereo_contour', [r for r in rows if r['contour_eyes'] == 2]),
                       ('mono_contour', [r for r in rows if r['contour_eyes'] == 1])]:
        summary[key] = {'pairs': len(group), 'metrics': {}}
        for metric in ('pa_mm', 'nonpa_mm'):
            v = np.array([[r[name][metric] for name in METHODS] for r in group])
            diff = v[:, 2] - v[:, 1]
            summary[key]['metrics'][metric] = dict(means=dict(zip(METHODS, v.mean(axis=0).tolist())),
                contour_minus_mount=float(diff.mean()), contour_better=int((diff < -.01).sum()),
                contour_worse=int((diff > .01).sum()), tied=int((np.abs(diff) <= .01).sum()))
    max_residual = max(r[m]['anchor_error_mm'] for r in rows for m in METHODS[1:])
    report = dict(base_commit='206f23c453ca2d26052f2062ac4272a392381bba', dirty_worktree=True,
                  optimizer_steps=fit['steps'], fit_inputs_sha256=sha(RUN / 'evidence/inputs.json'),
                  body_truth_sha256=sha(reference_path), archived_body_control_sha256=sha(CONTROL / 'budget_20000.npz'),
                  fit_sha256=sealed['sha256'], max_anchor_error_mm=max_residual,
                  ordered_joints=[g.common.NAMES[i] for i in obs], summary=summary, rows=rows, exit_status=0)
    write(RUN / 'evidence/results.json', report)
    render(rows)
    print(summary['all'], flush=True)


def render(rows):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.5), sharey=True, layout='constrained')
    y = np.arange(len(rows))
    handles = []
    for axis, metric, title in zip(axes, ('pa_mm', 'nonpa_mm'), ('PA-MPJPE', 'MPJPE')):
        for n, (method, color, marker, label) in enumerate([
            ('unanchored', '#8a95a5', '|', 'Body points only'),
            ('mount_nose', '#007c78', 'o', 'Head from mount + nose'),
            ('body_contour', '#b55121', 's', 'Head from body + outline'),
        ]):
            mark = axis.scatter([r[method][metric] for r in rows], y + (n - 1) * .16,
                                c=color, marker=marker, s=46, label=label)
            if axis is axes[0]:
                handles.append(mark)
        axis.set_yticks(y, [r['pair'] for r in rows])
        axis.set_xlabel(f'{title} (mm)')
        axis.grid(axis='x', color='#ebeff2')
        axis.spines[['right', 'top']].set_visible(False)
    axes[0].invert_yaxis()
    axes[1].legend(handles=handles, loc='best', frameon=False, fontsize=8)
    fig.savefig(RUN / 'assets/body_errors.png', dpi=160)


if __name__ == '__main__':
    score()
