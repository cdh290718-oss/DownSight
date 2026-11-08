"""Independently check saved stages, fixed blocks, and allowed oracle input."""
import ast
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/synthetic_head_anchor_20261108'
OLD = ROOT / 'outputs/synthetic_original_20261108'

for path in RUN.glob('*.py'):
    ast.parse(path.read_text())
execution = json.loads((RUN / 'evidence/execution.json').read_text())
assert len(execution) == 2 and all(r['exit_status'] == 0 for r in execution)
checks = []
for side, fit in [('left', 'left'), ('right', 'right_cpu')]:
    inputs = json.loads((OUT / side / 'inputs.json').read_text())
    stages = json.loads((OUT / side / 'stages.json').read_text())
    assert len(stages) == 10
    with np.load(OLD / 'fits' / fit / 'frames' / side / 'raw/params.npz') as z:
        raw = {k: z[k].copy() for k in inputs['fixed_parameters']}
    last = None
    differences = []
    for label, row in stages.items():
        assert row['frozen_block_equal']
        assert row['regenerated_mesh_max_abs_m'] <= 1e-5
        assert row['regenerated_selected_pixel_max_abs_px'] <= .05
        assert row['selected_best_loss'] <= row['initial_loss'] + 2e-6
        with np.load(OUT / side / label / 'params.npz') as z:
            p = {k: z[k].copy() for k in ['body_pose', 'global_orient', 'transl'] + inputs['fixed_parameters']}
        assert all(np.array_equal(p[k], raw[k]) for k in inputs['fixed_parameters'])
        if label != 'anchor_only_rigid':
            assert 'nose' not in row['selected_points']
            assert all(n in row['selected_points'] for n in ['left_shoulder', 'right_shoulder'])
            assert not row['all_vertex_near_guard']
        if last is not None and label.startswith('cycle'):
            if label.endswith('_pose'):
                frozen_delta = max(float(np.max(np.abs(p[k] - last[k]))) for k in ['global_orient', 'transl'])
            else:
                frozen_delta = float(np.max(np.abs(p['body_pose'] - last['body_pose'])))
            assert frozen_delta == 0
            differences.append(dict(label=label, frozen_artifact_max_abs_difference=frozen_delta))
        if label != 'anchor_only_rigid':
            last = p
    with np.load(OLD / 'dataset/samples/000001' / (side + '_vertices.npz'), allow_pickle=False) as gt:
        anchor_diff = float(np.max(np.abs(gt['joints_camera_cv_m'][15] - inputs['head_target_camera_m'])))
    assert anchor_diff < 1e-7
    checks.append(dict(side=side, stage_count=len(stages), head_target_rounding_max_m=anchor_diff,
                       frozen_blocks=differences, shape_and_hands_bitwise_fixed=True,
                       all_reload_checks_pass=True, all_block_objectives_nonincreasing=True))
result = dict(exit_status=0, views=checks, command='<BODY_PYTHON> runs/2026-11-08_synthetic-head-anchor/verify.py',
              scope='Independent saved-artifact checks plus each stage regenerated through SMPL-X during fitting',
              source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in RUN.glob('*.py')})
(RUN / 'evidence/verification.json').write_text(json.dumps(result, indent=2) + '\n')
print('saved-stage verification PASS')
