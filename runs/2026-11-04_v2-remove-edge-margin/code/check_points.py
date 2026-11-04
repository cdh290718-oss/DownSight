"""Verify the fixed inputs after removing only the border exclusion."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from egosmplx_pipeline.body_common import JOINT_MAPPING
from egosmplx_pipeline.v2_policy import body_allowed

p = argparse.ArgumentParser()
p.add_argument('--inputs', type=Path, required=True)
p.add_argument('--previous-inputs', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
read = lambda path: json.loads(Path(path).read_text())
os.environ['EGOSMPLX_V2'] = '1'
old = {r['id']: r for r in read(a.previous_inputs)['records']}
summary = {r['frame']: r for r in read('runs/2026-11-04_session-hand6-v2/evidence/input_summary.json')['frames']}
records = []
for r in read(a.inputs)['records']:
    lookup = {q['name']: q for q in read(r['row']['predictions']['sapiens'])['keypoints']}
    selected = [n for n in JOINT_MAPPING if n in lookup
                and np.isfinite([lookup[n]['x'], lookup[n]['y'], lookup[n]['score']]).all()
                and lookup[n]['score'] >= .3 and body_allowed(n, lookup[n])]
    previous = summary[r['id']]
    assert selected == previous['body_original_valid']
    with np.load(r['original']['body']) as body:
        assert len(selected) == len(body['stage2_selected_joint_indices'])
    context = read(r['context']); original_context = read(old[r['id']]['context'])
    assert context['point_filter'] == 'in_image_no_margin'
    unchanged = all(context[key] == original_context[key] for key in
                    ['matches', 'hand_xy', 'hand_weights', 'tri', 'whiten', 'tri_weights'])
    assert unchanged
    records.append(dict(frame=r['id'],pair=r['pair'],camera=r['camera'],
                        original_points=previous['body_original_valid'],
                        previous_v2_points=previous['body_retained'],current_v2_points=selected,
                        restored_points=[n for n in selected if n not in previous['body_retained']],
                        existing_hand_observations_unchanged=unchanged,
                        point_filter=context['point_filter']))
result = dict(passed=True,full_fitting_rerun=False,frames=records,
              shoulders_before=sum(sum(n.endswith('_shoulder') for n in r['previous_v2_points']) for r in records),
              shoulders_after=sum(sum(n.endswith('_shoulder') for n in r['current_v2_points']) for r in records),
              body_points_before=sum(len(r['previous_v2_points']) for r in records),
              body_points_after=sum(len(r['current_v2_points']) for r in records))
a.out.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='frames'}))
