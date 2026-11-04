"""Validate source, inputs and actual selected points for the no-margin rerun."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--previous-inputs', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
read = lambda path: json.loads(Path(path).read_text())
baseline = 'f3b411150bc4a2da893a7d859a92f8a26adccaea'
source = read(args.root/'batch_cpu/code_hashes.json')
for path, digest in source.items():
    assert hashlib.sha256(subprocess.check_output(['git','show',baseline+':'+path])).hexdigest() == digest
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
previous = {r['id']:r for r in read(args.previous_inputs)['records']}
rows = []
for r in read(args.root/'prepared/inputs.json')['records']:
    prior = previous[r['id']]; cfg = read(r['config']); oldcfg = read(prior['config'])
    exclude = {'paths','manifest','output'}
    assert {k:v for k,v in cfg.items() if k not in exclude} == {k:v for k,v in oldcfg.items() if k not in exclude}
    root = Path(cfg['output']); frame = root/'frames'/r['id']
    assert read(root/'run_state.json')['state'] == 'complete'
    context = read(r['context']); oldcontext = read(prior['context'])
    for key in ['matches','hand_xy','hand_weights','tri','whiten','tri_weights','peer_R','peer_t','peer_calibration']:
        assert context[key] == oldcontext[key], (r['id'],key)
    assert context['point_filter'] == 'in_image_no_margin'
    with np.load(frame/'fused/body_params.npz') as current, np.load(r['original']['body']) as original:
        shape_equal = np.array_equal(current['betas'], original['betas'])
        targets_equal = np.array_equal(current['stage2_selected_joint_indices'], original['stage2_selected_joint_indices'])
        pixels_equal = np.array_equal(current['stage2_target_joint_pixels'], original['stage2_target_joint_pixels'])
        assert shape_equal and targets_equal and pixels_equal
        count = len(current['stage2_selected_joint_indices'])
    preflight = read(root/'preflight.json'); earlier = read(Path(oldcfg['output'])/'preflight.json')
    assert preflight['resources_sha256'] == earlier['resources_sha256']
    assert preflight['backend']['source_tree']['tree_sha256'] == earlier['backend']['source_tree']['tree_sha256']
    provenance = [read(p) for p in (root/'provenance').glob('*.json')]
    assert provenance and all(q['state']=='complete' and q['pipeline_git']['commit']==baseline and q['pipeline_files_sha256']==source for q in provenance)
    rows.append(dict(frame=r['id'],body_target_count=count,original_body_target_indices_and_pixels_restored=True,
                     original_shape_unchanged=True,previous_hand_and_stereo_observations_unchanged=True,
                     backend_and_resource_hashes_unchanged=True,pipeline_source_matches_baseline=True,
                     provenance_dirty=[q['pipeline_git']['dirty'] for q in provenance]))
assert len(rows) == 10
args.out.write_text(json.dumps(dict(passed=True,baseline=baseline,numerical_source_unchanged=True,frames=rows),indent=2)+'\n')
print('PASS',len(rows),'completed views; source, targets, shape and resources verified')
