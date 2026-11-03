"""Check published analyses and result relationships without external assets."""
from pathlib import Path
import json,csv,math
import numpy as np
p=Path(__file__).resolve().parent;d=json.loads((p/'decisions.json').read_text());s=json.loads((p/'summary.json').read_text());look={(x['frame'],x['name']):x for x in d};assert len(look)==len(d);assert len(s['frames'])==20
coverage=list(csv.DictReader((p/'coverage.csv').open()));assert len(coverage)==400
assert sum(r['two_endpoint_proxy']=='True' and r['method']=='old' for r in coverage)==s['old_coverage'];assert sum(r['two_endpoint_proxy']=='True' and r['method']=='candidate' for r in coverage)==s['candidate_coverage']
for r in d:
 if r['stereo']:
  assert all(r['eligible']) and not r['reasons'] and r['xyz_cam2_m'] is not None
  cov=np.asarray(r['cov_cam2_m2']);assert np.isfinite(cov).all() and np.allclose(cov,cov.T,atol=1e-10) and np.linalg.eigvalsh(cov).min()>-1e-10
 if 'left_right_duplicate_ambiguous' in r['reasons']:assert r['recommended']=='discard' and not r['stereo']
assert look[270,'right_wrist']['recommended']=='discard';assert all(look[f,'nose']['recommended']=='out_of_scope' for f in s['frames'])
assert len(s['elbow_tests'])==s['elbow_pairs']
for item in s['elbow_tests']:
 for part in ['elbow','olecranon','cubital_fossa']:assert look[item['frame'],item['side']+'_'+part]['stereo']
print('PASS: 20 frames; unique observations; coverage counts; covariance PSD; collisions, right-wrist rejection and paired-test eligibility.')
