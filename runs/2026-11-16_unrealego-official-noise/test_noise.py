"""Verify generated 2D observations without accessing body 3D labels."""
import json,re
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[2];run=Path(__file__).resolve().parent;out=root/'outputs/unrealego_official_noise_v2_20261116';checks=[]
for pair in range(1,5):
 clean=json.loads((root/'outputs/unrealego_gt2d_ideal_20261115'/f'body{pair:02d}'/'observations_local.json').read_text());base=np.asarray(clean['views'],float)[:,:,:2]
 for sigma in [1,3,5,8]:
  for seed in [20261115,20261116,20261117]:
   obs=json.loads((out/f'body{pair:02d}'/f'gt_s{sigma}_seed{seed}'/'observations.json').read_text());xy=np.stack([np.asarray(p['xy']) for p in obs['points']],axis=1);rng=np.random.default_rng(np.random.SeedSequence([seed,pair,sigma]));expected=base+rng.normal(0,sigma,size=base.shape);delta=float(np.max(np.abs(xy-expected)));assert delta==0;checks.append({'pair':f'body{pair:02d}','sigma_px':sigma,'seed':seed,'max_coordinate_difference_px':delta})
(run/'evidence/noise_check.json').write_text(json.dumps({'exit_status':0,'checks':checks,'note':'Recomputed from official GT2D and fixed Gaussian seeds; no body 3D data read.'},indent=2)+'\n');print('Verified',len(checks),'independently reproduced noise arrays.')
