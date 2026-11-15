import json,sys
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];RUN=Path(__file__).resolve().parent;OUT=ROOT/'outputs/unrealego_auto_loss_comparison_20261115';ARCH=ROOT.parent/'EgoFishPose/experiments/benchmark_body_hand10_head01_20261014_v1'
from egosmplx_pipeline.camera import FishEyeCameraCalibrated
rows=[]
for pair in range(1,5):
 points=json.loads((OUT/f'body{pair:02d}'/'auto/observations.json').read_text())['points']; n3=0;n2=0;single=0
 for p in points:
  a=p['accepted']; v=sum(p['valid']); assert not a or (v==2 and p['tri']['geometry_ok'] and p['tri']['depth_ok'] and p['tri']['depth_std_m']<=.5)
  use3=a;use2=not a and v>0;assert not (use3 and use2);n3+=use3;n2+=use2;single+=v==1
 rows.append({'pair':f'body{pair:02d}','3d':n3,'2d_fallback':n2,'single_view':single})
checks=[]
for side in ['left','right']:
 c=FishEyeCameraCalibrated(ARCH/'body_calibration_fisheye'/('camera_'+side+'.json'));x=torch.tensor([[.1,.2,.8],[-.1,-.3,1.4],[.2,-.1,.5]],requires_grad=True);y=c.world2camera_pytorch(x);y.sum().backward();err=float(np.max(np.abs(c.world2camera(x.detach().numpy())-y.detach().numpy())));assert err<.1 and bool(torch.isfinite(x.grad).all());checks.append({'camera':side,'numpy_vs_torch_max_px':err,'gradients_finite':True})
result={'command':'<BODY_PYTHON> runs/2026-11-15_auto-loss-comparison/test_preflight.py','exit_status':0,'routing':rows,'projection':checks,'scope':'projection gradient finite and numpy comparison; routing no double counting, not a convergence test'};(RUN/'evidence/preflight.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
