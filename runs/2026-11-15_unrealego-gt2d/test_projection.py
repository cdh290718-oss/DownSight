import json
from pathlib import Path
import numpy as np,torch
from egosmplx_pipeline.camera import FishEyeCameraCalibrated
root=Path(__file__).resolve().parents[2]; p=root.parent/'EgoFishPose/experiments/benchmark_body_hand10_head01_20261014_v1/body_calibration_fisheye';a=[]
for side in ['left','right']:
 c=FishEyeCameraCalibrated(p/('camera_'+side+'.json'));x=np.array([[.1,.2,.8],[-.1,-.3,1.4],[.2,-.1,.5]],dtype=np.float32);d=np.max(np.abs(c.world2camera(x)-c.world2camera_pytorch(torch.tensor(x)).numpy()));a.append({'camera':side,'max_px':float(d)});assert d<.1
(Path(__file__).resolve().parent/'evidence/projection_check.json').write_text(json.dumps({'command':'PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 python runs/2026-11-15_unrealego-gt2d/test_projection.py','exit_status':0,'checks':a},indent=2)+'\n');print(a)
