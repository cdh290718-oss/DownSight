"""Compare CPU and original GPU Sapiens2 output on the same left image."""
import json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];RUN=Path(__file__).resolve().parent
OUT=ROOT/'outputs/synthetic_original_20261108';read=lambda p:json.loads(p.read_text())
gpu=read(OUT/'fits/left/observations/left/sapiens/keypoints.json');cpu=read(OUT/'cpu_observations/left_pose_control/keypoints.json')
a={q['name']:q for q in gpu['keypoints']};b={q['name']:q for q in cpu['keypoints']}
assert gpu['image_sha256']==cpu['image_sha256'] and set(a)==set(b)
names=['nose','left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist','left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle']
points=[]
for n in names:
    x,y=a[n],b[n]
    points.append(dict(name=n,position_delta_px=math.hypot(x['x']-y['x'],x['y']-y['y']),gpu_score=x['score'],cpu_score=y['score'],
        gpu_selected=bool(x['score']>=.3 and x['inside_image']),cpu_selected=bool(y['score']>=.3 and y['inside_image']),
        same_score_threshold=(x['score']>=.3)==(y['score']>=.3)))
result=dict(scope='Same left input and Sapiens2 checkpoint, original GPU BF16 versus CPU BF16; no fitting and no ground truth',points=points,
    mean_position_delta_px=sum(q['position_delta_px'] for q in points)/len(points),max_position_delta_px=max(q['position_delta_px'] for q in points),
    score_threshold_disagreements=sum(not q['same_score_threshold'] for q in points),selection_disagreements=sum(q['gpu_selected']!=q['cpu_selected'] for q in points))
(RUN/'evidence/device_control.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='points'},indent=2))
