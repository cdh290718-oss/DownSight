"""Expose retained/excluded observations and stereo support for every fixed view."""
import argparse,json
from pathlib import Path
import numpy as np
from egosmplx_pipeline.body_common import JOINT_MAPPING
from egosmplx_pipeline.v2_policy import EXEMPT,interior

p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();data=json.loads(a.inputs.read_text());frames=[]
for r in data['records']:
 lookup={q['name']:q for q in json.loads(Path(r['row']['predictions']['sapiens']).read_text())['keypoints']};ctx=json.loads(Path(r['context']).read_text())
 valid=[n for n in JOINT_MAPPING if n in lookup and lookup[n]['score']>=.3 and np.isfinite([lookup[n]['x'],lookup[n]['y'],lookup[n]['score']]).all() and 0<=lookup[n]['x']<1280 and 0<=lookup[n]['y']<720]
 retained=[n for n in valid if n in EXEMPT or interior([lookup[n]['x'],lookup[n]['y']])]
 frames.append(dict(pair=r['pair'],frame=r['id'],camera=r['camera'],body_original_valid=valid,body_retained=retained,body_excluded_by_margin=[n for n in valid if n not in retained],matches=ctx['matches'],hand_2d_retained_by_side=(np.asarray(ctx['hand_weights'])>0).sum(1).tolist(),hand_3d_retained_by_side=(np.asarray(ctx['tri_weights'])>0).sum(1).tolist(),time_difference_seconds=ctx['timestamp_difference_seconds']))
a.out.write_text(json.dumps(dict(frames=frames),indent=2)+'\n');print('Summarized',len(frames),'fixed views')
