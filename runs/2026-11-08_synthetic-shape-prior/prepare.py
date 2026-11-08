"""Reuse identical B observations/cameras/Head anchor; never extract GT shape."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];RUN=Path(__file__).resolve().parent;OUT=ROOT/'outputs/synthetic_shape_prior_20261108'
source=ROOT/'outputs/synthetic_stereo_abcd_20261108/inputs.json';data=source.read_bytes();q=json.loads(data)
assert set(q)=={'head_position_m','head_rotation_matrix','views','source_hashes','scope'}
assert all(len(v['hands'])==2 for v in q['views'].values())
OUT.mkdir(exist_ok=True);(OUT/'runtime').mkdir(exist_ok=True)
(OUT/'inputs.json').write_bytes(data);(RUN/'evidence/inputs.json').write_bytes(data)
result=dict(exit_status=0,command='python runs/2026-11-08_synthetic-shape-prior/prepare.py',inputs_sha256=hashlib.sha256(data).hexdigest(),inputs_identical_to_previous_B=True,shape_gt_extracted=False,settings_sha256=hashlib.sha256((RUN/'settings.json').read_bytes()).hexdigest())
(RUN/'evidence/preparation.json').write_text(json.dumps(result,indent=2)+'\n');print('prepared unchanged B observations and Head/camera metadata')
