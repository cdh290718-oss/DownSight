"""Match each archived predictor manifest to its configuration and actual model bytes."""
from pathlib import Path
import json
import hashlib

RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[1]
arch=ROOT.parent/'EgoFishPose/experiments/benchmark_body_hand10_head01_20261014_v1'
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for buf in iter(lambda:f.read(1048576),b''):h.update(buf)
    return h.hexdigest()
def read(path):return json.loads(path.read_text())

rows=[]
for i in range(1,5):
    pair='body%02d_left'%i
    config=arch/'body_configs_full_image'/(pair+'.json')
    mf=arch/'body_work_full_image'/pair/'observations'/pair/'network/manifest.json'
    cfg=read(config)
    manifest=read(mf)
    weight=Path(cfg['paths']['egosmplx_checkpoint'])
    assert Path(manifest['checkpoint_path'])==weight
    assert manifest['smplx_input_mode']=='full_image' and not manifest['detector_enabled']
    if rows:assert rows[0]['checkpoint_id']==str(weight.relative_to(ROOT.parent/'EgoSMPLX'))
    rows.append(dict(pair=pair,checkpoint_id=str(weight.relative_to(ROOT.parent/'EgoSMPLX')),
                     config_sha256=sha(config),prediction_manifest_sha256=sha(mf),
                     backend_config=Path(cfg['paths']['egosmplx_config']).name,profile=cfg['pipeline_profile'],
                     input_mode=manifest['smplx_input_mode'],rigid_head_weight=cfg['head_neck']['rigid_head_weight']))
digest=sha(weight)
assert digest in (ROOT/'docs/downsight/CHECKPOINT_zh.md').read_text()
(RUN/'evidence/model.json').write_text(json.dumps(dict(backend='EgoSMPLX',weight_sha256=digest,
            archive=arch.name,records=rows,exit_status=0),indent=2)+'\n')
