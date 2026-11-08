"""Hash additional licensed/runtime assets not covered by pipeline preflight."""
import argparse,hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/synthetic_original_20261108';RUN=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--phase',choices=['during','after'],required=True);a=p.parse_args()
cfg=json.loads((OUT/'configs/left.json').read_text());backend=Path(cfg['paths']['egosmplx_repo']);wilor=Path(cfg['paths']['wilor_repo'])
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
items={}
for group,folder in [('egosmplx_human_models',backend/'common/utils/human_model_files'),('wilor_mano_data',wilor/'mano_data')]:
    for f in sorted(folder.rglob('*')):
        if f.is_file():items[group+'/'+f.relative_to(folder).as_posix()]=dict(sha256=digest(f),bytes=f.stat().st_size)
for name in ['wilor_config']:
    f=Path(cfg['paths'][name]);items[name+'/'+f.name]=dict(sha256=digest(f),bytes=f.stat().st_size)
manifest=OUT/'fits/left/observations/left/network/manifest.json'
if manifest.exists():
    meta=json.loads(manifest.read_text())
    for key in ['det_config','det_checkpoint']:
        f=Path(meta[key]);items[key+'/'+f.name]=dict(sha256=digest(f),bytes=f.stat().st_size)
result=dict(phase=a.phase,scope='Files read by installed human model and MANO layers plus original person detector and WiLoR configuration; six principal checkpoints separately verified by pipeline preflight',assets=items,
            access='SMPL-X, SMPL and MANO files require their respective licenses; obtain model and checkpoint resources through their providers and configured deployment. Assets are not redistributed here.')
if a.phase=='during':(OUT/'resource_manifest_during.json').write_text(json.dumps(result,indent=2)+'\n')
else:
    before=json.loads((OUT/'resource_manifest_during.json').read_text());result['unchanged_since_during']=before['assets']==items
    assert result['unchanged_since_during']
    detectors=[]
    for root in [OUT/'fits/left',OUT/'fits/right',OUT/'fits/right_retry1',OUT/'fits/right_retry2']:
        for file in root.glob('observations/*/network/manifest.json'):
            d=json.loads(file.read_text());record={k:digest(Path(d[k])) for k in ['det_config','det_checkpoint']};detectors.append(record)
    result['detector_hashes_equal_across_completed_inferences']=bool(detectors) and all(x==detectors[0] for x in detectors)
    assert result['detector_hashes_equal_across_completed_inferences']
    (RUN/'evidence/resources.json').write_text(json.dumps(result,indent=2)+'\n')
print(a.phase,len(items),'additional resource hashes')
