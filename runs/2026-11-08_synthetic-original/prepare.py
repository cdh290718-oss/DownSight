"""Freeze the public synthetic dataset and make isolated original-flow jobs."""
import argparse,hashlib,json,subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
RUN=Path(__file__).resolve().parent
OUT=ROOT/'outputs/synthetic_original_20261108'
DATA=OUT/'dataset'
EXPECTED='192671271a74b99f41b9eb033d8f88b1dac77ada'

def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--template',type=Path,default=ROOT/'configs/pipeline.head01.developer.json');args=parser.parse_args()
    commit=subprocess.check_output(['git','-C',str(DATA),'rev-parse','HEAD'],text=True).strip()
    assert commit==EXPECTED
    for name,digest in json.loads((DATA/'SHA256SUMS.json').read_text()).items():
        assert hashlib.sha256((DATA/name).read_bytes()).hexdigest()==digest,name
    cfg=json.loads(args.template.read_text())
    assert cfg['pipeline_profile']=='mano_guided_original_seam_20260929_v1' and not cfg.get('stereo_v2',False)
    records=[]
    for side,cam,cal in [('left','cam3','cam3_left_lower.egosmplx.json'),('right','cam4','cam4_right_lower.egosmplx.json')]:
        row=dict(id=side,camera=cam,image=str(DATA/'samples/000001'/f'{side}.png'),calibration=str(DATA/'calibration'/cal))
        manifest=OUT/'manifests'/f'{side}.json';write(manifest,dict(frames=[row]))
        c=json.loads(json.dumps(cfg));c['manifest']=str(manifest);c['output']=str(OUT/'fits'/side);c['gpu']='0'
        config=OUT/'configs'/f'{side}.json';write(config,c)
        records.append(dict(id=side,camera=cam,config=str(config)))
    write(OUT/'inputs.json',dict(records=records))
    write(RUN/'evidence/inputs.json',dict(dataset_repository='https://github.com/cdh290718-oss/DownSightSmplxdata',dataset_commit=commit,
        manifest=records and [dict(side=r['id'],camera=r['camera']) for r in records],dataset_sha256=json.loads((DATA/'SHA256SUMS.json').read_text()),
        profile=cfg['pipeline_profile'],head_neck=cfg['head_neck'],fitting=cfg['fitting'],stereo_v2=False,checkpoint_sha256=cfg['checkpoint_sha256'],
        inference='Fresh original neural predictions; ground truth reserved for evaluation only',gpu='single device 0'))
