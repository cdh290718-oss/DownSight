"""Audit the designation using public code and per-frame evidence."""
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
import subprocess

RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[1]
def read(path):return json.loads((ROOT/path).read_text())
def digest(data):return hashlib.sha256(data).hexdigest()
def pinned(commit,path):return subprocess.check_output(['git','show',commit+':'+path],cwd=ROOT)

def main():
    manifest=read('configs/golden_d.2d-v3.json')
    old=manifest['previous_manifest']
    assert (ROOT/old).read_bytes()==pinned('8f53897','configs/golden_d.current.json')
    assert read(old)['method_id']=='golden-d-2d-v2' and manifest['method_id']=='golden-d-2d-v3'
    for lock in manifest['locked_files']:
        path=lock['path']
        data=(ROOT/path).read_bytes()
        assert data==pinned(manifest['source_commit'],path) and digest(data)==lock['sha256']
    ref=read(manifest['reference_results'])
    key=manifest['reference_result_key']
    assert key=='summary.joint' and len(ref['rows'])==ref['n_scored']==500
    joint=ref['summary']['joint']
    for k,v in joint.items():
        assert math.isclose(mean(r['scores']['joint'][k] for r in ref['rows'].values()),v)
    comparison={}
    for split,count in [('test',100),('validation',200)]:
        r=read('runs/2026-11-23_fullbody_joint_candidates/evidence/'+split+'_results.json')
        assert len(r['rows'])==count
        comparison[split]={m:{k:r['summary']['all'][m][k] for k in ('mpjpe','pa')} for m in ('baseline','fixed','joint')}
        pa=read('runs/2026-11-23_'+('joint_pa15' if split=='test' else 'validation_pa15')+'/evidence/results.json')
        comparison[split]['joint_calibrated_pa15_mm']=pa['summary']['joint']['mapped15_pa_mm']
    evidence=dict(exit_status=0,command='python -B runs/2026-11-23_golden_d_v3/verify.py',worktree_base='8f53897c7b8677033c01f0353debb8fda8075f11',
                  method=manifest['method_id'],source_commit=manifest['source_commit'],locked_files=len(manifest['locked_files']),
                  archived_v2_sha256=digest((ROOT/old).read_bytes()),manifest_sha256=digest((ROOT/'configs/golden_d.2d-v3.json').read_bytes()),
                  diagram_sha256=digest((ROOT/'docs/downsight/golden-d-v3/stereo-fit.png').read_bytes()),test500=ref['summary'],test500_counts=ref['counts'],comparison=comparison)
    (RUN/'evidence/verification.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(manifest['method_id'],key,joint)

if __name__=='__main__':main()
