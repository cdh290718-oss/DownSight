"""Launch authorized single-GPU training or smoke check in a fresh local job."""
import argparse
import datetime
import hashlib
import json
import os
import subprocess
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--confirmed',action='store_true')
p.add_argument('--probe-steps',type=int,default=0)
a=p.parse_args()
if not a.confirmed:p.error('Use only after the user has authorized this training run.')
run=Path(__file__).resolve().parents[1];root=run.parents[1];workspace=root.parent
assert (run/'evidence/preflight.json').exists()
if not a.probe_steps:
    probe=json.loads((run/'evidence/probe.json').read_text())
    assert probe['exit_code']==0 and probe['result']['finite_loss_and_gradients']
free=subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
assert int(free.strip())>=50000, 'Insufficient currently free GPU memory; no other process will be stopped.'
tag='rw_a20000_probe_' if a.probe_steps else 'rw_a20000_training_'
job=root/'outputs'/(tag+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
job.mkdir(exist_ok=False)
for name in ['tmp','cache','torch','triton','cuda','mpl','hf']:(job/name).mkdir()
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',
         TMPDIR=str(job/'tmp'),XDG_CACHE_HOME=str(job/'cache'),TORCH_HOME=str(job/'torch'),
         TRITON_CACHE_DIR=str(job/'triton'),CUDA_CACHE_PATH=str(job/'cuda'),MPLCONFIGDIR=str(job/'mpl'),HF_HOME=str(job/'hf'))
python=workspace/'miniconda3/envs/stereo-hand-fusion/bin/python'
cmd=[str(python),'-u',str(run/'code/worker.py'),'--job',str(job),'--probe-steps',str(a.probe_steps)]
with (job/'console.log').open('xb') as log:
    proc=subprocess.Popen(cmd,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(pid=proc.pid,command=cmd,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    job=str(job),probe_steps=a.probe_steps,
    baseline_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
    source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (run/'code').glob('*.py')},
    pipeline_sha256={str(f.relative_to(run)):hashlib.sha256(f.read_bytes()).hexdigest() for f in list(run.rglob('*.py'))+[run/'protocol.json']},
    config_sha256=hashlib.sha256((run/'config.json').read_bytes()).hexdigest())
(job/'launch.json').write_text(json.dumps(record,indent=2)+'\n')
public=json.loads(json.dumps(record).replace(str(workspace),'<WORKSPACE>'))
(run/'evidence'/('probe_launch.json' if a.probe_steps else 'launch.json')).write_text(json.dumps(public,indent=2)+'\n')
print(json.dumps(dict(pid=proc.pid,job=str(job.relative_to(root)),started_utc=record['started_utc'])),flush=True)
