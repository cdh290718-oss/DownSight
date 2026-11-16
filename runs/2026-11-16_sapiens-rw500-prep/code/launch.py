"""Detached launch, only after the user explicitly confirms training."""
import argparse, datetime, json, os, subprocess, sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--confirmed',action='store_true');p.add_argument('--workspace',type=Path,required=True);a=p.parse_args()
if not a.confirmed:p.error('Preparation only. Obtain explicit user confirmation before --confirmed.')
root=Path(__file__).resolve().parents[3];run=Path(__file__).resolve().parents[1]
assert (run/'evidence/preflight.json').exists()
job=root/'outputs'/('sapiens_rw500_training_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
job.mkdir(exist_ok=False);(job/'tmp').mkdir();(job/'cache').mkdir()
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',TMPDIR=str(job/'tmp'),XDG_CACHE_HOME=str(job/'cache'))
python=a.workspace.resolve()/'miniconda3/envs/stereo-hand-fusion/bin/python'
cmd=[str(python),'-u',str(run/'code/worker.py'),'--workspace',str(a.workspace.resolve()),'--job',str(job)]
with (job/'console.log').open('xb') as log:
 proc=subprocess.Popen(cmd,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(job/'launch.json').write_text(json.dumps(dict(pid=proc.pid,command=cmd,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2))
print(json.dumps(dict(pid=proc.pid,job=str(job),message='Background process launched; check status.json and console.log once for startup. No polling daemon.'),indent=2))
