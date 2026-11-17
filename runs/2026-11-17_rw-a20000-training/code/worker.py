"""Background supervisor records real child exit status, including startup errors."""
import argparse
import datetime
import json
import subprocess
import sys
import time
import os
import hashlib
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);p.add_argument('--probe-steps',type=int,default=0);a=p.parse_args()
run=Path(__file__).resolve().parents[1];root=run.parents[1]
cmd=[sys.executable,'-u',str(run/'code/train.py'),'--workspace',str(root.parent),'--confirmed','--output',str(a.job/'results')]
if a.probe_steps:cmd+=['--probe-steps',str(a.probe_steps)]
start=time.time()
if not a.probe_steps:
    (a.job/'pipeline_status.json').write_text(json.dumps(dict(state='training',stages=[]))+'\n')
try:
    process=subprocess.run(cmd,cwd=root)
    result=dict(exit_code=process.returncode,elapsed_seconds=time.time()-start,
                finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
except BaseException as exc:
    result=dict(exit_code=1,elapsed_seconds=time.time()-start,error=repr(exc),
                finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(a.job/'exit.json').write_text(json.dumps(result,indent=2)+'\n')
if not a.probe_steps and result['exit_code'] == 0:
    # Child training releases GPU memory before the evaluation process starts.
    # Results remain local/unpublished until reviewed in the next handoff.
    stages=[]
    status=a.job/'pipeline_status.json'
    env=dict(os.environ,DOWNSIGHT_TRAIN_JOB=str(a.job),D_INTERVENTION_OUTPUT='outputs/rw_a20000_evaluation_20261117/d_fit')
    body_python=root.parent/'miniconda3/envs/egosmplx/bin/python'
    def publish(state, **extra):
        value=dict(state=state,training_exit_code=result['exit_code'],stages=stages,**extra)
        status.write_text(json.dumps(value,indent=2)+'\n')
        (run/'evidence/pipeline_execution.json').write_text(json.dumps(value,indent=2)+'\n')
    try:
        launch=json.loads((a.job/'launch.json').read_text())
        for path, expected in launch['pipeline_sha256'].items():
            assert hashlib.sha256((run/path).read_bytes()).hexdigest()==expected, path
        (root/'outputs/rw_a20000_evaluation_20261117').mkdir(exist_ok=False)
        commands=[('detector_2d',[sys.executable,'-u',str(run/'code/evaluate_2d.py')]),
                  ('prepare_full_2d_fit',[sys.executable,str(run/'code/prepare_fit.py')]),
                  ('fit_full_2d',[str(body_python),'-u',str(run/'fit.py'),'--threads','2']),
                  ('evaluate_body',[str(body_python),'-u',str(run/'evaluate.py')]),
                  ('summarize',[sys.executable,str(run/'code/summarize.py')]),
                  ('verify_evidence',[sys.executable,str(run/'code/verify_evidence.py')])]
        for name, command in commands:
            publish('evaluating',current_stage=name)
            started=time.time()
            with (a.job/(name+'.log')).open('xb') as log:
                completed=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
            stages.append(dict(stage=name,exit_code=completed.returncode,elapsed_seconds=time.time()-started))
            if name=='fit_full_2d' and completed.returncode==0:
                completion=json.loads((root/'outputs/rw_a20000_evaluation_20261117/d_fit/fit_complete.json').read_text())
                (run/'evidence/fit_execution.json').write_text(json.dumps(dict(command='<BODY_PYTHON> fit.py --threads 2',exit_status=0,completion=completion),indent=2)+'\n')
            if completed.returncode: raise RuntimeError(name+' failed with exit '+str(completed.returncode))
        publish('completed',finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    except BaseException as exc:
        publish('failed',error=str(exc).replace(str(root.parent),'<WORKSPACE>'))
        sys.exit(1)
sys.exit(result['exit_code'])
