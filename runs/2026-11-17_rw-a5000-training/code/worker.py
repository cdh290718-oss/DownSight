"""Background supervisor records real child exit status, including startup errors."""
import argparse
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);p.add_argument('--probe-steps',type=int,default=0);a=p.parse_args()
run=Path(__file__).resolve().parents[1];root=run.parents[1]
cmd=[sys.executable,'-u',str(run/'code/train.py'),'--workspace',str(root.parent),'--confirmed','--output',str(a.job/'results')]
if a.probe_steps:cmd+=['--probe-steps',str(a.probe_steps)]
start=time.time()
try:
    process=subprocess.run(cmd,cwd=root)
    result=dict(exit_code=process.returncode,elapsed_seconds=time.time()-start,
                finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
except BaseException as exc:
    result=dict(exit_code=1,elapsed_seconds=time.time()-start,error=repr(exc),
                finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(a.job/'exit.json').write_text(json.dumps(result,indent=2)+'\n')
sys.exit(result['exit_code'])
