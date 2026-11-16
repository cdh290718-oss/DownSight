"""Background supervisor records exit code, including pre-model startup failures."""
import argparse, datetime, json, subprocess, sys, time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--workspace',required=True);p.add_argument('--job',type=Path,required=True);a=p.parse_args()
start=time.time();result=subprocess.run([sys.executable,'-u',str(Path(__file__).with_name('train.py')),'--workspace',a.workspace,'--confirmed','--output',str(a.job/'results')])
(a.job/'exit.json').write_text(json.dumps(dict(exit_code=result.returncode,elapsed_seconds=time.time()-start,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2))
sys.exit(result.returncode)
