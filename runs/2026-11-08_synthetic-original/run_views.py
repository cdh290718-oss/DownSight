"""Run views one at a time on one GPU, preserving all logs and exit statuses."""
import json,os,subprocess,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/synthetic_original_20261108'
records=json.loads((OUT/'inputs.json').read_text())['records'];attempts=[]
runtime=OUT/'runtime';runtime.mkdir(exist_ok=True)
env=os.environ.copy();env.update(PYTHONPATH=str(ROOT),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
    TMPDIR=str(runtime),XDG_CACHE_HOME=str(runtime/'cache'),XDG_CONFIG_HOME=str(runtime/'config'),MPLCONFIGDIR=str(runtime/'matplotlib'),
    YOLO_CONFIG_DIR=str(runtime/'ultralytics'),TORCH_HOME=str(runtime/'torch'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_MODE='disabled',YOLO_AUTOINSTALL='false')
for row in records:
    cfg=json.loads(Path(row['config']).read_text());cmd=[cfg['paths']['body_python'],'-m','egosmplx_pipeline','run','--config',row['config']]
    started=time.time()
    with (OUT/f'{row["id"]}.launcher.log').open('w') as f:r=subprocess.run(cmd,env=env,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    attempts.append(dict(side=row['id'],command=cmd,exit_status=r.returncode,elapsed_seconds=time.time()-started))
    (OUT/'attempts.json').write_text(json.dumps(attempts,indent=2)+'\n')
    print(row['id'],'exit',r.returncode,'seconds',round(attempts[-1]['elapsed_seconds'],1),flush=True)
if any(q['exit_status'] for q in attempts):raise SystemExit(1)
