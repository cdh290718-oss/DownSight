"""Preserve the CUDA failure and retry the unchanged original flow in a new output."""
import json,os,re,subprocess,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2];RUN=Path(__file__).resolve().parent;OUT=ROOT/'outputs/synthetic_original_20261108'
read=lambda p:json.loads(p.read_text())
inputs=read(OUT/'inputs.json');attempts=read(OUT/'attempts.json');assert attempts[-1]['side']=='right' and attempts[-1]['exit_status']!=0
row=next(r for r in inputs['records'] if r['id']=='right');old=read(Path(row['config']));old_root=Path(old['output'])
logs=RUN/'evidence/right_failure';logs.mkdir(exist_ok=False)
for i,p in enumerate([OUT/'right.launcher.log']+sorted(old_root.rglob('*.log'))):
    if not p.is_file():continue
    text=p.read_text(errors='replace').replace(str(ROOT.parent),'<WORKSPACE>').replace(str(Path.home()),'<HOME>').replace('/opt'+'/conda','<BUILD_ROOT>').replace('/tmp/','<TEMP>/').replace('/usr/','<SYSTEM>/')
    text=re.sub(r'(?i)(token|password|secret|api_key)\s*[:=]\s*[^\s,}]+',r'\1=<REDACTED>',text)
    (logs/f'{i:02d}_{p.name}.txt').write_text(text)
new=dict(old);new['output']=str(OUT/'fits/right_retry1');assert {k for k in new if new[k]!=old[k]}=={'output'}
config=OUT/'configs/right_retry1.json';config.write_text(json.dumps(new,indent=2)+'\n');row['config']=str(config)
(OUT/'inputs.json').write_text(json.dumps(inputs,indent=2)+'\n')
(RUN/'evidence/retry.json').write_text(json.dumps(dict(side='right',initial_exit_status=attempts[-1]['exit_status'],initial_failure='Sapiens2 CUDA unspecified launch failure',
    failure_logs='right_failure/',retry_config_changes=['output'],fresh_output=True,algorithm_and_inputs_unchanged=True,
    cuda_probe=dict(command='CUDA_VISIBLE_DEVICES=0 <POSE_PYTHON> small CUDA tensor multiplication and synchronize',exit_status=0,stdout='cuda_available True\nsmall_cuda_check 5559680.0\n',scope='small probe only; does not guarantee full model execution')),indent=2)+'\n')
runtime=OUT/'runtime';env=os.environ.copy();env.update(PYTHONPATH=str(ROOT),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
    TMPDIR=str(runtime),XDG_CACHE_HOME=str(runtime/'cache'),XDG_CONFIG_HOME=str(runtime/'config'),MPLCONFIGDIR=str(runtime/'matplotlib'),
    YOLO_CONFIG_DIR=str(runtime/'ultralytics'),TORCH_HOME=str(runtime/'torch'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_MODE='disabled',YOLO_AUTOINSTALL='false')
cmd=[new['paths']['body_python'],'-m','egosmplx_pipeline','run','--config',str(config)];start=time.time()
with (OUT/'right_retry1.launcher.log').open('w') as f:r=subprocess.run(cmd,env=env,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
attempts.append(dict(side='right',attempt='retry1',command=cmd,exit_status=r.returncode,elapsed_seconds=time.time()-start))
(OUT/'attempts.json').write_text(json.dumps(attempts,indent=2)+'\n')
print('right retry exit',r.returncode,'seconds',round(attempts[-1]['elapsed_seconds'],1),flush=True)
raise SystemExit(r.returncode)
