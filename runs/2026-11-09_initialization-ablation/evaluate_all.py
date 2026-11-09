"""Run held-out evaluation and publication reports only after every planned fit succeeds."""
import hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];RUN=Path(__file__).resolve().parent;OUT=ROOT/'outputs/initialization_ablation_20261109'
read=lambda p:json.loads(p.read_text())
cases=read(RUN/'cases.json');records=read(RUN/'evidence/execution.json');assert len(records)==len(cases) and all(q['exit_status']==0 for q in records)
logs=[]
for c in cases:
 name=c['case']+'_'+c['seed'];log=OUT/(name+'.log');content=log.read_text();lines=content.splitlines();logs.append(dict(case=name,sha256=hashlib.sha256(log.read_bytes()).hexdigest(),completed_lines=[x for x in lines if x.startswith('DONE ')],nonfatal_warnings=[x.split('UserWarning: ',1)[1].replace(str(ROOT.parent),'<WORKSPACE>').replace('/opt'+'/conda/','<BUILD>/') for x in lines if 'UserWarning: ' in x],standard_warnings=[x for x in lines if x.startswith('WARNING:')]))
(RUN/'evidence/fit_log_summary.json').write_text(json.dumps(logs,indent=2)+'\n')
python=read(ROOT/'outputs/synthetic_original_20261108/configs/left.json')['paths']['body_python'];env=os.environ.copy();env.update(PYTHONPATH=str(ROOT),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TMPDIR=str(OUT/'runtime'),XDG_CACHE_HOME=str(OUT/'runtime/cache'),MPLCONFIGDIR=str(OUT/'runtime/matplotlib'))
records=[]
for script in ['evaluate.py','evaluate_shape.py','evaluate_mesh.py','verify.py','plot_results.py']:
 start=time.time();log=OUT/(script+'.log')
 with log.open('x') as stream:q=subprocess.run([python,str(RUN/script)],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
 record=dict(command=['<BODY_PYTHON>',str((RUN/script).relative_to(ROOT))],exit_status=q.returncode,elapsed_seconds=time.time()-start,source_sha256=hashlib.sha256((RUN/script).read_bytes()).hexdigest());records.append(record);(RUN/'evidence/evaluation_execution.json').write_text(json.dumps(records,indent=2)+'\n');print(script,q.returncode,flush=True)
 if q.returncode:
  text=log.read_text().replace(str(ROOT.parent),'<WORKSPACE>').replace('/usr/','<SYSTEM>/');(RUN/'evidence'/(script+'_failure.txt')).write_text(text);raise SystemExit(q.returncode)
