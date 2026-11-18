"""Publish compact execution records, excluding local full provenance and raw logs."""
import json,hashlib,subprocess,platform
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];RUN=Path(__file__).resolve().parent;OUT=ROOT/'outputs/cross_dataset100_20261118'
jobs={}
for name in ['index_retry','prepare_unreal','inference_real_retry','inference_unreal']:
 d=json.loads((OUT/(name+'_status.json')).read_text());assert d['exit_status']==0,(name,d)
 jobs[name]={k:d[k] for k in ['exit_status','elapsed_seconds']}
logs={}
for name in ['inference_real_retry','inference_unreal']:
 lines=(OUT/(name+'.log')).read_text().splitlines();statuses=[line for line in lines if len(line.split())==3 and line.split()[0] in ['original','rw500','rw5000','rw20000']];counts={}
 for line in statuses:
  model,ident,status=line.split();dataset=ident.rsplit('_',1)[0];key=model+'/'+dataset+'/'+status;counts[key]=counts.get(key,0)+1
 assert all(v==100 and k.endswith('/ok') for k,v in counts.items());assert len(counts)==(8 if name=='inference_real_retry' else 4)
 logs[name]={'sample_status_counts':counts,'last_line':lines[-1]}
sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(RUN.glob('*.py'))}
back=ROOT.parent/'EgoSMPLX';result=subprocess.run(['git','-C',str(back),'rev-parse','HEAD'],capture_output=True,text=True)
record={'base_commit':'1d12311ec0c5ff5163000933012937003d927d36','working_tree':'new uncommitted run code during execution; unrelated existing modifications preserved','backend_commit':result.stdout.strip() if result.returncode==0 else 'not available','source_sha256':sources,'projector_sha256':hashlib.sha256((back/'main/FishEyeCalibrated.py').read_bytes()).hexdigest(),'prepare_real_exit_status':0,'jobs':jobs,'log_summaries':logs,'raw_outputs':'retained only in <WORKSPACE>/outputs/cross_dataset100_20261118','public_redaction':'absolute machine paths replaced by placeholders; only aggregated status and derived numerical errors published; initial interrupted attempts retained in initial_execution.json','command':'python runs/2026-11-18_cross_dataset100/collect_execution.py','exit_status':0}
(RUN/'evidence/execution.json').write_text(json.dumps(record,indent=2)+'\n');print('Execution and per-sample status counts verified')
