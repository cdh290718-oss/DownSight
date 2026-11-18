"""Resume only fixed-protocol CPU fitting/evaluation in a fresh local output."""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

RUN=Path(__file__).resolve().parents[1];ROOT=RUN.parents[1]

def main():
    out=ROOT/'outputs/rw_a20000_evaluation_20261118'
    assert (out/'d_fit/prepared.json').exists()
    for n in ['tmp','cache','mpl','torch']:(out/n).mkdir(exist_ok=False)
    env=dict(os.environ,PYTHONPATH=str(ROOT),PYTHONDONTWRITEBYTECODE='1',
             D_INTERVENTION_OUTPUT='outputs/rw_a20000_evaluation_20261118/d_fit',
             TMPDIR=str(out/'tmp'),XDG_CACHE_HOME=str(out/'cache'),MPLCONFIGDIR=str(out/'mpl'),
             TORCH_HOME=str(out/'torch'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    body=ROOT.parent/'miniconda3/envs/egosmplx/bin/python'
    reference=ROOT/'runs/2026-11-17_rw-a20000-training'
    for name in ['fit.py','geometry.py','protocol.json']:
        assert (RUN/name).read_bytes()==(reference/name).read_bytes()
    source=dict(command='source and protocol identity check before recovery',exit_status=0,
        execution_baseline=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        working_tree='New recovery run; preexisting unrelated changes preserved.',
        code_sha256={str(p.relative_to(RUN)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(RUN.rglob('*.py'))},
        recovery='Set PYTHONPATH to repository root for subprocess imports; new output destination; no loss, data, optimizer or model change.',
        unchanged_files=['fit.py','geometry.py','protocol.json'],
        original_training_execution_baseline='6fc30623c78c8eb4db2943955c9c0c5ed83a0563 plus code published at 4a4ead16ad79c31b30c705e5bfaf326fc65173fc')
    (RUN/'evidence/source.json').write_text(json.dumps(source,indent=2)+'\n')
    stages=[]
    commands=[('import_check',[str(body),'-c','from egosmplx_pipeline.body_common import BODY_POSE_NAMES; print(len(BODY_POSE_NAMES))']),
              ('fit',[str(body),'-u',str(RUN/'fit.py'),'--threads','2']),
              ('evaluate',[str(body),'-u',str(RUN/'evaluate.py')]),
              ('summarize',[sys.executable,str(RUN/'code/summarize.py')]),
              ('verify',[sys.executable,str(RUN/'code/verify_evidence.py')])]
    for name,command in commands:
        start=time.time()
        with (out/(name+'.log')).open('x') as f:
            result=subprocess.run(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        record=dict(stage=name,command=[s.replace(str(ROOT.parent),'<WORKSPACE>') for s in command],
                    exit_status=result.returncode,elapsed_seconds=time.time()-start)
        stages.append(record)
        (RUN/'evidence/recovery_execution.json').write_text(json.dumps(dict(stages=stages,
            environment=dict(PYTHONPATH='<REPO>',D_INTERVENTION_OUTPUT=env['D_INTERVENTION_OUTPUT']),
            completed=name=='verify' and result.returncode==0),indent=2)+'\n')
        print(name,'exit',result.returncode,flush=True)
        if result.returncode:raise SystemExit(result.returncode)
        if name=='fit':
            completion=json.loads((out/'d_fit/fit_complete.json').read_text())
            (RUN/'evidence/fit_execution.json').write_text(json.dumps(dict(command='<BODY_PYTHON> fit.py --threads 2',exit_status=0,completion=completion),indent=2)+'\n')
        if name in ['fit','evaluate']:
            text=(out/(name+'.log')).read_text().replace(str(ROOT.parent),'<WORKSPACE>')
            (RUN/'evidence'/(name+'_console.txt')).write_text(text)

if __name__=='__main__':main()
