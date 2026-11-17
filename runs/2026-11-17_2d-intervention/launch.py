"""Run one stage with caches confined to the authorized repository output."""
import argparse
import os
import subprocess
import time
from pathlib import Path


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('stage',choices=['prepare','verify','fit','evaluate','analyze','write_report','audit_results'])
    ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--output',default='outputs/2d_intervention_20261117_retry1')
    a=ap.parse_args()
    # Standard-library-only parent can launch the existing body environment.
    import json
    run=Path(__file__).resolve().parent;root=run.parents[1]
    out=root/a.output
    assert (root/'outputs').resolve() in out.resolve().parents
    for sub in ['runtime/tmp','runtime/cache','runtime/matplotlib']:
        (out/sub).mkdir(parents=True,exist_ok=True)
    python=json.loads((root/'outputs/synthetic_original_20261108/configs/left.json').read_text())['paths']['body_python']
    env=os.environ.copy()
    env.update(PYTHONPATH=str(root),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',
               D_INTERVENTION_OUTPUT=a.output,
               OMP_NUM_THREADS=str(a.threads),MKL_NUM_THREADS=str(a.threads),OPENBLAS_NUM_THREADS=str(a.threads),
               TMPDIR=str(out/'runtime/tmp'),XDG_CACHE_HOME=str(out/'runtime/cache'),
               MPLCONFIGDIR=str(out/'runtime/matplotlib'),NUMBA_CACHE_DIR=str(out/'runtime/numba'))
    command=[python,'-u',str(run/(a.stage+'.py'))]
    if a.stage=='fit':command+=['--threads',str(a.threads)]
    start=time.time()
    with (out/(a.stage+'.log')).open('x') as log:
        proc=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
    result=dict(stage=a.stage,command=command,exit_status=proc.returncode,elapsed_seconds=time.time()-start,
                git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
                source_worktree='New run source files; preexisting unrelated modifications retained')
    (out/(a.stage+'_execution.json')).write_text(json.dumps(result,indent=2)+'\n')
    public=json.loads(json.dumps(result).replace(str(root.parent),'<WORKSPACE>'))
    dest=run/'evidence';dest.mkdir(exist_ok=True)
    (dest/(a.stage+'_execution.json')).write_text(json.dumps(public,indent=2)+'\n')
    raw=(out/(a.stage+'.log')).read_text()
    (dest/(a.stage+'_console.txt')).write_text(raw.replace(str(root.parent),'<WORKSPACE>'))
    print(json.dumps(public),flush=True)
    if proc.returncode:
        print(raw[-6000:].replace(str(root.parent),'<WORKSPACE>'),flush=True)
    raise SystemExit(proc.returncode)


if __name__=='__main__':
    main()
