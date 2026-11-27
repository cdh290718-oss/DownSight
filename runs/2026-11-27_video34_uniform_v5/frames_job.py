"""Run independent stereo pairs with bounded CPU workers and separate logs."""
from concurrent.futures import ThreadPoolExecutor
from base import *
import subprocess,os
def job(i):
    key=read(RUN/'evidence/frames.json')[i]['pair']
    for mode in ['v4','finish_frame','render_pairs']:
        dest={'v4':RUN/'evidence'/(key+'.json'),'finish_frame':OUT/(key+'_hands.json'),'render_pairs':RUN/'evidence'/(key+'_render.json')}[mode]
        if dest.exists() and read(dest)['exit_status']==0:continue
        with (OUT/(key+'_'+mode+'.log')).open('a') as f:
            result=subprocess.run(['bash',str(RUN/'run.sh'),mode,str(i)],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT)
            if result.returncode:return (key,mode,result.returncode)
    return (key,'complete',0)
if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=6) as pool:
        results=list(pool.map(job,range(20)))
    write(OUT/'jobs.json',results)
    if any(r[-1] for r in results):raise SystemExit(1)
