import os,time,json,subprocess,sys,datetime
from pathlib import Path
r=Path(__file__).resolve().parents[1];root=r.parents[1];out=root/'outputs/rw_test_completion_20261117';launch=root/'outputs/rw_test_completion_launcher_20261117';pid=json.loads((launch/'launch.json').read_text())['pid']
def state(value,**kw):
 (launch/'status.json').write_text(json.dumps(dict(state=value,time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw),indent=2)+'\n')
state('downloading',pid=pid)
while not (out/'download_complete.json').exists():
 p=Path('/proc')/str(pid)/'stat'
 if not p.exists() or p.read_text().split(') ')[1][0]=='Z':state('failed',stage='download',reason='process ended without completion marker; see download.log');sys.exit(1)
 time.sleep(30)
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='8',CUDA_VISIBLE_DEVICES='0');python=root.parent/'miniconda3/envs/stereo-hand-fusion/bin/python'
for name in ['extract','manifest','evaluate_2d']:
 state(name)
 with (launch/(name+'.log')).open('xb') as log:
  result=subprocess.run([str(python),'-u',str(r/'code'/(name+'.py'))],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
 if result.returncode:state('failed',stage=name,exit_code=result.returncode);sys.exit(result.returncode)
state('completed')
