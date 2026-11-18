"""Detached local job with explicit end status."""
import argparse,subprocess,sys,json,time,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('name');p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args();root=Path(__file__).resolve().parents[2];out=root/'outputs/real001_golden_d_four_models_20261118'
if not a.worker:
 with (out/(a.name+'.log')).open('wb') as log:proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker',a.name,*a.command],cwd=root,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
 print(json.dumps({'name':a.name,'pid':proc.pid}));sys.exit(0)
start=time.time();d={'name':a.name,'pid':os.getpid(),'started_unix':start};(out/(a.name+'_status.json')).write_text(json.dumps(d))
try:d['exit_status']=subprocess.run(a.command,cwd=root).returncode
except Exception as e:d.update(exit_status=-1,error=repr(e))
d['elapsed_seconds']=time.time()-start;(out/(a.name+'_status.json')).write_text(json.dumps(d,indent=2))
