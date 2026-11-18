"""Detached local job with explicit exit-status record; no scheduler dependency."""
import argparse,subprocess,sys,json,time,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('name');p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args();root=Path(__file__).resolve().parents[2];out=root/'outputs/cross_dataset100_20261118'
if not a.worker:
 with (out/(a.name+'.log')).open('wb') as log:
  proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker',a.name,*a.command],cwd=root,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
 print(json.dumps({'name':a.name,'pid':proc.pid}));sys.exit(0)
start=time.time();result={'name':a.name,'pid':os.getpid(),'started_unix':start,'command':a.command};(out/(a.name+'_status.json')).write_text(json.dumps(result))
try:result['exit_status']=subprocess.run(a.command,cwd=root).returncode
except Exception as e:result.update(exit_status=-1,error=repr(e))
result['elapsed_seconds']=time.time()-start;(out/(a.name+'_status.json')).write_text(json.dumps(result,indent=2))
