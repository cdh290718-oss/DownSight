"""Stop only this run's original owned workers after their final needed batch is fully written."""
import argparse,os,signal,time,zipfile
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--v0-pid',type=int,required=True);p.add_argument('--wilor-pid',type=int,required=True);a=p.parse_args();pending={'v0':a.v0_pid,'wilor':a.wilor_pid}
while pending:
 for mode,pid in list(pending.items()):
  proc=Path('/proc')/str(pid)/'cmdline';file=a.root/'fits'/f'{mode}_last120'/'batch_110.npz'
  if not proc.exists():raise RuntimeError(f'owned worker {mode} ended before planned boundary')
  args=proc.read_bytes().replace(b'\0',b' ').decode();assert 'fit_180.py' in args and f'fits/{mode}_last120' in args and '--first 60 --last 180' in args
  if file.exists():
   try:
    with np.load(file,allow_pickle=False) as d:
     frames=d['frame_ids'];j=d['joints_cam2'];assert len(frames)==10 and j.shape[0]==10 and np.isfinite(j).all()
   except (OSError,ValueError,EOFError,zipfile.BadZipFile):continue
   os.kill(pid,signal.SIGTERM);print(f'{mode}: stopped original owned worker after batch_110 was complete; next partition is 120..180',flush=True);del pending[mode]
 time.sleep(5)
