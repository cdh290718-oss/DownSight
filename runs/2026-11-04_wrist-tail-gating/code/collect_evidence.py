"""Collect compact gate decisions, configuration counts, and logs without host paths."""
import argparse,json,re,hashlib
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--published',type=Path,required=True);p.add_argument('--recomputed',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--fits',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();source=np.load(a.inputs/'baseline/observations.npz',allow_pickle=False);names=set(source['names'].tolist());pub=[r for r in json.loads(a.published.read_text()) if r['joint'] in names];rec=[r for r in json.loads(a.recomputed.read_text()) if r['joint'] in names];summary=json.loads((a.inputs/'summary.json').read_text());out={'published_calibration':'prior published calibration; original status is preserved','recomputed_calibration':'image-estimated fit calibration; status recomputed without DFM','published':pub,'recomputed':rec,'mode_counts':{k:{'test_cam2_by_name':v['test_cam2_by_name'],'test_stereo_by_name':v['test_stereo_by_name']} for k,v in summary.items()}}
with (a.output/'gating_decisions.json').open('x') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
files=sorted(a.fits.glob('*.txt'));assert len(files)==28;lines=['28 fits completed with exit status 0. Logs are reduced to settings, selected optimization checkpoints, completion and warnings; host paths are replaced.']
for path in files:
 s=path.read_text();assert 'DONE full seconds' in s;lines.append('\n### '+path.stem+' (exit 0)')
 for line in s.splitlines():
  keep=line.startswith(('settings ','frames ','DONE ','WARNING:','/vepfs-','  warnings.warn'))
  if line.startswith('frames ') and not re.search(r' step (0|400|499|1400|1499) ',line):keep=False
  if keep:lines.append(re.sub(r'/(?:vepfs-mlp2|home|root)/[^ :]*','<REDACTED_PATH>',line))
with (a.output/'fit_checkpoints.txt').open('x') as f:f.write('\n'.join(lines)+'\n')
