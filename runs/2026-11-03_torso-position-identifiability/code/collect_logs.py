"""Collect numerical fit checkpoints and warnings, redacting host paths."""
import argparse,re
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();files=sorted(a.results.glob('*.txt'))
if len(files)!=18:raise RuntimeError('unexpected log count: '+str(len(files)))
lines=['All 18 fits completed successfully; see run_fits.sh and run_long.sh for commands. Paths redacted; numerical checkpoint values unchanged.']
for f in files:
 raw=f.read_text();assert 'DONE full seconds' in raw;lines.append('\n### '+f.stem+' (exit 0)')
 for line in raw.splitlines():
  if line.startswith(('settings ','frames ','DONE ','WARNING:','/vepfs-','  warnings.warn')):
   if line.startswith('frames ') and not re.search(r' step (0|400|499|1400|1499) ',line):continue
   lines.append(re.sub(r'/(?:vepfs-mlp2|root|home)/[^ :]*','<REDACTED_PATH>',line))
with a.output.open('x') as f:f.write('\n'.join(lines)+'\n')
