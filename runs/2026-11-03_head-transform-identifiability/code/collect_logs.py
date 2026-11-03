"""Extract progress checkpoints, preserving numerical values and warnings; redact host paths."""
import argparse,re
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
logs=sorted(a.results.glob('*.txt'))
expected=27
if len(logs)!=expected: raise RuntimeError('unexpected number of run logs: '+str(len(logs)))
lines=['Run command: see code/run_experiments.sh; paths supplied via environment variables.', 'The set -e batch completed with exit status 0; each listed child returned 0.', 'Selected optimizer checkpoints, settings, completion lines and warnings. Machine root redacted.']
for path in logs:
 raw=path.read_text();
 if 'DONE full seconds' not in raw: raise RuntimeError('missing completion: '+path.name)
 lines.append('\n### '+path.stem+' (exit 0)')
 for line in raw.splitlines():
  if line.startswith(('settings ', 'frames ', 'DONE ', 'WARNING:', '/vepfs-', '  warnings.warn')):
   if line.startswith('frames ') and not re.search(r' step (0|400|499|1400|1499) ',line): continue
   lines.append(re.sub(r'/(?:vepfs-mlp2|home|root)/[^ :]*','<REDACTED_PATH>',line))
with a.output.open('x') as f:f.write('\n'.join(lines)+'\n')
