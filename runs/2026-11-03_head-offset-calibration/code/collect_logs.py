"""Preserve numerical checkpoints, status, warnings and failures with host paths redacted."""
import argparse,re
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--gen',type=Path,required=True);p.add_argument('--synth',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();g=sorted(a.gen.glob('*.txt'));s=sorted(a.synth.glob('*.txt'));assert len(g)==32 and len(s)==25
lines=['Gen 25 short and 7 long fits: all completed (batch exits 0). Synthetic: 20 short and 5 long cases, per-case exit below. Paths redacted; numeric checkpoints preserved.']
for path in g+s:
 text=path.read_text();isgen=path.parent==a.gen
 if isgen:assert 'DONE full seconds' in text
 else:assert 'CASE_EXIT ' in text
 failed=not isgen and 'CASE_EXIT 0' not in text
 lines.append('\n### '+('gen/' if isgen else 'synthetic/')+path.stem+(' (exit 0)' if isgen else ''))
 for line in text.splitlines():
  keep=failed or line.startswith(('settings ','frames ','step ','DONE ','WARNING:','CASE_EXIT ','/vepfs-','  warnings.warn'))
  if line.startswith('frames ') and not re.search(r' step (0|400|499|1400|1499) ',line):keep=False
  if line.startswith('step ') and not re.search(r'^step (0|200|400|600|799|2800|2999) ',line):keep=False
  if keep:lines.append(re.sub(r'/(?:vepfs-mlp2|home|root)/[^ :]*','<REDACTED_PATH>',line))
with a.output.open('x') as f:f.write('\n'.join(lines)+'\n')
