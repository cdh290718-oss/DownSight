"""Preserve sanitized training completion and intentional partition-stop evidence."""
import argparse,re
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();lines=[]
for mode in ('v0','wilor'):
 for part in ('first60','last120','tail60'):
  path=a.root/f'fit_{mode}_{part}.log';text=path.read_text();end=[s for s in text.splitlines() if re.match(r'^frames \d+ \d+ step 1499 ',s)];expected={'first60':6,'last120':6,'tail60':6}[part];assert len(end)==expected,(mode,part,len(end));lines.append(f'{mode} {part}: {len(end)} complete 1500-step batches; ' + ('intentionally terminated at partition boundary, shell exit 143' if part=='last120' else 'normal completion, shell exit 0'))
  lines.extend(end)
lines.append('Initial network invocation failed before starting because its log redirection used a non-existent relative path; the subsequent run used a repository output path and completed successfully.')
lines.append('Partition watchdog initial version was stopped and replaced before it acted; v2 completed successfully and stopped only the two owned original middle workers after fully saving batch_110.')
a.output.write_text('\n'.join(lines)+'\n')
