"""Read current local training/evaluation state without modifying artifacts."""
import json
from pathlib import Path

run=Path(__file__).resolve().parents[1];root=run.parents[1]
launch=json.loads((run/'evidence/launch.json').read_text())
job=root/'outputs'/Path(launch['job']).name
result={'job':str(job.relative_to(root)),'pid':launch['pid']}
for name in ['results/status.json','exit.json','pipeline_status.json']:
    p=job/name
    if p.exists():
        try:result[name]=json.loads(p.read_text())
        except json.JSONDecodeError:result[name]='being updated; read again'
print(json.dumps(result,indent=2))
