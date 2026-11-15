"""Record documentation checks without modifying prior run evidence."""
import json
import re
import subprocess
from pathlib import Path
root = Path(__file__).resolve().parents[2]
out = Path(__file__).resolve().parent / 'evidence'
readme = (root / 'README.md').read_text()
links = [u for u in re.findall(r'\]\(([^)]+)\)', readme) if not u.startswith(('https://', 'http://', '#'))]
missing = [u for u in links if not (root / u.split('#')[0]).exists()]
checks = []
for cmd in [['python3', 'tools/check_golden_d.py'], ['git', 'diff', '--check']]:
    p = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
    checks.append({'command': cmd, 'exit_status': p.returncode, 'stdout': p.stdout.replace(str(root), '<WORKSPACE>'), 'stderr': p.stderr.replace(str(root), '<WORKSPACE>')})
source = 'experiments/synthetic_template_head_stereo_20261109/evidence/results.json'
results = json.loads((root/source).read_text())
rank_records = {k: v['retained_rank'] for k, v in results.items() if isinstance(v, dict) and 'retained_rank' in v}
data = {'baseline': subprocess.check_output(['git','rev-parse','HEAD'], cwd=root, text=True).strip(), 'readme_local_links': len(links), 'missing_links': missing, 'checks': checks, 'shape_evidence': {'source': source, 'records': rank_records}, 'diagram_review': 'Manual syntax/data-flow review only; Mermaid renderer/browser not run.'}
(out / 'validation.json').write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(data, ensure_ascii=False, indent=2))
assert not missing and all(c['exit_status']==0 for c in checks)
assert rank_records and all(v == 1 for v in rank_records.values())
