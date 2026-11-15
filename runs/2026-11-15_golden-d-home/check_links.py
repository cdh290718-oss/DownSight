"""Check local inline Markdown links in both root README files."""
import json
import re
from pathlib import Path

root = Path(__file__).resolve().parents[2]
errors = []
count = 0
for name in ['README.md', 'README_LEGACY.md']:
    path = root / name
    for url in re.findall(r'\]\(([^)]+)\)', path.read_text()):
        if url.startswith(('https://', 'http://', '#', 'mailto:')):
            continue
        local = url.split('#', 1)[0]
        if not local:
            continue
        count += 1
        if not (path.parent / local).exists():
            errors.append({'document': name, 'target': url})
result = {'command': 'python3 runs/2026-11-15_golden-d-home/check_links.py',
          'exit_status': 0 if not errors else 1, 'local_links_checked': count,
          'broken': errors}
(Path(__file__).resolve().parent / 'evidence/links.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
if errors:
    raise SystemExit(1)
