"""Compare locally available frozen assets by logical path without publishing their contents."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[2]
run = Path(__file__).resolve().parent
lock = json.loads((root / 'experiments/synthetic_template_head_stereo_20261109/golden_d.json').read_text())
rows = []
for item in lock['external_inputs'] + lock['external_outputs']:
    rel = Path(item['resource'])
    if rel.is_absolute() or '..' in rel.parts:
        raise ValueError('Invalid asset path')
    path = root / rel
    available = path.is_file()
    digest = hashlib.sha256(path.read_bytes()).hexdigest() if available else None
    rows.append({'resource': rel.as_posix(), 'available': available,
                 'sha256_matches_manifest': digest == item['sha256'] if available else None})
result = {'command': 'python3 runs/2026-11-15_golden-d-home/check_external.py',
          'exit_status': 0, 'scope': 'logical paths and digest comparison only; no weights or environment audit',
          'assets': rows}
(run / 'evidence/external_hashes.json').write_text(json.dumps(result, indent=2) + '\n')
print('assets:', len(rows), 'available:', sum(x['available'] for x in rows),
      'matching:', sum(x['sha256_matches_manifest'] is True for x in rows))
