"""Recompute per-view upper-body RMSE from published per-point distances."""
import hashlib
import json
import math
from pathlib import Path

root = Path(__file__).resolve().parents[2]
source = root / 'runs/2026-11-15_real-d-stereo/evidence/metrics.json'
out = Path(__file__).resolve().parent / 'evidence/upper_body_rmse.json'
groups = {
    'shoulder_elbow_wrist': [f'{side}_{part}' for part in ('shoulder', 'elbow', 'wrist') for side in ('left', 'right')],
    'elbow_wrist': [f'{side}_{part}' for part in ('elbow', 'wrist') for side in ('left', 'right')],
}
data = json.loads(source.read_text())
rows = []
for row in data['frames']:
    if row['case'] not in ('D', 'original', 'v2'):
        continue
    values = {}
    for name, points in groups.items():
        distances = [row['per_point'][point] for point in points]
        values[name] = {'point_names': points, 'distances_px': distances,
                        'rmse_px': math.sqrt(sum(v*v for v in distances) / len(distances))}
    rows.append({'camera': row['camera'], 'case': row['case'], 'groups': values})
assert len(rows) == 6 and len({(r['camera'], r['case']) for r in rows}) == 6
result = {'source': str(source.relative_to(root)), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
          'command': 'python runs/2026-11-15_upper-body-rmse/calculate.py', 'exit_status': 0,
          'formula': 'sqrt(mean of squared 2D Euclidean point distances in pixels)', 'frames': rows}
out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
print(json.dumps(result, indent=2))
