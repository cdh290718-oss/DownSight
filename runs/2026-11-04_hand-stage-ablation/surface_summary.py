"""Preserve local surface-check failures separately from numerical reload checks."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];RUN=Path(__file__).resolve().parent
OUT=ROOT/'outputs/hand_stage_ablation_20261104';read=lambda p:json.loads(p.read_text())
rows=[]
for group in ['A','B','C']:
    for rec in read(OUT/group/'inputs.json')['records']:
        cfg=read(Path(rec['config']));p=Path(cfg['output'])/'frames'/rec['id']/'fused/geometry_report.json'
        q=read(p)
        rows.append(dict(group=group,pair=rec['pair'],camera=rec['camera'],passed=q['all_local_crossing_checks_passed'],source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),sides={side:{stage:{k:v for k,v in data[stage].items() if k!='face_pairs'} for stage in ['before','after']} for side,data in q['local_intersection_checks'].items()}))
result=dict(scope='Pipeline local forearm/seam checks versus full surface; excludes shared vertices and coplanar overlap; not a global intersection certificate',rows=rows)
(RUN/'evidence/surface_checks.json').write_text(json.dumps(result,indent=2)+'\n')
(RUN/'site/surface_checks.json').write_text(json.dumps(result,indent=2)+'\n')
for r in rows:
    if not r['passed']:print(r['group'],r['pair'],r['camera'],{s:{k:v['proper_crossings'] for k,v in d.items()} for s,d in r['sides'].items()})
