"""Score the observed arm pixels in the gallery's original image coordinates."""
from pathlib import Path
import json
import numpy as np

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
def read(path):
    return json.loads(path.read_text())

rows = read(ROOT/'runs/2026-11-23_fixed_floor24/evidence/gallery.json')['rows']
joint_ids = {'L_Shoulder':5,'R_Shoulder':6,'L_Elbow':7,'R_Elbow':8,'L_Wrist':62,'R_Wrist':41}
out = {}
for split, prepared, manifest, condition in [
        ('test','rw_frozen200_free12_20261119','rw_test_completion_20261117/test_manifest.json','rw20000'),
        ('validation','stereo_wrist_candidates_20261123/validation_fit','rw_a20000_dataset_20261117/validation.json','baseline')]:
    obs = {r['pair']:r for r in read(ROOT/'outputs'/prepared/'prepared.json')['items'] if r['condition']==condition}
    views = {r['image_sha256']:r for r in read(ROOT/'outputs'/manifest)['records']}
    for row in rows:
        pair = row['pair']
        if row['split'] != split or pair not in ['pair055','pair068','val055','val031']:
            continue
        result = out[pair] = {}
        for eye, side in enumerate(['left','right']):
            image = views[row['image_sha256'][eye]]
            assert (image['sequence'],image['frame'],image['view'])==(row['sequence'],row['frame'],side)
            a = read(ROOT/'outputs/rw_frozen200_golden_d_20261118'/('camera_'+side+'.json'))['input_adapter']
            for point in obs[pair]['points']:
                name = point['name']
                if name in joint_ids and point['valid'][eye]:
                    target = np.asarray(image['keypoints308_xy'][joint_ids[name]])
                    pixels = (np.array(point['xy'][eye])-a['offset'])/a['scale']
                    result.setdefault(name,{})[side] = float(np.linalg.norm(pixels-target))
(RUN/'evidence/pixels.json').write_text(json.dumps(out,indent=2)+'\n')
for case, values in out.items():
    print(case, {n:[round(x,1) for x in v.values()] for n,v in values.items()})
