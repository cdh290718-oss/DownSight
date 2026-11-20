"""Check the public manifests and recompute the matched frozen-set comparison."""
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
import subprocess

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]


def read(path):
    return json.loads((ROOT / path).read_bytes())


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    current = read('configs/golden_d.current.json')
    previous = read(current['previous_manifest'])
    assert (previous['method_id'],current['method_id']) == ('golden-d-2d-v1','golden-d-2d-v2')
    for version in [previous,current]:
        for row in version['locked_files']:
            data = (ROOT / row['path']).read_bytes()
            assert digest(data) == row['sha256']
            pinned = subprocess.check_output(['git','show',version['source_commit']+':'+row['path']],cwd=ROOT)
            assert data == pinned
    assert current['conditions'] == ['rw20000']
    settings = read(current['protocol'])
    assert current['step_budget'] == settings['budgets'][0] == 20000
    assert settings['head_rotation_weight'] == 0 and settings['head_position_weight'] > 0
    assert settings['methods'] == current['methods'] == ['2d']
    assert settings['shoulder_weight'] == current['reprojection_groups']['shoulder'] == .25
    old = Path('runs/2026-11-18_rw_frozen200_golden_d/evidence')
    new = Path(current['reference_results']).parent
    first_inputs = read(old/'preparation.json')
    inputs = read(new/'inputs.json')
    assert first_inputs['prediction_sha256']['rw20000'] == inputs['prediction_sha256']['rw20000']
    assert first_inputs['prepared_local_sha256'] == inputs['source_sha256']
    assert first_inputs['reference_local_sha256'] == read(new/'execution.json')['reference_sha256']
    def keyed(path,field):
        records = [r for r in read(path)[field] if r['model']=='rw20000']
        by_pair = {r['pair']:r for r in records}
        assert len(by_pair) == len(records) == 100
        return by_pair
    before, after = keyed(old/'results.json','records'), keyed(new/'results.json','rows')
    assert before.keys() == after.keys()
    assert all(before[k]['status']=='ok' and after[k]['feasible'] for k in before)
    old_summary = read(old/'summary.json')['summary']['rw20000']
    new_summary = read(current['reference_results'])['overall']['rw20000']
    comparison = {}
    for old_key,new_key in [('absolute_12_mpjpe_mm','mpjpe12_mm'),('pa_12_mpjpe_mm','pa12_mm')]:
        a=mean(r[old_key] for r in before.values())
        b=mean(r[new_key] for r in after.values())
        assert math.isclose(a,old_summary[old_key]) and math.isclose(b,new_summary[new_key])
        comparison[new_key] = dict(v1_rw20000=a,v2_rw20000=b,delta=b-a)
    result = dict(base_commit='7aa682a50658ac1c84adc22122fc84b5cb0d86c4',
                  command='python -B runs/2026-11-20_golden_d_v2/verify.py',exit_status=0,
                  v1_manifest_sha256=digest((ROOT/current['previous_manifest']).read_bytes()),
                  v2_manifest_sha256=digest((ROOT/'configs/golden_d.current.json').read_bytes()),
                  locked_files={v['method_id']:len(v['locked_files']) for v in (previous,current)},
                  old_reference_detector=previous['reference_detector'],new_detector=read(current['detector_assets'])['sha256']['rw20000'],
                  predictions_sha256=inputs['prediction_sha256']['rw20000'],common_pairs=len(after),
                  comparison=comparison,metrics_v2=new_summary,settings=settings)
    (RUN/'evidence/verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(comparison))


if __name__ == '__main__':
    main()
