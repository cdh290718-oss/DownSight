"""Create isolated B/C configs from frozen no-margin inputs; reuse A results."""
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'outputs/hand_stage_ablation_20261104'
SOURCE = ROOT/'outputs/session_hand6_v2_no_margin_rerun_20261104/prepared/inputs.json'

def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')

if __name__ == '__main__':
    OUT.mkdir(exist_ok=False)
    source=json.loads(SOURCE.read_text())
    records=[r for r in source['records'] if r['pair'].startswith('new')]
    assert len(records)==6 and {r['pair'] for r in records}=={'new_001','new_022','new_100'}
    write(OUT/'A/inputs.json',dict(records=records))
    comparisons=[]
    for group,mode in [('B','body_rigid'),('C','arm_routed')]:
        fresh=[]
        for r in records:
            cfg=json.loads(Path(r['config']).read_text())
            changed=copy.deepcopy(cfg)
            changed['v2_hand_stage']=mode
            changed['output']=str(OUT/group/'fits'/r['id'])
            config=OUT/group/'configs'/(r['id']+'.json')
            write(config,changed)
            new=copy.deepcopy(r);new['config']=str(config);fresh.append(new)
            assert {k for k in changed if changed[k]!=cfg.get(k)}=={'v2_hand_stage','output'}
            comparisons.append(dict(group=group,frame=r['id'],mode=mode,changed_config_keys=['v2_hand_stage','output']))
        write(OUT/group/'inputs.json',dict(records=fresh))
    write(ROOT/'runs/2026-11-04_hand-stage-ablation/evidence/input_comparison.json',dict(
        A='Reused completed no-margin v2 results; unchanged inputs',
        comparisons=comparisons,network_rerun=False,old_pairs_included=False))
