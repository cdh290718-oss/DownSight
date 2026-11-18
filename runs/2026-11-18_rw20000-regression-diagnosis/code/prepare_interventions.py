"""Reproduce coordinate-only counterfactual inputs; --check validates executed inputs."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

RUN=Path(__file__).resolve().parents[1];ROOT=RUN.parents[1]
CONDITIONS={'baseline':[], 'old_all':['*'],'old_right_wrist':['R_Wrist'],
    'old_shoulders':['R_Shoulder','L_Shoulder'],'old_hips':['R_Hip','L_Hip'],
    'old_right_leg':['R_Hip','R_Knee','R_Ankle'],'old_right_knee_ankle':['R_Knee','R_Ankle'],
    'old_left_arm':['L_Shoulder','L_Elbow','L_Wrist'],'old_wrists':['R_Wrist','L_Wrist']}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--check',action='store_true');args=ap.parse_args()
    source=ROOT/'outputs/rw_a20000_evaluation_20261118/d_fit/prepared.json'
    rows=json.loads(source.read_text())['items']
    old={r['pair']:r for r in rows if r['condition']=='rw5000'}
    new={r['pair']:r for r in rows if r['condition']=='rw20000'}
    items=[]
    for pair,template in new.items():
        for condition,joints in CONDITIONS.items():
            if condition in ['old_right_leg','old_right_knee_ankle','old_left_arm','old_wrists'] and pair!='body07':continue
            item=copy.deepcopy(template);item['condition']=condition
            for pt,op in zip(item['points'],old[pair]['points']):
                assert pt['name']==op['name']
                if '*' in joints or pt['name'] in joints:pt['xy']=copy.deepcopy(op['xy'])
            check=copy.deepcopy(item);check['condition']=template['condition']
            for pt,orig in zip(check['points'],template['points']):pt['xy']=orig['xy']
            assert check==template
            items.append(item)
    payload=json.dumps(dict(items=items,protocol_sha256=hashlib.sha256((RUN/'protocol.json').read_bytes()).hexdigest()),indent=2)+'\n'
    target=ROOT/'outputs/rw20000_regression_diagnosis_20261118/d_fit/prepared.json'
    if args.check:assert target.read_text()==payload
    else:
        assert not target.exists();target.parent.mkdir(parents=True,exist_ok=True);target.write_text(payload)
    print(json.dumps(dict(instances=len(items),exact_payload_match=args.check,sha256=hashlib.sha256(payload.encode()).hexdigest())))

if __name__=='__main__':main()
