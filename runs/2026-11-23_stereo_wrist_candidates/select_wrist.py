"""Calibrate the image rule on validation subjects only; freeze prior to test inference."""
import sys
from itertools import product
from shared import *

ARMS=[(16,18,20),(17,19,21)]

def rank(choices,rule):
    if rule is None or len(choices)==1 or choices[0]['photo'] is None: return 0
    candidates=[]
    threshold,weight,gap=rule
    for i,c in enumerate(choices):
        if c['photo'] is None or c['photo'][1]<4: continue
        score=c['photo'][0]+weight*max(0,c['loss']-choices[0]['loss'])
        candidates.append((score,i))
    if not candidates: return 0
    best,i=min(candidates)
    if i==0 or choices[0]['photo'][0]-best<threshold: return 0
    other=[s for s,j in candidates if abs(choices[i]['depth']-choices[j]['depth'])>=.035]
    return 0 if other and min(other)-best<gap else i

def choose(rows,rule):
    return {p:[rank(c,rule) for c in row['arms']] for p,row in rows.items()}

def arm_errors(rows,truths):
    cases=[]
    for pair,row in rows.items():
        for choices,(s,e,w) in zip(row['arms'],ARMS):
            gt=np.array(truths[pair]['gt'])
            ew=[ref.common.NATIVE_IDS.index(n) for n in [e,w]]
            distances=np.array([[np.linalg.norm(np.array(c[k])-gt[q])*1000 for k,q in zip(['elbow','wrist'],ew)] for c in choices])
            cases.append((row['subject'],choices,distances))
    return cases

def assess(cases,rule):
    w2=[]; initial=[]; diffs=[]; elbows=[];changed=0
    for _,choices,d in cases:
        idx=rank(choices,rule)
        w2.append(d[idx,1]**2);initial.append(d[0,1]**2)
        elbows.append(d[idx,0]-d[0,0])
        diffs.append(d[idx].sum()-d[0].sum());changed+=idx!=0
    return dict(wrists=len(w2),changed=changed,wrist_rmse=float(np.sqrt(np.mean(w2))),
                baseline_wrist_rmse=float(np.sqrt(np.mean(initial))),
                mpjpe_change_mm=float(np.mean(diffs)/6),other10_change_mm=float(np.mean(elbows)/5))

def freeze():
    rows=read(OUT/'validation_candidates.json')
    cases=arm_errors(rows,read(OUT/'validation_truth.json'))
    tune=[c for c in cases if c[0] in ['S3','S5']]
    held=[c for c in cases if c[0] in ['S13','S15']]
    proposals=[]
    for rule in product([.03,.10,.20],[0,10,50],[0,.03]):
        result=assess(tune,rule)
        proposals.append(dict(rule=list(rule),tune=result))
    eligible=[p for p in proposals if p['tune']['changed'] and p['tune']['other10_change_mm']<=.25 and p['tune']['mpjpe_change_mm']<=0]
    # Keep a real challenger for diagnosis even if its performance fails the release gate.
    best=min(eligible or proposals,key=lambda p:p['tune']['wrist_rmse'])
    rule=best['rule']
    held_result=assess(held,rule)
    def pass_gate(result):
        return result['wrist_rmse']<result['baseline_wrist_rmse'] and result['mpjpe_change_mm']<=0 and result['other10_change_mm']<=.25
    active=pass_gate(best['tune']) and pass_gate(held_result)
    write(RUN/'evidence/frozen_rule.json',dict(rule=rule,active=active,grid=proposals,
          tuning=best['tune'],confirmation=held_result,tuning_subjects=['S3','S5'],confirmation_subjects=['S13','S15'],
          candidates_sha256=sha(OUT/'validation_candidates.json'),selector_sha256=sha(Path(__file__)),
          generator_sha256=sha(RUN/'candidates.py')))
    print('rule',rule,'confirmed',active,best['tune'],held_result,flush=True)

def predict(split):
    lock=read(RUN/'evidence/frozen_rule.json')
    assert lock['selector_sha256']==sha(Path(__file__)) and lock['generator_sha256']==sha(RUN/'candidates.py')
    rows=read(OUT/(split+'_candidates.json'))
    proposed=choose(rows,lock['rule'])
    original={p:[0,0] for p in rows}
    objective={p:[min(range(len(c)),key=lambda i:c[i]['loss']) for c in row['arms']] for p,row in rows.items()}
    write(OUT/(split+'_predictions.json'),dict(baseline=original,objective=objective,image=proposed,
          gated=proposed if lock['active'] else original))
    write(RUN/('evidence/'+split+'_prediction.json'),dict(choices_sha256=sha(OUT/(split+'_predictions.json')),
          frozen_rule_sha256=sha(RUN/'evidence/frozen_rule.json'),pairs=len(rows)))
if __name__=='__main__':
    if sys.argv[1]=='freeze': freeze()
    else: predict(sys.argv[1])
