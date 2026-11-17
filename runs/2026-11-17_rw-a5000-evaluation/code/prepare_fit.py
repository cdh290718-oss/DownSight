"""Change detector coordinates only; preserve the current golden D nuisance inputs."""
import copy
import hashlib
import json
from pathlib import Path

RUN=Path(__file__).resolve().parents[1];ROOT=RUN.parents[1]
OUT=ROOT/'outputs/rw_a5000_evaluation_20261117'
ids={'R_Shoulder':6,'R_Elbow':8,'R_Wrist':41,'L_Shoulder':5,'L_Elbow':7,'L_Wrist':62,
     'R_Hip':10,'R_Knee':12,'R_Ankle':14,'L_Hip':9,'L_Knee':11,'L_Ankle':13}
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    source=ROOT/'outputs/newdetector_d10_20261117/prepared.json'
    base=read(source)['items'];assert len(base)==10
    for item in base:assert item['condition']=='baseline' and all(not p['accepted'] for p in item['points'])
    rows=[];identity=[]
    for version in ['original','rw500','rw5000']:
        pred={r['id']:r for r in read(OUT/(version+'_diagnostic_predictions.json'))}
        assert len(pred)==20
        for template in base:
            item=copy.deepcopy(template);item['condition']='baseline' if version=='original' else version
            for p in item['points']:
                for v,side in enumerate(['left','right']):
                    d=pred[item['pair']+'_'+side];x=d['xy'][ids[p['name']]]
                    p['xy'][v]=[x[0]*(720/872)+279.91284403669727,x[1]*(720/872)-0.08715596330275227]
                    p['scores'][v]=d['score'][ids[p['name']]]
            # Every other field must exactly match the already executed golden D reference.
            nuisance=copy.deepcopy(item);nuisance['condition']='baseline'
            for old,p in zip(template['points'],nuisance['points']):p['xy']=old['xy'];p['scores']=old['scores']
            assert nuisance==template
            identity.append(dict(version=version,pair=item['pair'],nuisance_inputs_identical=True))
            rows.append(item)
    target=OUT/'d_fit';target.mkdir(exist_ok=False)
    (target/'prepared.json').write_text(json.dumps(dict(items=rows,protocol_sha256=sha(RUN/'protocol.json')),indent=2)+'\n')
    reference=read(ROOT/'runs/2026-11-17_newdetector_d10/protocol.json');current=read(RUN/'protocol.json')
    operational=['methods','pairs','sigma_px_fixed','metric_loss_weight','metric_huber_delta','shoulder_weight',
        'head_position_sigma_m','head_rotation_sigma_deg','head_position_weight','head_rotation_weight',
        'pose_prior_weight','shape_prior_weight','shape_information_threshold','minimum_observed_joint_z_m',
        'pose_lr','rotation_lr','translation_lr','shape_lr','budgets','checkpoint_steps','trace_steps','lr_schedule',
        'selection','convergence','absolute_pose_caps_deg']
    assert all(reference[k]==current[k] for k in operational)
    (RUN/'evidence/fit_preparation.json').write_text(json.dumps(dict(command='python code/prepare_fit.py',exit_status=0,
        source_prepared_sha256=sha(source),prepared_sha256=sha(target/'prepared.json'),protocol_sha256=sha(RUN/'protocol.json'),
        operational_protocol_identical_fields=operational,identity_checks=identity,
        conditions={'baseline':'original','rw500':'500-image full epoch1','rw5000':'5000-image full epoch1'},
        scope='same prior S9 ten pairs, mean shape, annotation-derived Head/calibration/masks; only body XY and diagnostic scores changed'),indent=2)+'\n')
    print('Prepared',len(rows),'controlled full-2D fits',flush=True)
if __name__=='__main__':main()
