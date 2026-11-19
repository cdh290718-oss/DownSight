"""Test actual RW20000 predictions; fit a single foot readout on the existing validation split."""
import copy
import runpy
import sys
from pathlib import Path
import numpy as np
import study
from study import OUT, ROOT, RUN, OLD, common, read, write, sha

DEST=OUT/'automatic'
meta_path=ROOT.parent/'EgoSMPLX/official_baseline/sapiens2/sapiens/pose/configs/_base_/keypoints308.py'


def prepare():
    indices={v['name']:i for i,v in runpy.run_path(str(meta_path))['dataset_info']['keypoint_info'].items()}
    assert len(indices)==308
    pred_dir=ROOT/'outputs/rw_a20000_evaluation_20261117'
    val_path=pred_dir/'rw20000_validation_predictions.json'
    diag_path=pred_dir/'rw20000_diagnostic_predictions.json'
    val_ref_path=ROOT/'outputs/rw_a20000_dataset_20261117/validation.json'
    val={x['id']:x for x in read(val_ref_path)['records']}
    diag={x['id']:x for x in read(diag_path)}
    ds=ROOT.parent/'EgoSMPLX/data/UnrealEgo_RW_Train/prepared'
    rw_names=read(ds/'metadata.json')['train_joint_names']
    def segment(xy,side):
        heel=xy[indices[side+'_heel']]
        toes=xy[[indices[side+'_big_toe'],indices[side+'_small_toe']]].mean(0)
        return heel,toes-heel
    num,den,n=0.,0.,0
    with np.load(ds/'samples.npz') as z:
        xyval=z['keypoints2d']
    subjects=set()
    for row in read(val_path):
        r=val[row['id']]
        assert r['split'] in ['val','validation'] and r['subject']!='S9' and r['image_sha256']==row['image_sha256']
        subjects.add(r['subject'])
        for letter,side in [('L','left'),('R','right')]:
            target=xyval[r['source_index'],rw_names.index(letter+'_Toe'),:2]
            if not ((target>=0)&(target<872)).all(): continue
            a,d=segment(np.array(row['xy']),side)
            num+=d@(target-a);den+=d@d;n+=1
    fraction=float(np.clip(num/den,0,1))
    masks=read(OUT/'prepared.json')['items']
    archive={s['id']:s for s in read(common.ARCH/'body_samples.json')['samples']}
    mapping={m['name']:m for m in read(ROOT/'runs/2026-11-16_sapiens-unrealego-audit/evidence/audit.json')['joint_mapping']}
    errors=[];items=[]
    cfg=read(OUT/'protocol.json')
    for ref in masks:
        if ref['condition'] not in ['baseline','15history']: continue
        item=copy.deepcopy(ref)
        obs=read(OLD/item['pair']/'observations_local.json')
        for i,side in enumerate(['left','right']):
            sid=item['pair']+'_'+side
            record=diag[sid]
            assert record['image_sha256']==archive[sid]['image_sha256']
            xy=np.asarray(record['xy'])
            adapter=read(common.camera_files()[i])['input_adapter']
            for p in item['points']:
                name=p['name']
                if name=='Neck':
                    raw=xy[indices['neck']]
                elif name.endswith('_Toe'):
                    a,d=segment(xy,'left' if name[0]=='L' else 'right')
                    raw=a+fraction*d
                else:
                    letter,part=name.split('_')
                    key=('left' if letter=='L' else 'right')+'_'+part.lower()
                    raw=xy[mapping[key]['sapiens308_id']]
                assert np.isfinite(raw).all()
                adapted=raw*adapter['scale']+adapter['offset']
                target=np.array(obs['views'][i][obs['names'].index(name)][:2])
                p['xy'][i]=adapted.tolist()
                if ref['condition']=='15history':
                    errors.append(dict(pair=item['pair'],view=side,joint=name,
                        original_px=float(np.linalg.norm(adapted-target)/adapter['scale'])))
        items.append(item)
    cfg['conditions']=['baseline','15history']
    write(DEST/'protocol.json',cfg)
    write(DEST/'prepared.json',dict(items=items))
    write(RUN/'evidence/automatic_inputs.json',dict(predictor='RW20000, archived 308 channels, no new inference',
        hashes={p.name:sha(p) for p in [val_path,diag_path,val_ref_path,meta_path]},
        formula='heel + bounded least-squares fraction * (mean(bigToe,smallToe) - heel); one shared scalar, calibration on validation split; raw detected neck unchanged',
        foot_fraction=fraction,unbounded_fraction=float(num/den),foot_calibration_points=n,
        validation_subjects=sorted(subjects),source_sizes_px='872 original; both losses use frozen image adapter.',
        errors=errors,notes='All four S9 pairs pass reference in-frame mask. Not an independent visibility/deployment test.'))


if __name__=='__main__':
    prepare()
    study.OUT=DEST
    study.optimize()
