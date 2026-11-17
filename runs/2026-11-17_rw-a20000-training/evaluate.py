"""Full SMPL-X evaluation after every fit has finished; raw coordinates stay local."""
import sys

import numpy as np
import torch

from common import ARCH, EVIDENCE, MODEL_IDS, NAMES, NATIVE_IDS, OUT, RUN, read, setup_model, sha, write
from geometry import BodyKinematics, rotation


def align(predicted,target):
    x,y=np.asarray(predicted,float),np.asarray(target,float)
    mx,my=x.mean(0),y.mean(0)
    x0,y0=x-mx,y-my
    u,s,vh=np.linalg.svd(x0.T@y0)
    correction=np.array([1.,1.,1. if np.linalg.det(vh.T@u.T)>0 else -1.])
    r=(vh.T*correction)@u.T
    scale=np.sum(s*correction)/np.sum(x0*x0)
    assert np.isfinite(scale) and scale>0
    return scale*x0@r.T+my


def main():
    complete=read(OUT/'fit_complete.json')
    assert complete['exit_status']==0
    cfg=read(RUN/'protocol.json')
    assert complete['protocol_sha256']==sha(RUN/'protocol.json')
    ids=complete['ids']
    # This is deliberately the first body-3D GT read in the new experiment.
    gtfile=ARCH/'body_gt_official.npz'
    with np.load(gtfile) as gt:
        ground_truth=gt['joints_m'].astype(float)
    assert len(ground_truth)>=20
    model,_=setup_model();layer=model.layer['neutral'].cpu().eval();layer.requires_grad_(False)
    assert [model.joint_idx[j] for j in MODEL_IDS]==NATIVE_IDS
    fk=BodyKinematics(layer)
    torch.set_num_threads(2)
    records=[];equivalence=[]
    for budget in cfg['budgets']:
        path=OUT/('budget_'+str(budget)+'.npz')
        with np.load(path) as src:
            p={k:src[k] for k in src.files}
        diag=read(OUT/('diagnostic_'+str(budget)+'.json'))
        assert ids==diag['ids']
        predictions=[]
        for start in range(0,len(ids),10):
            end=min(start+10,len(ids));b=end-start
            t=lambda key:torch.tensor(p[key][start:end],dtype=torch.float32)
            fixed={key:torch.zeros(b,n) for key,n in [('left_hand_pose',45),('right_hand_pose',45),
                                                   ('jaw_pose',3),('leye_pose',3),('reye_pose',3),('expression',10)]}
            with torch.no_grad():
                full=layer(body_pose=t('body_pose').reshape(b,-1),betas=t('betas'),
                           global_orient=t('global0'),**fixed)
                rd=rotation(t('delta'));pivot=full.joints[:,:1]
                native=(full.joints-pivot)@rd.transpose(1,2)+pivot+t('translation')[:,None]
                fast,_=fk(t('body_pose'),t('betas'),t('global0'),t('delta'),t('translation'))
                difference=float((native[:,:22]-fast).abs().max())
                assert difference<3e-6,difference
                equivalence.append(dict(budget=budget,batch_start=start,body_joint_max_difference_m=difference))
                predictions.extend(native[:,NATIVE_IDS].numpy())
        for k,(pair,condition,method) in enumerate(ids):
            if p['selected_step'][k]<0:
                records.append(dict(pair=pair,condition=condition,method=method,budget=budget,status='failed',
                                    reason='No feasible checkpoint in the training budget'))
                continue
            prediction=np.asarray(predictions[k],float)
            target=ground_truth[(int(pair[-2:])-1)*2]
            assert prediction.shape==target.shape==(15,3)
            absolute=np.linalg.norm(prediction-target,axis=1)*1000
            pa=np.linalg.norm(align(prediction,target)-target,axis=1)*1000
            records.append(dict(pair=pair,condition=condition,method=method,budget=budget,status='ok',
                                absolute_per_joint_mm=absolute.tolist(),pa_per_joint_mm=pa.tolist(),
                                absolute_mean_mm=float(absolute.mean()),pa_mean_mm=float(pa.mean()),absolute_rmse_mm=float(np.sqrt(np.mean(absolute**2))),
                                selected_step=int(p['selected_step'][k]),training_loss=float(p['loss'][k]),
                                convergence_pass=bool(diag['convergence_pass'][k]),
                                selected_feasible=bool(diag['selected_feasible'][k]),
                                relative_loss_change=diag['relative_loss_change'][k],
                                max_joint_motion_mm=diag['max_joint_motion_mm'][k]))
    write(EVIDENCE/'results.json',dict(
        command='<BODY_PYTHON> runs/2026-11-17_rw-a20000-training/evaluate.py',exit_status=0,
        joint_names=NAMES,records=records,ground_truth_sha256=sha(gtfile),
        fit_complete_local_sha256=sha(OUT/'fit_complete.json'),
        full_model_body_joint_equivalence=equivalence,
        scope='Reference body3D read only after fitting; Head and calibration annotation-derived; Toe uses heel proxy.'))
    print('Evaluated',len(records),'records; failures',sum(r['status']!='ok' for r in records),flush=True)


if __name__=='__main__':
    main()
