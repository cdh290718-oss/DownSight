"""Evaluate after fixed-budget fits; never use 3D error to select checkpoints."""
import collections
import numpy as np,torch
from common import ROOT,RUN,OUT,EVIDENCE,NAMES,MODEL_IDS,NATIVE_IDS,OBS,read,write,sha,setup_model
from geometry import BodyKinematics,rotation

def align(x,y):
    x,y=np.asarray(x,float),np.asarray(y,float)
    xc,yc=x-x.mean(0),y-y.mean(0)
    u,s,vh=np.linalg.svd(xc.T@yc)
    d=np.array([1.,1.,1. if np.linalg.det(vh.T@u.T)>0 else -1.])
    R=(vh.T*d)@u.T;scale=np.sum(s*d)/np.sum(xc*xc)
    assert np.isfinite(scale) and scale>0
    return scale*xc@R.T+y.mean(0)

def main():
    done=read(OUT/'fit_complete.json');assert done['exit_status']==0
    assert done['protocol_sha256']==sha(RUN/'protocol.json')
    items=read(OUT/'prepared.json')['items'];ids=done['ids']
    assert [x[0] for x in ids]==[i['pair'] for i in items]
    refs=read(OUT/'reference.json');p=dict(np.load(OUT/'budget_20000.npz'));diag=read(OUT/'diagnostic_20000.json');assert diag['ids']==ids
    model,_=setup_model();torch.set_num_threads(2)
    layer=model.layer['neutral'].cpu().eval();layer.requires_grad_(False)
    assert [model.joint_idx[j] for j in MODEL_IDS]==NATIVE_IDS
    fk=BodyKinematics(layer);pred=[];equivalence=[]
    for start in range(0,len(items),10):
        end=min(start+10,len(items));b=end-start
        t=lambda k:torch.tensor(p[k][start:end],dtype=torch.float32)
        fixed={k:torch.zeros(b,n) for k,n in [('left_hand_pose',45),('right_hand_pose',45),('jaw_pose',3),('leye_pose',3),('reye_pose',3),('expression',10)]}
        with torch.no_grad():
            full=layer(body_pose=t('body_pose').reshape(b,-1),betas=t('betas'),global_orient=t('global0'),**fixed)
            rd=rotation(t('delta'));pivot=full.joints[:,:1]
            native=(full.joints-pivot)@rd.transpose(1,2)+pivot+t('translation')[:,None]
            fast,_=fk(t('body_pose'),t('betas'),t('global0'),t('delta'),t('translation'))
            diff=float((native[:,:22]-fast).abs().max());assert diff<3e-6
            equivalence.append(diff);pred.extend(native[:,NATIVE_IDS].numpy())
    records=[]
    for i,item in enumerate(items):
        ref=refs[item['base_pair']]
        rec=dict(pair=item['base_pair'],model=item['detector'],subject=ref['subject'],sequence=ref['sequence'],frame=ref['frame'],selected_step=int(p['selected_step'][i]),convergence_pass=bool(diag['convergence_pass'][i]),relative_loss_change=diag['relative_loss_change'][i],max_joint_motion_mm=diag['max_joint_motion_mm'][i])
        if p['selected_step'][i]<0 or not diag['selected_feasible'][i]:
            rec.update(status='failed',reason='no feasible selected checkpoint');records.append(rec);continue
        x=np.asarray(pred[i],float);y=np.array(ref['gt']);assert x.shape==y.shape==(15,3)
        assert np.isfinite(x).all() and np.isfinite(y).all()
        absolute=np.linalg.norm(x-y,axis=1)*1000;pa=np.linalg.norm(align(x,y)-y,axis=1)*1000
        # Mid-hip translation alignment is supplementary, not the main non-PA metric.
        xp=x-x[[7,11]].mean(0);yp=y-y[[7,11]].mean(0)
        rr=np.linalg.norm(xp-yp,axis=1)*1000
        pa12=np.linalg.norm(align(x[OBS],y[OBS])-y[OBS],axis=1)*1000
        rec.update(status='ok',absolute_per_joint_mm=absolute.tolist(),pa_per_joint_mm=pa.tolist(),absolute_mpjpe_mm=float(absolute.mean()),pa_mpjpe_mm=float(pa.mean()),root_relative_mpjpe_mm=float(rr.mean()),absolute_12_mpjpe_mm=float(absolute[OBS].mean()),pa_12_mpjpe_mm=float(pa12.mean()),absolute_rmse_mm=float(np.sqrt(np.mean(absolute**2))),training_loss=float(p['loss'][i]))
        records.append(rec)
    metric_names=['absolute_mpjpe_mm','pa_mpjpe_mm','root_relative_mpjpe_mm','absolute_12_mpjpe_mm','pa_12_mpjpe_mm','absolute_rmse_mm']
    summary={}
    for m in ['original','rw500','rw5000','rw20000']:
        allr=[r for r in records if r['model']==m];ok=[r for r in allr if r['status']=='ok']
        subjects={}
        for s in sorted({r['subject'] for r in allr}):
            selected=[r for r in ok if r['subject']==s]
            subjects[s]=dict(count=len(selected),**{k:float(np.mean([r[k] for r in selected])) if selected else None for k in metric_names})
        summary[m]=dict(total=len(allr),success=len(ok),failed=len(allr)-len(ok),converged=sum(r['convergence_pass'] for r in ok),**{k:float(np.mean([r[k] for r in ok])) if ok else None for k in metric_names},subjects=subjects)
    # Paired comparison uses the intersection, with all failures retained above.
    bymodel={m:{r['pair']:r for r in records if r['model']==m and r['status']=='ok'} for m in summary}
    common=set.intersection(*[set(d) for d in bymodel.values()]);paired={m:{k:float(np.mean([d[p][k] for p in common])) if common else None for k in metric_names} for m,d in bymodel.items()}
    write(EVIDENCE/'results.json',dict(command='<BODY_PYTHON> evaluate.py',exit_status=0,joint_names=NAMES,records=records,full_vs_fast_fk_max_m=max(equivalence),reference_local_sha256=sha(OUT/'reference.json'),parameter_local_sha256=sha(OUT/'budget_20000.npz')))
    write(EVIDENCE/'summary.json',dict(summary=summary,common_success_pairs=len(common),paired=paired,fit_elapsed_seconds=done['elapsed_seconds'],fit_steps=done['steps'],protocol_sha256=done['protocol_sha256'],metrics='non-PA: original left-camera coordinates, no alignment; PA: per-pair similarity with translation, proper rotation and scale, all GT15 joints; toes compared with heel proxies. 12-joint subset removes neck/toes; PA re-estimated on subset.',exit_status=0))
    np.savez_compressed(OUT/'predicted_gt15.npz',joints_m=np.array(pred),ids=np.array([i['pair'] for i in items]))
    print(summary,flush=True)

if __name__=='__main__':main()
