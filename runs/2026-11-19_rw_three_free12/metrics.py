"""Evaluate numerical joint errors only; do not export geometry or make images."""
import numpy as np
import torch
from experiment import RUN,ROOT,OUT,REF,MODELS,common,read,write,sha
from evaluate import align
from geometry import rotation

E=RUN/'evidence'
FIELDS=['mpjpe12_mm','pa12_mm','legacy15_mm','legacy15_pa_mm','mapped15_mm','mapped15_pa_mm']
RMSE=['fit_vs_detection','fit_vs_gt','detection_vs_gt']


def joint_errors(pred,truth):
    return [float(np.mean(np.linalg.norm(a-truth,axis=1))*1000) for a in (pred,align(pred,truth))]


def main():
    conds=read(OUT/'fit_complete.json')
    diag=read(OUT/'diagnostic_20000.json')
    config=read(OUT/'protocol.json')
    assert conds['exit_status']==0 and all(diag['selected_feasible'])
    model,camera_cls=common.setup_model()
    layer=model.layer['neutral'].cpu().eval()
    cams=[camera_cls(p) for p in common.camera_files()]
    adapters=[read(p)['input_adapter'] for p in common.camera_files()]
    gtfile=common.ARCH/'body_gt_official.npz'
    items=read(OUT/'prepared.json')['items']
    rows=[]
    with np.load(gtfile) as gt,np.load(OUT/'budget_20000.npz') as fit:
        for i,(pair,version,_) in enumerate(conds['ids']):
            item=items[i]
            assert (pair,version)==(item['pair'],item['condition'])
            poses=fit['body_pose'][i].reshape(21,3)
            assert not np.any(poses[[11,14]]) and not np.any(fit['betas'][i])
            tensor=lambda a: torch.as_tensor(a,dtype=torch.float32).reshape(1,-1)
            extra={f:torch.zeros(1,n) for f,n in [('left_hand_pose',45),('right_hand_pose',45),('jaw_pose',3),('leye_pose',3),('reye_pose',3),('expression',10)]}
            with torch.no_grad():
                j=layer(global_orient=tensor(fit['global0'][i]),body_pose=tensor(poses),betas=tensor(fit['betas'][i]),**extra).joints[0]
                root=j[:1]
                j=((j-root)@rotation(tensor(fit['delta'][i]))[0].T+root+tensor(fit['translation'][i])).numpy()
            original=j[common.NATIVE_IDS]
            mapped=original.copy()
            for index,a,b,ratio in zip([0,10,14],[15,8,7],[12,11,10],config['landmark_ratios']):
                mapped[index]=j[a]+ratio*(j[b]-j[a])
            idx=int(pair[-2:])*2-2
            assert gt['ids'][idx:idx+2].tolist()==[pair+'_left',pair+'_right']
            target=gt['joints_m'][idx].astype(float)
            e=joint_errors(original[common.OBS],target[common.OBS])+joint_errors(original,target)+joint_errors(mapped,target)
            assert [p['name'] for p in item['points']]==[common.NAMES[n] for n in common.OBS]
            views=[]
            for v,side in enumerate(['left','right']):
                a=adapters[v]
                raw=lambda xy:(np.asarray(xy)-a['offset'])/a['scale']
                T=np.asarray(item['inputs']['views'][side]['camera']['left_to_view'])
                uv=raw(cams[v].world2camera(original[common.OBS]@T[:3,:3].T+T[:3,3]))
                detection=raw([p['xy'][v] for p in item['points']])
                reference=gt['joints2d'][idx+v,common.OBS].astype(float)
                mask=np.array([p['valid'][v] for p in item['points']],bool)
                diffs=[uv-detection,uv-reference,detection-reference]
                views.append(dict(side=side,points=int(mask.sum()),squared_error_px2={k:float((d[mask]**2).sum()) for k,d in zip(RMSE,diffs)}))
            rows.append(dict(pair=pair,model=MODELS[version],**dict(zip(FIELDS,e)),views=views,converged=diag['convergence_pass'][i]))
    summary={}
    for model in MODELS.values():
        records=[r for r in rows if r['model']==model]
        assert len(records)==10
        v=[eye for rec in records for eye in rec['views']]
        n=sum(t['points'] for t in v)
        summary[model]=dict(**{f:np.mean([r[f] for r in records]) for f in FIELDS},
            **{k+'_rmse_px':np.sqrt(sum(t['squared_error_px2'][k] for t in v)/n) for k in RMSE},points=n,
            converged=sum(r['converged'] for r in records))
    previous=read(ROOT/'runs/2026-11-19_rw_auto_history/evidence/results_auto.json')['rows']
    diffs=[]
    for r in previous:
        if r['condition']=='body':
            this=next(s for s in rows if s['pair']==r['pair'] and s['model']=='rw20000')
            diffs.extend([abs(this[f]-r[g]) for f,g in [('mapped15_mm','mapped15_nonpa_mm'),('mapped15_pa_mm','mapped15_pa_mm')]])
    assert len(diffs)==8 and max(diffs)<.2,diffs
    write(E/'results.json',dict(rows=rows,gt_sha256=sha(gtfile),fit_npz_sha256=sha(OUT/'budget_20000.npz')))
    # Original report's larger 2D-only splits: RMSE, not its mean distance.
    errors=read(REF/'evidence/2d_errors.json')['records']
    expanded={}
    for model in MODELS.values():
        expanded[model]={}
        for split in ['validation','same_person','test_available']:
            subset=[s for s in errors if s['version']==model and s['dataset']==split]
            vals=[v for s in subset for v in s['per_joint_error_px'] if v is not None]
            assert len(subset)=={'validation':400,'same_person':100,'test_available':46}[split]
            expanded[model][split]=dict(images=len(subset),points=len(vals),radial_rmse_px=float(np.sqrt(np.mean(np.square(vals)))))
    write(E/'summary.json',dict(ten_pairs=summary,detector_only_other_splits=expanded,previous_four_max_error_delta_mm=max(diffs)))
    write(E/'execution.json',dict(base='ce5a8ad65b9c98f7952559ff895d8d1f4ddbf97d',worktree='new run and unrelated existing edits',fit_complete=conds,
        diagnostics=diag,commands=['<BODY_PYTHON> -Bu '+str((RUN/f).relative_to(ROOT)) for f in ['experiment.py','metrics.py']],
        exit_fit=int((OUT/'fit.exit').read_text()),evaluation_exit=0,
        weights=read(REF/'evidence/checkpoints.json')['sha256'],calibration_sha256={p.name:sha(p) for p in common.camera_files()}))
    print(summary,flush=True)

if __name__=='__main__':main()
