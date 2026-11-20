"""Paired, label-blind checkpoint selection followed by official 12-joint scoring."""
import numpy as np
import torch

from compare import base, REF, RUN, OUT, read, write, sha
import measure
from metrics import align

FIELDS=['nonpa_mm','pa_mm','gt_rmse_px','detection_rmse_px']


def summarize(rows):
    n=sum(r['n'] for r in rows)
    return dict(nonpa_mm=float(np.mean([r['nonpa_mm'] for r in rows])),
                pa_mm=float(np.mean([r['pa_mm'] for r in rows])),
                gt_rmse_px=float(np.sqrt(sum(r['gt_squared_px'] for r in rows)/n)),
                detection_rmse_px=float(np.sqrt(sum(r['detection_squared_px'] for r in rows)/n)))


def group_scores(pred, target, item, images, joint_ids, cameras, adapters):
    names=[p['name'].split('_')[1].lower() for p in item['points']]
    indices={'all':list(range(len(names)))}
    for i,name in enumerate(names):
        indices.setdefault(name,[]).append(i)
    native=np.linalg.norm(pred-target,axis=1)*1000
    aligned=np.linalg.norm(align(pred,target)-target,axis=1)*1000
    result={}
    for name,ids in indices.items():
        subset=dict(item,points=[item['points'][i] for i in ids])
        views=measure.pixel_errors(subset,pred[ids],images,[joint_ids[i] for i in ids],cameras,adapters)
        n=sum(v['n'] for v in views)
        sse=[sum(v['sum_sq_px2'][k] for v in views) for k in ['fit_vs_gt','fit_vs_detection']]
        result[name]=dict(nonpa_mm=float(native[ids].mean()),pa_mm=float(aligned[ids].mean()),
                          n=n,gt_squared_px=sse[0],detection_squared_px=sse[1])
    return result


def main():
    torch.set_num_threads(1)
    done=read(OUT/'fit_complete.json')
    diag=read(OUT/'diagnostic_20000.json')
    assert done['exit_status']==0 and done['steps']==20000 and done['ids']==diag['ids']
    assert sha(OUT/'protocol.json')==done['protocol_sha256']==sha(base.OUT/'protocol.json')
    items=read(OUT/'prepared.json')['items']
    refs_path=base.SOURCE/'reference.json'
    assert sha(refs_path)==read(REF/'evidence/execution.json')['reference_sha256']
    refs=read(refs_path)
    images,jids=measure.reference_images()
    model,camera=base.common.setup_model()
    cameras=[camera(p) for p in base.camera_files()]
    adapters=[read(p)['input_adapter'] for p in base.camera_files()]
    params=dict(np.load(OUT/'budget_20000.npz'))
    assert not params['body_pose'][:,[11,14]].any() and not params['betas'].any()
    assert (params['selected_step']==diag['selected_step']).all()
    init=read(OUT/'initialization.json')
    ref_init=read(base.OUT/'initialization.json')
    original={r['pair']:r for r in read(REF/'evidence/results.json')['rows'] if r['model']=='rw20000'}
    rows={}
    archive_diff=[]
    for i,(pair,label,method) in enumerate(done['ids']):
        assert (items[i]['pair'],items[i]['condition'],method)==(pair,label,'2d')
        j=ref_init['ids'].index([pair,'rw20000','2d'])
        for key in ['global0','initial_parameters']:
            old=ref_init[key] if key=='initial_parameters' else [ref_init[key]]
            now=init[key] if key=='initial_parameters' else [init[key]]
            assert all(np.allclose(a[i],b[j],atol=1e-6,rtol=0) for a,b in zip(now,old))
        if label=='equal':
            assert dict(items[i],condition='baseline')==items[i-1]
        ref=refs[pair]
        assert all(img['frame']==ref['frame'] and img['sequence']==ref['sequence'] for img in images[pair])
        group='grouped' if label=='baseline' else label
        row=rows.setdefault(pair,dict(pair=pair,subject=ref['subject']))
        rec=dict(feasible=bool(diag['selected_feasible'][i] and params['selected_step'][i]>=0),
                 converged=diag['convergence_pass'][i],selected_step=int(params['selected_step'][i]))
        row[group]=rec
        if rec['feasible']:
            pred=measure.full_joints(params,model.layer['neutral'],i)[base.common.NATIVE_IDS][base.common.OBS]
            target=np.asarray(ref['gt'])[base.common.OBS]
            rec['groups']=group_scores(pred,target,items[i],images[pair],jids,cameras,adapters)
            if group=='grouped':
                current=summarize([rec['groups']['all']])
                former=original[pair]
                archive_diff.append(dict(pair=pair,nonpa_mm=current['nonpa_mm']-former['mpjpe12_mm'],
                                         pa_mm=current['pa_mm']-former['pa12_mm'],
                                         selected_step_difference=rec['selected_step']-former['selected_step']))
    paired=[r for r in rows.values() if all(r[m]['feasible'] for m in ['grouped','equal'])]
    def compare(samples,name):
        out={}
        for condition in ['grouped','equal']:
            out[condition]=summarize([r[condition]['groups'][name] for r in samples])
        out['delta']={k:out['equal'][k]-out['grouped'][k] for k in FIELDS}
        out['wins']={}
        out['median_paired_delta']={}
        for k in FIELDS:
            diffs=[summarize([r['equal']['groups'][name]])[k]-summarize([r['grouped']['groups'][name]])[k] for r in samples]
            out['wins'][k]=sum(d<-1e-6 for d in diffs)
            out['median_paired_delta'][k]=float(np.median(diffs))
        return out
    assert len(rows)==100 and len(paired)>0
    for row in paired:
        for arm in ['grouped','equal']:
            groups=row[arm]['groups']
            for field in ['nonpa_mm','pa_mm']:
                assert np.isclose(groups['all'][field],np.mean([g[field] for k,g in groups.items() if k!='all']))
            for field in ['n','gt_squared_px','detection_squared_px']:
                assert np.isclose(groups['all'][field],sum(g[field] for k,g in groups.items() if k!='all'))
    scores={name:compare(paired,name) for name in paired[0]['grouped']['groups']}
    subjects={sub:compare([r for r in paired if r['subject']==sub],'all') for sub in sorted({r['subject'] for r in paired})}
    ranks=sorted(paired,key=lambda r:r['equal']['groups']['all']['nonpa_mm']-r['grouped']['groups']['all']['nonpa_mm'])
    # One-frame influence is reported for every possible omission, without choosing exclusions.
    leave_one_out={}
    for k in FIELDS:
        changes=[]
        for omitted in paired:
            rest=[r for r in paired if r['pair']!=omitted['pair']]
            a,b=[summarize([r[m]['groups']['all'] for r in rest])[k] for m in ['grouped','equal']]
            changes.append((b-a,omitted['pair']))
        leave_one_out[k]=dict(min_delta=min(changes)[0],min_omitted=min(changes)[1],
                              max_delta=max(changes)[0],max_omitted=max(changes)[1])
    write(RUN/'evidence/rows.json',dict(rows=list(rows.values())))
    write(RUN/'evidence/summary.json',dict(scores=scores,subjects=subjects,pairs=len(paired),leave_one_out=leave_one_out,
          n=sum(r['grouped']['groups']['all']['n'] for r in paired),
          converged={m:sum(r[m]['converged'] for r in rows.values()) for m in ['grouped','equal']},
          feasible={m:sum(r[m]['feasible'] for r in rows.values()) for m in ['grouped','equal']},
          extremes_nonpa=[r['pair'] for r in [ranks[0],ranks[-1]]]))
    write(RUN/'evidence/verification.json',dict(base_commit='8f3fa992825235e1ba758097e63ad369e600d7e9',
          commands=['<BODY_PYTHON> -Bu compare.py','<BODY_PYTHON> -Bu score.py'],fit_exit=0,eval_exit=0,
          steps=done['steps'],elapsed_seconds=done['elapsed_seconds'],protocol_sha256=done['protocol_sha256'],
          shared_input_sha256=sha(base.OUT/'prepared.json'),paired_input_sha256=sha(OUT/'prepared.json'),
          params_sha256=sha(OUT/'budget_20000.npz'),reference_sha256=sha(refs_path),
          equal_inputs_except_arm=True,initial_parameters_checked=True,archived_grouped_delta=archive_diff))
    print(scores['all'],flush=True)


if __name__=='__main__':
    main()
