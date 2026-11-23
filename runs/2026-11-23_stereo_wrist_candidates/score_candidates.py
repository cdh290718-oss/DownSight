"""Use official labels after selection. Oracle minimizes per-wrist 3D error in the same bank."""
import sys
from collections import Counter
import torch
from shared import *
from select_wrist import ARMS

def summarize(rows):
    n=len(rows)
    joints=np.array([r['mm'] for r in rows]); pa=np.array([r['pa_mm'] for r in rows])
    wrists=[k for k,j in enumerate(ref.common.OBS) if 'Wrist' in ref.common.NAMES[j]]
    others=[i for i in range(12) if i not in wrists]
    return dict(pairs=n,mpjpe=float(joints.mean()),rmse=float(np.sqrt((joints**2).mean())),pa=float(pa.mean()),
        wrist_rmse=float(np.sqrt((joints[:,wrists]**2).mean())),other10=float(joints[:,others].mean()),
        px=float(np.sqrt(sum(v['px_sq'] for v in rows)/sum(v['n_px'] for v in rows))))

def measure(split):
    torch.set_num_threads(1)
    folder,manifest,truthfile=(VAL,VM,OUT/'validation_truth.json') if split=='validation' else (base.OUT,TM,ref.SOURCE/'reference.json')
    rows=read(OUT/(split+'_candidates.json')); choicefile=OUT/(split+'_predictions.json')
    assert sha(choicefile)==read(RUN/('evidence/'+split+'_prediction.json'))['choices_sha256']
    choices=read(choicefile)
    assert all(set(c)==set(rows) for c in choices.values())
    truths=read(truthfile);pairs=pair_images(manifest)
    pack=dict(np.load(folder/'budget_20000.npz')); items=read(folder/'prepared.json')['items']
    if split=='test':
        assert sha(truthfile)==read(base.RUN/'evidence/execution.json')['reference_sha256']
    model,cam=ref.common.setup_model();layer=model.layer['neutral']
    cams=[cam(p) for p in ref.camera_files()]
    adapters=[read(p)['input_adapter'] for p in ref.camera_files()]
    ids2=mapping()
    keys=['baseline','objective','image','gated','oracle']
    records={k:{} for k in keys}; wrists=[]
    posepacks={k:[] for k in ['objective','image','gated']}
    for pair,row in rows.items():
        i=row['index']; item=items[i]; assert pair==item['pair']
        pos=int(pair[-3:])-1
        im=pairs[pos];gt=np.asarray(truths[pair]['gt'])
        assert row['subject']==truths[pair]['subject']==im[0]['subject']
        oracle=[]; per_w=[]
        for side,(arm,cands) in enumerate(zip(ARMS,row['arms'])):
            w=arm[-1];g=gt[ref.common.NATIVE_IDS.index(w)]
            errs=np.array([np.linalg.norm(np.array(c['wrist'])-g)*1000 for c in cands])
            oracle.append(int(errs.argmin()))
            wr=dict(pair=pair,subject=row['subject'],side='left' if side==0 else 'right',n_candidates=len(cands))
            for k in keys:
                idx=oracle[-1] if k=='oracle' else choices[k][pair][side]
                wr[k]=dict(candidate=idx,error_mm=float(errs[idx]))
            per_w.append(wr)
        wrists.extend(per_w)
        cache={}
        for k in keys:
            indices=tuple(oracle if k=='oracle' else choices[k][pair])
            if indices not in cache:
                params={name:pack[name][i:i+1].copy() for name in ['global0','delta','translation','betas','body_pose']}
                for (s,e,w),c,idx in zip(ARMS,row['arms'],indices):
                    params['body_pose'][0,[s-1,e-1]]=c[idx]['pose']
                native=ref.full_joints(params,layer,0)
                for (s,e,w),c,idx in zip(ARMS,row['arms'],indices):
                    assert np.linalg.norm(native[w]-c[idx]['wrist'])<5e-5 and np.linalg.norm(native[e]-c[idx]['elbow'])<5e-5
                pred=native[ref.common.NATIVE_IDS][ref.common.OBS]; target=gt[ref.common.OBS]
                views=ref.pixel_errors(item,pred,im,ids2,cams,adapters)
                result=dict(mm=(1000*np.linalg.norm(pred-target,axis=1)).tolist(),
                  pa_mm=(1000*np.linalg.norm(align(pred,target)-target,axis=1)).tolist(),
                  n_px=sum(v['n'] for v in views),px_sq=sum(v['sum_sq_px2']['fit_vs_gt'] for v in views))
                cache[indices]=(result,params)
            records[k][pair]=cache[indices][0]
            if k in posepacks: posepacks[k].append(cache[indices][1])
    for k,ps in posepacks.items():
        np.savez_compressed(OUT/(split+'_'+k+'.npz'),pair=list(rows),**{f:np.concatenate([p[f] for p in ps]) for f in ps[0]})
    base_wr={p:sum(w['baseline']['error_mm']**2 for w in wrists if w['pair']==p) for p in rows}
    benefit={p:sum(w['baseline']['error_mm']**2-w['image']['error_mm']**2 for w in wrists if w['pair']==p) for p in rows}
    worst=sorted(rows,key=base_wr.get,reverse=True)[:2]
    top=sorted(rows,key=benefit.get,reverse=True)[:2]
    scopes={'all':list(rows),'without_worst_baseline':[p for p in rows if p not in worst],
            'without_top_image_gains':[p for p in rows if p not in top]}
    if split=='test': scopes['without_prior_outliers']=[p for p in rows if p not in ('pair003','pair004')]
    for sub in sorted({r['subject'] for r in rows.values()}):
        scopes[sub]=[p for p,r in rows.items() if r['subject']==sub]
    summary={name:{k:summarize([records[k][p] for p in subset]) for k in keys} for name,subset in scopes.items()}
    def counts(k):
        changed=[w for w in wrists if w[k]['candidate']!=0]
        delta=[w[k]['error_mm']-w['baseline']['error_mm'] for w in changed]
        return dict(changed=len(changed),improved=sum(v<-.001 for v in delta),worsened=sum(v>.001 for v in delta),
              tied=sum(abs(v)<=.001 for v in delta),affected_pairs=len({w['pair'] for w in changed}))
    if split=='test':
        frozen=[61.23,79.62,44.58,108.91,60.36,18.69]
        assert np.max(abs(np.array([summary['all']['baseline'][k] for k in ['mpjpe','rmse','pa','wrist_rmse','other10','px']])-frozen))<=.00501
    write(RUN/('evidence/'+split+'_results.json'),dict(summary=summary,counts={k:counts(k) for k in keys},
          baseline_worst_pairs=worst,biggest_image_gain_pairs=top,
          candidate_count_histogram=dict(Counter(w['n_candidates'] for w in wrists)),wrists=wrists,
          rows={p:{k:dict(summarize([records[k][p]]),n_px=records[k][p]['n_px']) for k in keys} for p in rows},
          reference_sha256=sha(truthfile),prediction_sha256=sha(choicefile),exit_status=0))
    print(split,summary['all'],{k:counts(k) for k in keys},flush=True)

if __name__=='__main__': measure(sys.argv[1])
