"""Swap only shape at the selected v3 pose. Quantify surface crossings after selection."""
from collections import Counter
import torch
from data_shape import *

sys.path.insert(0,str(ROOT/'runs/2026-11-23_floor24_intersections'))
from inspect_obj import self_intersections, groups, angle
sys.path.insert(0,str(ROOT/'runs/2026-11-19_rw_landmarks_head_views'))
from export import geometry


def metrics(pred, target, item, images, ids, cams, adapters):
    views=ref.pixel_errors(item,pred,images,ids,cams,adapters)
    return dict(mpjpe_mm=1000*np.linalg.norm(pred-target,axis=1).mean(),
        pa_mm=1000*np.linalg.norm(shared.align(pred,target)-target,axis=1).mean(),
        px_sq=sum(v['sum_sq_px2']['fit_vs_gt'] for v in views),n_px=sum(v['n'] for v in views))


def surface(pair, v, faces, labels, diag, arm):
    hits=self_intersections(v,faces)
    counts=Counter(' / '.join(sorted([labels[a],labels[b]])) for a,b in hits)
    if arm=='mean':
        obj=ROOT/'outputs/fixed_floor24_20261123/gallery'/pair/(pair+'.obj')
        assert sha(obj)==diag[pair]['obj_sha256']
        with obj.open() as f:
            archived=np.array([[float(x) for x in line.split()[1:]] for line in f if line.startswith('v ')])
        assert abs(archived-v).max()<1e-7
        assert all(counts[k]==diag[pair]['surface_crossing_face_pairs'].get(k,0) for k in FOUR[pair])
    return dict(original_regions={s:counts[s] for s in FOUR[pair]},all_intersections=dict(counts))


def summarize(rows):
    return dict(n=len(rows), mpjpe_mm=np.mean([v['mpjpe_mm'] for v in rows]),
                pa_mm=np.mean([v['pa_mm'] for v in rows]),
                px_rmse=np.sqrt(sum(v['px_sq'] for v in rows)/sum(v['n_px'] for v in rows)))


def main():
    torch.set_num_threads(1)
    cal=read(EV/'calibration.json')
    assert sha(OUT/'betas.json')==cal['betas_sha256']
    betas=read(OUT/'betas.json')
    model,camera=ref.common.setup_model()
    layer=model.layer['neutral']
    faces=np.asarray(model.face)
    labels=groups(layer,faces)
    cams=[camera(p) for p in ref.camera_files()]
    adapt=[read(p)['input_adapter'] for p in ref.camera_files()]
    ids=shared.mapping()
    diag=read(ROOT/'runs/2026-11-23_floor24_intersections/evidence/results.json')['rows']
    obs=ref.common.OBS
    native=np.asarray(ref.common.NATIVE_IDS)[obs]
    lookup={ref.common.NAMES[j]:i for i,j in enumerate(obs)}
    rows,details,hashes={},{},{}
    for split,collection in [('validation','Train'),('test','Test')]:
        archive=ROOT/'runs/2026-11-23_fullbody_joint_candidates/evidence'
        provenance=read(archive/(split+'_search.json'))
        for suffix,key in [('bank.npz','parameter_sha256'),('choices.json','choices_sha256')]:
            p=BANK/(split+'_'+suffix)
            assert sha(p)==provenance[key]
            hashes[p.name]=provenance[key]
        old=read(archive/(split+'_results.json'))['rows']
        params=dict(np.load(BANK/(split+'_bank.npz')))
        data=ROOT.parent/'EgoSMPLX/data'/('UnrealEgo_RW_'+collection)/'prepared'
        with np.load(data/'samples.npz') as z:
            sequences,frames,eyes,xyz=[z[k] for k in ['source_sequence','frame_index','camera_side','keypoints3d']]
        for pair,item,im,r in inputs(split):
            subject=im[0]['subject']
            seq=im[0]['sequence']
            if seq not in cal['split'][subject]['holdout']:
                continue
            idx=r['offset']+r['best']
            assert old[pair]['choice']==r['names'][r['best']] and np.all(params['betas'][idx]==0)
            source=im[0]['source_index']
            assert (str(sequences[source]),int(frames[source]),eyes[source])==(seq,im[0]['frame'],'left')
            gt=xyz[source,obs].astype(float)
            row=dict(subject=subject,split=split,sequence=seq,choice=old[pair]['choice'],scores={})
            for arm in ['mean','cohort','personal','stereo12']:
                p={k:params[n][idx] for k,n in [('global_orient','global0'),('body_pose','body_pose'),('transl','translation')]}
                p['betas']=np.asarray(betas[subject][arm])
                v,j,_,_=geometry(layer,p,params['delta'][idx],False,None)
                if arm=='mean':
                    head0=j[15].copy()
                score=metrics(j[native],gt,item,im,ids,cams,adapt)
                if arm=='mean':
                    assert abs(score['mpjpe_mm']-old[pair]['scores']['joint']['mpjpe'])<.001
                row['scores'][arm]=score
                row['scores'][arm+'_head_aligned']=metrics(j[native]+head0-j[15],gt,item,im,ids,cams,adapt)
                if pair in FOUR:
                    arm_diag=surface(pair,v,faces,labels,diag,arm)
                    arm_diag['forearm_direction_error_deg']={}
                    for side in ['L','R']:
                        elbow,wrist=[lookup[side+'_'+n] for n in ['Elbow','Wrist']]
                        arm_diag['forearm_direction_error_deg'][side]=angle(j[native[wrist]]-j[native[elbow]],gt[wrist]-gt[elbow])
                    details.setdefault(pair,{})[arm]=arm_diag
            rows[pair]=row
    assert set(details)==set(FOUR)
    sets={'all':list(rows)}
    for field in ['subject','split']:
        for value in sorted({r[field] for r in rows.values()}):
            sets[value]=[k for k,r in rows.items() if r[field]==value]
    results={tag:{a:summarize([rows[k]['scores'][a] for k in keys]) for a in rows[keys[0]]['scores']} for tag,keys in sets.items()}
    write(EV/'poses.json',dict(exit_status=0,reference_sha256=hashes,summary=results,rows=rows))
    write(EV/'surfaces.json',dict(exit_status=0,rows=details,baseline_obj_sha256={k:diag[k]['obj_sha256'] for k in FOUR}))
    print(results['all'],flush=True)


if __name__=='__main__':
    main()
