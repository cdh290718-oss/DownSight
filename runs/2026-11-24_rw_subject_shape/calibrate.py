"""Separate actions before estimating a person's shared neutral-shape coefficients."""
from collections import defaultdict
import zipfile
import warnings
import torch
from data_shape import *

sys.path.insert(0,str(ROOT/'runs/2026-11-23_rw20000_wrist_depth'))
from check_depth import geometry


def load_labels():
    data = ROOT.parent/'EgoSMPLX/data'
    archive = data/'UnrealEgo_RW_Train/official_splits/UnrealEgoData_rw.zip'
    with zipfile.ZipFile(archive) as z:
        split_lists = {part:z.read('UnrealEgoData_rw/'+part+'.txt').decode().splitlines()
                       for part in ['train','validation','test']}
    audit = read(ROOT/'runs/2026-11-23_rw_joint_ranges/evidence/survey.json')
    values, source_ids = {}, {'splits':sha(archive)}
    for group in ['Train','Test']:
        folder = data/('UnrealEgo_RW_'+group)/'prepared'
        expected = audit['sources'][group]['samples_sha256']
        assert sha(folder/'samples.npz')==expected
        source_ids[group]=expected
        names = read(folder/'metadata.json')['fit_joint_names']
        with np.load(folder/'samples.npz') as z:
            keep = z['camera_side']=='left'
            seq, frame = z['source_sequence'][keep], z['frame_index'][keep]
            spans = sizes(z['keypoints3d_fit'][keep].astype(float),dict(zip(names,range(len(names)))))
        assert len(set(zip(seq,frame))) == len(seq) and np.isfinite(spans).all() and spans.min()>0
        for name in np.unique(seq):
            assert name not in values
            values[name]=spans[seq==name]
    assert set(values)==set(sum(split_lists.values(),[]))
    return values,split_lists,source_ids


def partition(values):
    forced={im[0]['sequence'] for split in ('test','validation') for k,_,im,_ in inputs(split) if k in FOUR}
    subjects=sorted({s.split('/')[2] for s in values})
    parts={}
    for p in subjects:
        seqs={s for s in values if s.split('/')[2]==p}
        eligible=sorted(seqs-forced,key=rank_sequence)
        cal=sorted(eligible[:len(seqs)//2])
        hold=sorted(seqs-set(cal))
        assert len(cal)>=2 and not set(cal)&set(hold) and set(cal).isdisjoint(forced)
        parts[p]=dict(calibration=cal,holdout=hold)
    return parts


def stereo_targets(parts,cameras):
    seqvalues=defaultdict(list)
    inputs_used=defaultdict(list)
    for split in ['test','validation']:
        for pair,item,images,_ in inputs(split):
            s=images[0]['sequence']
            if s not in parts[images[0]['subject']]['calibration']:
                continue
            T=np.asarray(item['inputs']['views']['right']['camera']['left_to_view'])
            pts={}
            for p in item['points']:
                if all(p['valid']):
                    tri=geometry.triangulate(cameras,T[:3,:3],T[:3,3],p['xy'])
                    if tri['accepted']:
                        pts[p['name']]=np.asarray(tri['point_cam3_m'])
            seqvalues[s].append([np.linalg.norm(pts[a]-pts[b]) if a in pts and b in pts else np.nan for a,b in SPANS])
            inputs_used[s].append(pair)
    return seqvalues,inputs_used


def sparse_median(arr, min_count):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='All-NaN slice encountered')
        measure=np.nanmedian(arr,axis=0)
    measure[np.isfinite(arr).sum(0)<min_count]=np.nan
    return measure


def main():
    torch.set_num_threads(1)
    model,cam=ref.common.setup_model()
    basis=ShapeModel(model.layer['neutral'])
    values,lists,source=load_labels()
    parts=partition(values)
    stereo,used=stereo_targets(parts,[cam(p) for p in ref.camera_files()])
    training={s.split('/')[2] for s in lists['train']}
    targets={p:np.median([np.median(values[s],axis=0) for s in part['calibration']],axis=0) for p,part in parts.items()}
    params,records={},{}
    for p,part in parts.items():
        pooled=np.median([targets[q] for q in sorted(training-{p})],axis=0)
        fits={'mean':np.zeros(10),'cohort':basis.solve(pooled,.010),'personal':basis.solve(targets[p],.010)}
        seqs=[s for s in part['calibration'] if s in stereo]
        record={'calibration_frames':sum(len(values[s]) for s in part['calibration']),
                'holdout_frames':sum(len(values[s]) for s in part['holdout'])}
        if seqs:
            frames=np.concatenate([stereo[s] for s in seqs])
            per_seq=np.array([sparse_median(np.asarray(stereo[s]),1) for s in seqs])
            target=sparse_median(per_seq,2)
            target[np.isfinite(frames).sum(0)<3]=np.nan
            fits['stereo12']=basis.solve(target,.025)
            record['stereo']=dict(frames=len(frames),sequences=len(seqs),span_observations=np.isfinite(frames).sum(0).tolist(),
                valid_spans=np.isfinite(target).tolist(),frame_ids={s:used[s] for s in seqs})
        params[p]={arm:beta.tolist() for arm,beta in fits.items()}
        errors={}
        for arm,beta in fits.items():
            pred=basis.predict(beta)
            seq_abs=np.array([np.abs(pred-values[s]).mean(0)*1000 for s in part['holdout']])
            seq_ratio=np.array([np.abs(ratios(pred)-ratios(values[s])).mean(0) for s in part['holdout']])
            errors[arm]=dict(span_mae_mm=seq_abs.mean(0).tolist(),mae_mm=seq_abs.mean(),
                proportion_mae_pp=seq_ratio.mean(),max_abs_beta=max(abs(beta)))
        records[p]=dict(**record,scores=errors)
    pub={}
    for subset,persons in [('all',list(parts)),('stereo_people',[p for p in parts if 'stereo12' in params[p]])]:
        keys=list(params[persons[0]]) if subset=='stereo_people' else ['mean','cohort','personal']
        pub[subset]={k:{metric:np.mean([records[p]['scores'][k][metric] for p in persons]) for metric in
                         ['mae_mm','proportion_mae_pp']} for k in keys}
    write(OUT/'betas.json',params)
    write(EV/'calibration.json',dict(exit_status=0,input_sha256=source,split=parts,records=records,summary=pub,
        official_train_subjects=sorted(training),spans=SPANS,betas_sha256=sha(OUT/'betas.json'),
        solver=dict(official_sigma_m=.010,stereo_sigma_m=.025,beta_limit=3,regularizer='sum(beta^2)')))
    print(pub,flush=True)


if __name__=='__main__':
    main()
