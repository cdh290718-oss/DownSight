"""Evaluate joint-level detector and fitted errors against the same archived labels."""
from pathlib import Path
import sys
import numpy as np
import torch

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
BASE = ROOT/'runs/2026-11-19_rw_frozen200_free12'
sys.path.insert(0,str(BASE))
import measure as ref
from metrics import align

def rmse(dist):
    return float(np.sqrt(np.mean(dist**2)))


def main():
    torch.set_num_threads(1)
    execution=ref.read(BASE/'evidence/execution.json')
    inputs=ref.read(BASE/'evidence/inputs.json')
    paths={ref.OUT/'budget_20000.npz':execution['params_sha256'],ref.OUT/'prepared.json':inputs['solver_input_sha256'],
           ref.SOURCE/'reference.json':execution['reference_sha256']}
    paths.update({p:inputs['cameras'][p.name] for p in ref.camera_files()})
    assert all(ref.sha(p)==digest for p,digest in paths.items())
    items=ref.read(ref.OUT/'prepared.json')['items']
    ids=ref.read(ref.OUT/'fit_complete.json')['ids']
    params=dict(np.load(ref.OUT/'budget_20000.npz'))
    truths=ref.read(ref.SOURCE/'reference.json')
    images,jids=ref.reference_images()
    model,cam=ref.common.setup_model()
    cams=[cam(p) for p in ref.camera_files()]
    adapters=[ref.read(p)['input_adapter'] for p in ref.camera_files()]
    joints=[ref.common.NAMES[i] for i in ref.common.OBS]
    refs={r['pair']:r for r in ref.read(BASE/'evidence/results.json')['rows'] if r['model']=='rw20000'}
    detected=[]; fitted=[]; body=[]; pnames=[]; subjects=[]
    for i,(pair,model_name,method) in enumerate(ids):
        if model_name!='rw20000': continue
        item=items[i]
        assert (item['pair'],item['condition'],method)==(pair,'rw20000','2d')
        assert [p['name'] for p in item['points']]==joints and refs[pair]['feasible']
        pred=ref.full_joints(params,model.layer['neutral'],i)[ref.common.NATIVE_IDS][ref.common.OBS]
        gt=np.asarray(truths[pair]['gt'])[ref.common.OBS]
        error=np.linalg.norm(pred-gt,axis=1)*1000
        assert abs(error.mean()-refs[pair]['mpjpe12_mm'])<.001
        assert abs(np.linalg.norm(align(pred,gt)-gt,axis=1).mean()*1000-refs[pair]['pa12_mm'])<.001
        d=[]; f=[]
        for k,side in enumerate(['left','right']):
            im=images[pair][k]
            assert (im['view'],im['sequence'],im['frame'])==(side,truths[pair]['sequence'],truths[pair]['frame'])
            valid=np.array([p['valid'][k] for p in item['points']],bool)
            assert np.array_equal(valid,np.asarray(im['keypoints308_supervision_mask'])[jids])
            xy=np.asarray(im['keypoints308_xy'])[jids]
            a=adapters[k]
            T=np.asarray(item['inputs']['views'][side]['camera']['left_to_view'])
            obs=np.asarray([p['xy'][k] for p in item['points']])
            proj=cams[k].world2camera(pred@T[:3,:3].T+T[:3,3])
            for coords,target,field in [(obs,d,'detection_vs_gt'),(proj,f,'fit_vs_gt')]:
                residual=(coords-a['offset'])/a['scale']-xy
                assert np.isclose(np.sum(residual[valid]**2),refs[pair]['views'][k]['sum_sq_px2'][field],rtol=1e-5,atol=.002)
                target.append(np.where(valid[:,None],residual,np.nan))
        subjects.append(truths[pair]['subject']);pnames.append(pair);detected.append(d);fitted.append(f);body.append(error)
    D,F=np.array(detected),np.array(fitted)
    B=np.array(body)
    assert len(set(pnames))==len(pnames)==100
    def aggregate(d,f,b):
        valid=np.isfinite(d).all(-1)
        assert np.array_equal(valid,np.isfinite(f).all(-1))
        dist=np.linalg.norm(d[valid],axis=-1)
        fd=np.linalg.norm(f[valid],axis=-1)
        return dict(n2=int(valid.sum()),n3=b.size,input_rmse_px=rmse(dist),fit_rmse_px=rmse(fd),fit_rmse_mm=rmse(b),
                    input_mean_distance_px=float(dist.mean()),input_median_px=float(np.median(dist)),
                    input_p90_px=float(np.percentile(dist,90)),input_mean_dxdy_px=d[valid].mean(0).tolist(),
                    fit_mpjpe_mm=float(b.mean()))
    result={}
    for j,name in enumerate(joints):
        d,f,b=D[:,:,j],F[:,:,j],B[:,j]
        stats=aggregate(d,f,b)
        radii=np.linalg.norm(d,axis=-1)
        valid_rows=np.argwhere(np.isfinite(radii))
        valid_distances=radii[tuple(valid_rows.T)]
        largest=np.argsort(valid_distances)[-2:][::-1]
        stats['largest_two_observations']=[
            dict(pair=pnames[i],eye=['left','right'][k],distance_px=float(radii[i,k]))
            for i,k in valid_rows[largest]]
        stats['largest_two_fraction_of_input_sse']=float(np.sum(valid_distances[largest]**2)/np.sum(valid_distances**2))
        stats['eyes']={eye:aggregate(d[:,k],f[:,k],b)['input_rmse_px'] for k,eye in enumerate(['left','right'])}
        stats['by_subject']={}
        for subject in sorted(set(subjects)):
            mask=np.array(subjects)==subject
            stats['by_subject'][subject]=aggregate(d[mask],f[mask],b[mask])
        result[name]=stats
    overall=aggregate(D,F,B)
    previous=ref.read(BASE/'evidence/summary.json')['overall']['rw20000']
    assert abs(overall['input_rmse_px']-previous['detection_vs_gt_rmse_px'])<.001
    assert abs(overall['fit_rmse_px']-previous['fit_vs_gt_rmse_px'])<.001
    input_summary=ref.read(ROOT/'runs/2026-11-18_rw20000_frozen200/evidence/2d_summary.json')['summary']['rw20000']['test_frozen']
    for j,name in enumerate(joints):
        assert abs(result[name]['input_mean_distance_px']-input_summary['per_joint_px'][str(jids[j])])<.001
    ref.write(RUN/'evidence/results.json',dict(joints=result,overall=overall,
              joint_ids_308=dict(zip(joints,jids)),subject_counts={s:subjects.count(s) for s in sorted(set(subjects))},
              masked=[dict(pair=p,eye=['left','right'][k],joint=joints[j]) for i,p in enumerate(pnames)
                      for k,j in np.argwhere(~np.isfinite(D[i]).all(-1))],
              input_sha256={str(p.relative_to(ROOT)):h for p,h in paths.items()},manifest_sha256=execution['manifest_sha256'],
              base='b03962086b54015141422b63e027f14e3e76b6a4',command='<BODY_PYTHON> -Bu measure_joints.py',exit_status=0))
    print('joint, input 2D RMSE, fit 2D RMSE, fit 3D RMSE, median, p90, mean dxdy')
    for n,a in sorted(result.items(),key=lambda x:x[1]['input_rmse_px']):
        print(n,*[round(a[k],2) for k in ['input_rmse_px','fit_rmse_px','fit_rmse_mm','input_median_px','input_p90_px']],np.round(a['input_mean_dxdy_px'],2))
    print(overall)


if __name__=='__main__':main()
