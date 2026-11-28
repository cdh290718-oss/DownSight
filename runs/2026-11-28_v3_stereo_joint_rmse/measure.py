"""Compare unfiltered closest-ray triangulation with v3 on matched points."""
import sys
from pathlib import Path
from importlib.util import spec_from_file_location,module_from_spec
from collections import Counter
import numpy as np
import torch
RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[1]
sys.path.insert(0,str(RUN.parent/'2026-11-23_fullbody_joint_candidates'))
import score
prev=score.prev
spec=spec_from_file_location('stereo_geometry',ROOT/'runs/2026-11-04_session-hand6-v2/code/prepare.py')
geo=module_from_spec(spec);spec.loader.exec_module(geo)

def main():
    torch.set_num_threads(1)
    search=prev.read(score.RUN/'evidence/test_search.json')
    bank=score.OUT/'test_bank.npz'
    choices=score.OUT/'test_choices.json'
    assert prev.sha(bank)==search['parameter_sha256'] and prev.sha(choices)==search['choices_sha256']
    truthpath=prev.ref.SOURCE/'reference.json'
    assert prev.sha(truthpath)==prev.read(prev.RUN/'evidence/test_results.json')['reference_sha256']
    prepared=prev.base.OUT/'prepared.json'
    old=prev.read(prev.RUN/'evidence/splits.json')
    assert prev.sha(prev.TM)==old['hashes']['test_manifest']
    records=prev.read(choices);truth=prev.read(truthpath);items=prev.read(prepared)['items']
    poses=dict(np.load(bank))
    model,cam=prev.ref.common.setup_model()
    cameras=[cam(p) for p in prev.ref.camera_files()]
    obs=prev.ref.common.OBS
    names=[prev.ref.common.NAMES[k] for k in obs]
    native=np.asarray(prev.ref.common.NATIVE_IDS)[obs]
    out=[]
    for key,rec in records.items():
        row=items[rec['index']]
        assert row['pair']==key
        T=np.array(row['inputs']['views']['right']['camera']['left_to_view'])
        assert np.allclose(row['inputs']['views']['left']['camera']['left_to_view'],np.eye(4))
        R,t=T[:3,:3],T[:3,3]
        fit=prev.ref.full_joints(poses,model.layer['neutral'],rec['offset']+rec['best'])[native]
        target=np.array(truth[key]['gt'])[obs]
        for idx,name in enumerate(names):
            pt=next(p for p in row['points'] if p['name']==name)
            assert pt['native_index']==native[idx]
            result=dict(pair=key,joint=name,valid_views=pt['valid'],v3_mm=float(1000*np.linalg.norm(fit[idx]-target[idx])))
            if not all(pt['valid']):
                result['status']='missing_view';out.append(result);continue
            uv=np.asarray(pt['xy'],float)
            rays=[c.camera2world_ray(u[None])[0] for c,u in zip(cameras,uv)]
            b=R.T@rays[1];origin=-R.T@t
            matrix=np.stack([rays[0],-b],axis=1)
            distances,_,rank,_=np.linalg.lstsq(matrix,origin,rcond=None)
            if rank<2:
                result['status']='parallel';out.append(result);continue
            p=(rays[0]*distances[0]+origin+b*distances[1])/2
            if not np.isfinite(p).all():
                result['status']='nonfinite';out.append(result);continue
            g=geo.triangulate(cameras,R,t,uv)
            result.update(status='computed',tri_mm=float(1000*np.linalg.norm(p-target[idx])),
                          ray_depths_m=distances.tolist(),gap_mm=float(1000*np.linalg.norm(rays[0]*distances[0]-(origin+b*distances[1]))),
                          angle_deg=float(np.rad2deg(np.arccos(np.clip(rays[0]@b,-1,1)))),
                          geometry_status=g['reason'])
            if g['accepted']:
                result['refined_mm']=float(1000*np.linalg.norm(np.array(g['point_cam3_m'])-target[idx]))
            out.append(result)
    def stats(rows):
        a=np.array([r['tri_mm'] for r in rows]);b=np.array([r['v3_mm'] for r in rows])
        return dict(n=len(rows),tri_rmse_mm=float(np.sqrt(np.mean(a*a))),v3_rmse_mm=float(np.sqrt(np.mean(b*b))),
                    tri_median_mm=float(np.median(a)),tri_p90_mm=float(np.percentile(a,90)),
                    tri_max_mm=float(a.max()),negative_ray_depths=sum(min(r['ray_depths_m'])<=0 for r in rows))
    results={}
    for name in names:
        allrows=[r for r in out if r['joint']==name]
        ok=[r for r in allrows if r['status']=='computed']
        accepted=[r for r in ok if r['geometry_status']=='accepted']
        results[name]=dict(all_bilateral=stats(ok),missing=len(allrows)-len(ok),
                           geometry_accepted_n=len(accepted),
                           geometry_accepted_refined_rmse_mm=float(np.sqrt(np.mean([r['refined_mm']**2 for r in accepted]))) if accepted else None,
                           geometry_accepted_v3_rmse_mm=float(np.sqrt(np.mean([r['v3_mm']**2 for r in accepted]))) if accepted else None,
                           reasons=dict(Counter(r.get('geometry_status',r['status']) for r in allrows)))
    valid=[r for r in out if r['status']=='computed']
    all_v3=np.array([r['v3_mm'] for r in out])
    assert len(out)==1200 and abs(np.sqrt(np.mean(all_v3**2))-71.41345067498837)<1e-6
    prev.write(RUN/'evidence/results.json',dict(rows=out,joints=results,overall=stats(valid),v3_all_1200_rmse_mm=float(np.sqrt(np.mean(all_v3**2))),
        input_sha256=dict(bank=prev.sha(bank),choices=prev.sha(choices),truth=prev.sha(truthpath),prepared=prev.sha(prepared),cameras=[prev.sha(p) for p in prev.ref.camera_files()]),
        command='bash runs/2026-11-28_v3_stereo_joint_rmse/run.sh',exit_status=0))
    print('overall',stats(valid),flush=True)
    for name,v in results.items():print(name,v,flush=True)
if __name__=='__main__':main()
