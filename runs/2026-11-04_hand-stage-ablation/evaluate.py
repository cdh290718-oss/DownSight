"""Evaluate fixed automatic targets at every saved fitting stage."""
import argparse,json,os,hashlib,importlib.util
from pathlib import Path
import numpy as np
import cv2
from egosmplx_pipeline.io import read,load
from egosmplx_pipeline.bootstrap import setup

ROOT=Path(__file__).resolve().parents[2]
RUN=Path(__file__).resolve().parent
OUT=ROOT/'outputs/hand_stage_ablation_20261104'

def rmse(errors):
    return float(np.sqrt(np.mean(np.square(errors))))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--partial',action='store_true');a=ap.parse_args()
    inputs={g:read(OUT/g/'inputs.json')['records'] for g in ['A','B','C']}
    cfg=read(inputs['A'][0]['config'])
    os.environ.update(EGOSMPLX_CONFIG=cfg['paths']['egosmplx_config'],EGOSMPLX_REPO_ROOT=cfg['paths']['egosmplx_repo'],EGOSMPLX_REFERENCE_ROOT=str(OUT/'unused'))
    model,Camera=setup(cfg['paths']['egosmplx_repo'])
    from egosmplx_pipeline import fusion
    from egosmplx_pipeline.body_common import JOINT_MAPPING
    fusion.initialize(model)
    masks=model.layer['neutral'].lbs_weights.detach().numpy().argmax(1)
    head=masks==model.orig_joints_name.index('Head')
    if not a.partial:
        spec=importlib.util.spec_from_file_location('rasterizer',ROOT/'runs/2026-11-04_mesh-projection-review/code/rasterizer.py')
        raster=importlib.util.module_from_spec(spec);spec.loader.exec_module(raster)
        (RUN/'site/assets').mkdir(parents=True,exist_ok=True)
    records=[];source_hashes={};stage_order=['raw','rigid','body_pose','wrist','contour','guided','fused']
    for ri,original in enumerate(inputs['A']):
        row=dict(pair=original['pair'],camera=original['camera'],frame=original['id'],methods={})
        camera=Camera(original['row']['calibration'])
        lookup={q['name']:q for q in read(original['row']['predictions']['sapiens'])['keypoints']}
        ctx=read(original['context']);w=load(original['original']['wilor']);matches=read(original['original']['recipe'])['matches']
        names=[n for n in JOINT_MAPPING if lookup[n]['score']>=.3 and np.isfinite([lookup[n]['x'],lookup[n]['y']]).all() and 0<=lookup[n]['x']<1280 and 0<=lookup[n]['y']<720]
        ids=[model.joints_name.index(JOINT_MAPPING[n]) for n in names]
        target=np.array([[lookup[n]['x'],lookup[n]['y']] for n in names]);shoulder=[n.endswith('_shoulder') for n in names]
        base_params=load(Path(read(original['config'])['output'])/'frames'/original['id']/'raw/params.npz')
        for group in ['A','B','C']:
            rec=inputs[group][ri];config=read(rec['config']);root=Path(config['output']);frame=root/'frames'/rec['id'];ref=root/'reference';rel=Path(rec['id'])/rec['camera']
            paths=dict(raw=frame/'raw/params.npz',rigid=frame/'rigid/optimized_params.npz',body_pose=frame/'body/params.npz',wrist=ref/'wrist/final_frames'/rel/'body_params.npz',contour=ref/'contour/final_frames'/rel/'body_params.npz',guided=frame/'guided_body/params.npz',fused=frame/'fused/body_params.npz')
            if not a.partial:assert read(root/'run_state.json')['state']=='complete'
            method=dict(stages={})
            if (frame/'validation.json').exists():
                val=read(frame/'validation.json');method['validation']={k:val.get(k) for k in ['passed','body_reload_max_abs_m','fusion_reload_max_abs_m','local_surface_accepted']}
            for stage in stage_order:
                path=paths[stage]
                if not path.exists():
                    if a.partial:continue
                    raise FileNotFoundError(path)
                p=load(path);v,j=fusion.geometry(p)
                assert np.array_equal(p['betas'],base_params['betas'])
                assert np.array_equal(p['left_hand_pose'],base_params['left_hand_pose']) and np.array_equal(p['right_hand_pose'],base_params['right_hand_pose'])
                source_hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
                uv=camera.world2camera(j.astype(float));err=np.linalg.norm(uv[ids]-target,axis=1)
                if stage=='fused':
                    mesh=load(frame/'fused/mesh.npz');v=mesh['vertices_cam'].astype(float)
                    actual_matches=read(ref/'baseline/frames'/rel/'recipe.json')['matches']
                    assert {s:q['candidate'] for s,q in actual_matches.items()}=={s:q['candidate'] for s,q in matches.items()}
                st=dict(shoulder_rmse_px=rmse(err[shoulder]),body_rmse_px=rmse(err),body_errors_px=dict(zip(names,err.tolist())),pelvis_z_m=float(j[0,2]),projected_shoulder_width_px=float(np.linalg.norm(uv[8]-uv[9])),transl_m=p['transl'].reshape(3).tolist(),global_orient=p['global_orient'].reshape(3).tolist(),hands={})
                for si,side in enumerate(['left','right']):
                    hj=model.orig_hand_regressor[side]@v;handuv=camera.world2camera(hj.astype(float))
                    goal=w['joints_2d'][matches[side]['candidate']];valid=np.isfinite(goal).all(1)&(goal[:,0]>=0)&(goal[:,0]<1280)&(goal[:,1]>=0)&(goal[:,1]<720)
                    e2=np.linalg.norm(handuv-goal,axis=1)
                    kin=hj.copy();kin[0]=j[12+si]
                    tri=np.array(ctx['tri'])[si];ok=np.array(ctx['tri_weights'])[si]>0
                    e3=np.linalg.norm(kin-tri,axis=1)
                    st['hands'][side]=dict(rmse_2d_px=rmse(e2[valid]),rmse_3d_m=rmse(e3[ok]),bone_wrist_2d_px=float(np.linalg.norm(uv[12+si]-goal[0])),bone_wrist_3d_m=float(e3[0]),errors_2d_px=e2[valid].tolist(),errors_3d_m=e3[ok].tolist())
                h_uv=camera.world2camera(v)
                st['head_visible_vertices']=int((head&(v[:,2]>0)&np.isfinite(h_uv).all(1)&(h_uv[:,0]>=0)&(h_uv[:,0]<1280)&(h_uv[:,1]>=0)&(h_uv[:,1]<720)).sum())
                method['stages'][stage]=st
                if stage=='fused' and not a.partial:
                    im=cv2.imread(rec['row']['image']);Y,X=np.indices((720,1280));rays=camera.camera2world_ray(np.stack([X.ravel(),Y.ravel()],1)).reshape(720,1280,3).astype(np.float32)
                    colors=np.zeros_like(im);depth=np.full((720,1280),np.inf,np.float32);hp=np.zeros((720,1280),np.float32)
                    raster.raster(v.astype(np.float32),mesh['faces'].astype(np.int32),head.astype(np.float32),h_uv.astype(np.float32),rays,depth,colors,hp,180.)
                    visible=np.isfinite(depth);im[visible]=(.45*im[visible]+.55*colors[visible]).astype(np.uint8)
                    name=rec['pair']+'-'+rec['camera']+'-'+group+'.jpg'
                    assert cv2.imwrite(str(RUN/'site/assets'/name),im,[cv2.IMWRITE_JPEG_QUALITY,93])
                    method['image']='assets/'+name
            row['methods'][group]=method
        records.append(row)
    summary={}
    for group in ['A','B','C']:
        summary[group]={}
        for stage in stage_order:
            values=[r['methods'][group]['stages'][stage] for r in records if stage in r['methods'][group]['stages']]
            if not values:continue
            summary[group][stage]={k:float(np.mean([v[k] for v in values])) for k in ['shoulder_rmse_px','body_rmse_px','pelvis_z_m','projected_shoulder_width_px']}
            summary[group][stage].update(views=len(values),hand_rmse_2d_px=float(np.mean([h['rmse_2d_px'] for v in values for h in v['hands'].values()])),hand_rmse_3d_m=float(np.mean([h['rmse_3d_m'] for v in values for h in v['hands'].values()])),head_visible_vertices=sum(v['head_visible_vertices'] for v in values))
    comparisons={}
    if not a.partial:
        previous={r['frame']:r['methods']['no_margin'] for r in read(ROOT/'runs/2026-11-04_v2-no-margin-rerun/evidence/metrics.json')['frames']}
        for r in records:
            old=previous[r['frame']];now=r['methods']['A']['stages']['fused']
            assert np.isclose(old['body_shoulders']['rmse'],now['shoulder_rmse_px'],rtol=0,atol=1e-6)
            for side in ['left','right']:
                assert np.isclose(old['hands'][side]['all']['rmse'],now['hands'][side]['rmse_2d_px'],rtol=0,atol=1e-6)
                assert np.isclose(old['hands'][side]['triangulated_3d_m']['rmse'],now['hands'][side]['rmse_3d_m'],rtol=0,atol=1e-8)
        for group in ['B','C']:
            changes=[]
            for r in records:
                for side in ['left','right']:
                    old=r['methods']['A']['stages']['fused']['hands'][side];new=r['methods'][group]['stages']['fused']['hands'][side]
                    changes.append(dict(pair=r['pair'],camera=r['camera'],side=side,delta_2d_px=new['rmse_2d_px']-old['rmse_2d_px'],delta_3d_m=new['rmse_3d_m']-old['rmse_3d_m']))
            mean2=float(np.mean([q['delta_2d_px'] for q in changes]));mean3=float(np.mean([q['delta_3d_m'] for q in changes]))
            max2=max(q['delta_2d_px'] for q in changes);max3=max(q['delta_3d_m'] for q in changes)
            comparisons[group]=dict(changes=changes,mean_delta_2d_px=mean2,max_delta_2d_px=max2,mean_delta_3d_m=mean3,max_delta_3d_m=max3,hand_noninferiority_passed=mean2<=1 and max2<=3 and mean3<=.005 and max3<=.015)
        # B and C have identical rigid objectives and must share the same start.
        for r in records:
            assert r['methods']['B']['stages']['rigid']==r['methods']['C']['stages']['rigid']
    data=dict(reference='Fixed automatic observations; not ground truth',aggregation='Mean of six view RMSEs, or twelve hand/view RMSEs',frames=records,summary=summary,comparisons=comparisons,source_sha256=source_hashes)
    dst=OUT/'partial_metrics.json' if a.partial else RUN/'evidence/metrics.json'
    dst.write_text(json.dumps(data,indent=2)+'\n')
    if not a.partial:
        (RUN/'site/metrics.json').write_text(json.dumps({k:v for k,v in data.items() if k!='source_sha256'},indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
