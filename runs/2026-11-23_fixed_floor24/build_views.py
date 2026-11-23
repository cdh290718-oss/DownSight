"""Render the published fixed-trunk choice from a reloaded OBJ in ground-aligned views."""
import sys
import shutil
import cv2
import numpy as np
import torch
from sample import ROOT, RUN, OLD, PRIOR, OUT, MANIFEST, grouped, read, write, sha

sys.path.insert(0,str(OLD))
import render_gallery as renderer
from patch_match import estimate as patch
g,floor=renderer.g,renderer.floor
BANK=ROOT/'outputs/fullbody_joint_candidates_20261123'
VAL=ROOT/'outputs/stereo_wrist_candidates_20261123/validation_fit'
EVID=RUN/'evidence'

def rank(q):
    penalty=max(q['checker_angle_deg']/5,q['checker_distance_diff_mm']/50,q['sparse_median_residual_mm']/50,q['sparse_vs_dense_deg']/10)
    return q['accepted'],q['ransac_inliers']>=8,-penalty

def estimate_ground(images,cams,T,regions):
    height,width=images[0].shape[:2]
    polys=[np.rint(np.array(poly)*[width/300,height/300]).astype(np.int32) for poly in regions]
    attempts={}; options=[]
    for name,fn in [('sift',floor.estimate),('epipolar_patch',patch)]:
        try:
            up,d,q,_=fn(images,cams,T,polys)
            assert np.isfinite(up).all() and np.isclose(up@up,1) and np.isfinite(d)
        except (ValueError,AssertionError,cv2.error,np.linalg.LinAlgError) as exc:
            attempts[name]=dict(error=type(exc).__name__,reason=str(exc).split('\n')[0][:200])
            continue
        options.append((up,d,q,name))
        attempts[name]=q
        if q['accepted']:
            break
    if not options:
        from dense_plane import estimate as dense
        up,d,q=dense(images,cams,T,polys)
        attempts['dense_sweep']=q
        return dict(up=up.tolist(),height_m=d,accepted=False,method='dense_sweep',attempts=attempts)
    up,d,q,method=max(options,key=lambda opt:rank(opt[2]))
    return dict(up=up.tolist(),height_m=d,accepted=q['accepted'],method=method,attempts=attempts)

def main(split, only=()):
    cv2.setNumThreads(1);torch.set_num_threads(1)
    archive=read(OLD/'evidence/renders_floor.json')
    archived={r['pair']:r for r in archive['rows']}
    sample=[r for r in read(EVID/'sample.json')['rows'] if r['split']==split]
    search=read(PRIOR/('evidence/'+split+'_search.json'))
    results=read(PRIOR/('evidence/'+split+'_results.json'))
    folder=g.SOURCE if split=='test' else VAL
    files=[BANK/(split+'_bank.npz'),BANK/(split+'_choices.json'),folder/'prepared.json',MANIFEST[split]]
    for path,digest in zip(files,[search['parameter_sha256'],search['choices_sha256'],search['prepared_sha256'],results['manifest_sha256']]):
        assert sha(path)==digest
    pack=dict(np.load(files[0]))
    choices,items=read(files[1]),read(files[2])['items']
    eyes=grouped(split)
    refs=g.DATA/'reference.json' if split=='test' else VAL.parent/'validation_truth.json'
    assert sha(refs)==results['truth_sha256']
    truth=read(refs)
    regions=read(RUN/'regions.json')
    model,cam=g.common.setup_model()
    layer=model.layer['neutral'].cpu().eval()
    faces=np.asarray(model.face,np.int32)
    cams=[]
    for side in ('left','right'):
        path=g.DATA/('camera_'+side+'.json')
        assert sha(path)==archive['calibration_sha256'][path.name]
        cams.append(floor.RawCamera(cam(path),read(path)['input_adapter']))
    for s in sample:
        pair=s['pair']
        if only and pair not in only:
            continue
        r=choices[pair]; base=r['offset']; idx=base+r['fixed']
        assert r['fixed']==int(np.argmin(r['loss'][:3])) and r['names'][2]=='fixed'
        assert all(np.array_equal(pack[k][base],pack[k][idx]) for k in ['betas','global0','delta','translation'])
        unchanged=[i for i in range(21) if i not in (0,1,3,4,15,16,17,18)]
        assert np.array_equal(pack['body_pose'][base,unchanged],pack['body_pose'][idx,unchanged])
        item=items[r['index']]
        assert item['pair']==pair and item['condition']==('rw20000' if split=='test' else 'baseline')
        src={side:MANIFEST[split].parent/eyes[pair][side]['image'] for side in ('left','right')}
        assert [sha(path) for path in src.values()]==s['image_sha256']
        images=[cv2.imread(str(path)) for path in src.values()]
        if split=='test':
            rec=archived[pair]
            assert s['image_sha256']==rec['image_sha256'] and rec['subject']==s['subject']
            ground=dict(up=rec['up'],height_m=rec['height_m'],accepted=rec['validation']['accepted'],method='archived_'+rec['method'])
        else:
            T=np.asarray(item['inputs']['views']['right']['camera']['left_to_view'])
            ground=estimate_ground(images,cams,T,regions[pair])
            rechecks=read(RUN/'regions_recheck.json')
            if not ground['accepted'] and pair in rechecks:
                enlarged=[(np.asarray(poly)*300/450).tolist() for poly in rechecks[pair]]
                revised=estimate_ground(images,cams,T,enlarged)
                def strength(gnd):
                    q=gnd['attempts'][gnd['method']]
                    if gnd['method']=='dense_sweep':
                        return (False,False,-1e6)
                    return rank(q)
                use_revision=bool(strength(revised)>strength(ground))
                original=ground
                ground=dict(revised if use_revision else original)
                ground.update(revised_used=use_revision,first_pass=original,recheck=revised)
        body={k:pack[n][idx] for k,n in [('global_orient','global0'),('body_pose','body_pose'),('betas','betas'),('transl','translation')]}
        v,j,head,chest=g.geometry(layer,body,pack['delta'][idx],False,None)
        assert np.allclose(head,chest)
        pred=j[g.common.NATIVE_IDS][g.common.OBS]
        err=np.mean(np.linalg.norm(pred-np.asarray(truth[pair]['gt'])[g.common.OBS],axis=1))*1000
        expected=results['rows'][pair]['scores']['fixed']['mpjpe']
        assert abs(err-expected)<.001
        local=OUT/'gallery'/pair
        local.mkdir(parents=True,exist_ok=True)
        obj=local/(pair+'.obj')
        with obj.open('w') as f:
            f.write('# SMPL-X surface in left camera CV coordinates, meters\n')
            np.savetxt(f,v,fmt='v %.9f %.9f %.9f')
            np.savetxt(f,faces+1,fmt='f %d %d %d')
        v2,f2=g.load_obj(obj)
        assert np.array_equal(faces,f2) and np.max(abs(v2-v))<1e-8
        forward=(g.rot(pack['delta'][idx])@g.rot(pack['global0'][idx]))[:,2]
        renders=list(renderer.render_views(v2,f2,np.asarray(ground['up']),forward))
        asset=RUN/'assets'/pair
        for name,im in zip(['front','side'],renders):
            assert np.count_nonzero(im[:,:,0]<235)>300
            g.save(asset/(name+'.jpg'),im)
            shutil.copyfile(asset/(name+'.jpg'),local/(name+'.jpg'))
        for (name,path),im in zip(src.items(),images):
            shutil.copyfile(path,local/(name+'.png'))
            assert sha(path)==sha(local/(name+'.png'))
            g.save(asset/(name+'.jpg'),cv2.resize(im,(436,436),interpolation=cv2.INTER_AREA))
        changed=not np.array_equal(pack['body_pose'][idx],pack['body_pose'][base])
        write(OUT/(pair+'.json'),dict(**s,ground=ground,choice=r['names'][r['fixed']],changed_from_golden=changed,
              fixed_objective=r['loss'][r['fixed']],verified_mpjpe_mm=float(err),expected_mpjpe_mm=expected,
              front_forward_left_cv=forward.tolist(),obj_sha256=sha(obj),vertices=len(v2),faces=len(f2),
              assets={p.name:sha(p) for p in sorted(asset.iterdir())}))
        print(pair,ground['method'],ground['accepted'],flush=True)

if __name__=='__main__':
    main(sys.argv[1],sys.argv[2:])
