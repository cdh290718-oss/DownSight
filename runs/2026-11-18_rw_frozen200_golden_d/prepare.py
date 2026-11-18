"""Identity checks and annotation-assisted shared nuisance inputs, before fitting."""
import collections,copy
import cv2,numpy as np,torch
from common import ROOT,RUN,OUT,EVIDENCE,NAMES,MODEL_IDS,NATIVE_IDS,OBS,read,write,sha,camera_files
from egosmplx_pipeline.camera import FishEyeCameraCalibrated
from geometry import project

def main():
    manifest=ROOT/'outputs/rw_test_completion_20261117/test_manifest.json'
    rows=read(manifest)['records'];assert len(rows)==200
    pairs={}
    for r in rows:
        assert sha(manifest.parent/r['image'])==r['image_sha256']
        assert r['view'] not in pairs.setdefault(r['pair_id'],{})
        pairs[r['pair_id']][r['view']]=r
    assert len(pairs)==100 and all(set(v)=={'left','right'} for v in pairs.values())
    ds=ROOT.parent/'EgoSMPLX/data/UnrealEgo_RW_Test/prepared/samples.npz'
    z=np.load(ds);seq=z['source_sequence'];frames=z['frame_index'];sides=z['camera_side']
    xyz=z['keypoints3d'];head=z['keypoints3d_fit'][:,0]
    for r in rows:
        i=r['source_index']
        assert seq[i]==r['sequence'] and frames[i]==r['frame'] and sides[i]==r['view']
    # A fixed stereo rig is recovered independently for each recording session.
    # Prefer excluded sequences; otherwise disclose same-sequence, excluded-frame calibration.
    sessions=sorted({r['sequence'].rsplit('/',1)[0] for r in rows})
    excluded={r['sequence'] for r in rows};excluded_frames={(r['sequence'],r['frame']) for r in rows};lookup={(str(s),int(f),str(v)):i for i,(s,f,v) in enumerate(zip(seq,frames,sides))}
    rigs={};rig_evidence=[]
    for session in sessions:
        candidates=[(str(s),int(f),i) for i,(s,f,v) in enumerate(zip(seq,frames,sides)) if v=='left' and str(s).rsplit('/',1)[0]==session and s not in excluded and (str(s),int(f),'right') in lookup]
        if not candidates:
            # Some sessions have all sequences represented in the frozen set.
            # Use a different frame, disclose the weaker separation explicitly.
            candidates=[(str(s),int(f),i) for i,(s,f,v) in enumerate(zip(seq,frames,sides)) if v=='left' and str(s).rsplit('/',1)[0]==session and (str(s),int(f)) not in excluded_frames and (str(s),int(f),'right') in lookup]
        assert candidates,session
        s,f,li=sorted(candidates)[0];ri=lookup[(s,f,'right')]
        a,b=xyz[li].astype(float),xyz[ri].astype(float);am,bm=a.mean(0),b.mean(0)
        u,_,vh=np.linalg.svd((a-am).T@(b-bm));R=vh.T@np.diag([1,1,np.linalg.det(vh.T@u.T)])@u.T;t=bm-R@am
        T=np.eye(4);T[:3,:3]=R;T[:3,3]=t;rigs[session]=T
        residual=[]
        for p in pairs.values():
            if p['left']['sequence'].rsplit('/',1)[0]!=session:continue
            x,y=xyz[p['left']['source_index']],xyz[p['right']['source_index']]
            residual.extend(np.linalg.norm(x@R.T+t-y,axis=1)*1000)
        assert max(residual)<.01,(session,max(residual))
        rig_evidence.append(dict(session=session,calibration_sequence=s,calibration_frame=f,calibration_sample_indices=[li,ri],calibration_sequence_excluded=s not in excluded,calibration_frame_excluded=(s,f) not in excluded_frames,left_to_right=T.tolist(),heldout_mean_residual_mm=float(np.mean(residual)),heldout_max_residual_mm=float(np.max(residual))))
    # Keep the reference D pixel scale (872 -> 720, pad horizontally to 1280).
    arch=ROOT.parent/'EgoFishPose/experiments/benchmark_body_hand10_head01_20261014_v1'
    calinfo=[];adapters=[];project_errors=[]
    for v,path in zip(['left','right'],camera_files()):
        src=arch/'body_calibration_fisheye'/('camera_'+v+'.json');c=read(src);adapters.append(c['input_adapter'])
        c['input_adapter'].pop('source_calibration',None);write(path,c);cam=FishEyeCameraCalibrated(path)
        err=[];eq=[]
        for r in rows:
            if r['view']!=v:continue
            gt=xyz[r['source_index']].astype(float);uv=z['keypoints2d'][r['source_index'],:,:2]*c['input_adapter']['scale']+np.array(c['input_adapter']['offset'])
            projected=cam.world2camera(gt);err.extend(np.linalg.norm(projected-uv,axis=1))
            eq.append(float(np.max(np.abs(project(torch.tensor(gt,dtype=torch.float32),cam).numpy()-projected))))
        assert max(eq)<.01
        calinfo.append(dict(view=v,source_sha256=sha(src),local_sha256=sha(path),projection_vs_official_mean_px=float(np.mean(err)),projection_vs_official_max_px=float(np.max(err)),torch_numpy_max_px=max(eq)))
        project_errors.extend(err)
    assert np.mean(project_errors)<2,'Check camera convention and original labels'
    cache=ROOT.parent/'EgoSMPLX/data/UnrealEgo_RW_Test/soft_fusion/unrealego_rw_test_snapshot11_root_cam.npz'
    net=np.load(cache);assert list(net['camera_side'])==['left','right']
    net_lookup={int(s):(k,v) for k,row in enumerate(net['sample_index']) for v,s in enumerate(row)}
    predpaths={m:ROOT/('outputs/rw_test_completion_20261117/'+m+'_test_frozen_predictions.json') for m in ['original','rw500','rw5000']}
    predpaths['rw20000']=ROOT/'outputs/rw20000_frozen200_20261118/rw20000_test_frozen_predictions.json'
    pred={m:{p['id']:p for p in read(path)} for m,path in predpaths.items()}
    summaries={}
    for m,pr in pred.items():
        assert set(pr)=={r['id'] for r in rows};errs=[]
        for r in rows:
            p=pr[r['id']];assert p['image_sha256']==r['image_sha256'];assert np.isfinite(p['xy']).all()
            mask=np.array(r['keypoints308_supervision_mask'],bool)
            errs.extend(np.linalg.norm(np.array(p['xy'])-np.array(r['keypoints308_xy']),axis=1)[mask])
        summaries[m]=dict(count=len(errs),mean_px=float(np.mean(errs)),pck20=float(np.mean(np.array(errs)<=20)*100))
    mapping={m['rw_index']:m['sapiens308_id'] for m in read(ROOT/'runs/2026-11-16_sapiens-unrealego-audit/evidence/audit.json')['joint_mapping'] if m['rw_index'] is not None}
    items=[];refs={};identities=[]
    for n,(pair,p) in enumerate(sorted(pairs.items())):
        tag='pair%03d'%(n+1);left,right=p['left'],p['right'];li=left['source_index'];ri=right['source_index'];k,v=net_lookup[li]
        assert v==0 and list(net['sample_index'][k])==[li,ri] and net['frame_index'][k]==left['frame'] and net['source_sequence'][k]==left['sequence']
        R=cv2.Rodrigues(net['root_pose'][k,0].astype(float))[0]
        for j in [3,6,9,12,15]:R=R@cv2.Rodrigues(net['body_pose'][k,0,j-1].astype(float))[0]
        assert np.allclose(R@R.T,np.eye(3),atol=1e-6) and np.linalg.det(R)>.999
        anchor=dict(position=head[li].astype(float).tolist(),rotation=R.tolist())
        refs[tag]=dict(gt=xyz[li].astype(float).tolist(),subject=left['subject'],sequence=left['sequence'],frame=left['frame'])
        identities.append(dict(pair=tag,subject=left['subject'],sequence=left['sequence'],frame=left['frame'],source_indices=[li,ri],image_ids=[left['id'],right['id']],image_sha256=[left['image_sha256'],right['image_sha256']],network_cache_row=k))
        views={'left':{'camera':{'left_to_view':np.eye(4).tolist()}},'right':{'camera':{'left_to_view':rigs[left['sequence'].rsplit('/',1)[0]].tolist()}}}
        for m,pr in pred.items():
            points=[]
            for g in OBS:
                sid=mapping[g];part=NAMES[g].split('_')[1].lower();xy=[];valid=[];scores=[]
                for vi,r in enumerate([left,right]):
                    q=pr[r['id']];xy.append((np.array(q['xy'][sid])*adapters[vi]['scale']+np.array(adapters[vi]['offset'])).tolist());valid.append(bool(r['keypoints308_supervision_mask'][sid]));scores.append(float(q['score'][sid]))
                points.append(dict(name=NAMES[g],index=MODEL_IDS[g],native_index=NATIVE_IDS[g],part=part,group='shoulder' if part=='shoulder' else 'wrist' if part=='wrist' else 'body',xy=xy,valid=valid,scores=scores,accepted=False))
            items.append(dict(pair=tag+'_'+m,base_pair=tag,detector=m,condition='baseline',subject=left['subject'],points=points,anchor=copy.deepcopy(anchor),inputs={'views':copy.deepcopy(views)}))
    write(OUT/'prepared.json',dict(items=items));write(OUT/'reference.json',refs)
    write(EVIDENCE/'preparation.json',dict(command='<BODY_PYTHON> prepare.py',exit_status=0,manifest_sha256=sha(manifest),dataset_sha256=sha(ds),prediction_sha256={m:sha(p) for m,p in predpaths.items()},network_cache_sha256=sha(cache),network_checkpoint_sha256=read(cache.with_suffix('.json'))['checkpoint_sha256'],network_head_source='left-view image-network root and body rotations; FK chain 3,6,9,12,15; no fitted body or shape initialization',head_position_source='official Head annotation',body_3d_access='preparation reads annotations for excluded-frame calibration and coordinate audit; separate local reference is read only by post-fit evaluation, never optimizer',rigs=rig_evidence,cameras=calinfo,recomputed_2d=summaries,pairs=identities,total_fits=len(items),subject_pairs=dict(collections.Counter(r['subject'] for r in refs.values())),fit_sha256=sha(RUN/'fit.py'),geometry_sha256=sha(RUN/'geometry.py'),protocol_sha256=sha(RUN/'protocol.json'),prepared_local_sha256=sha(OUT/'prepared.json'),reference_local_sha256=sha(OUT/'reference.json')))
    print('Prepared',len(items),'fits. Recomputed 2D:',summaries,flush=True)

if __name__=='__main__':main()
