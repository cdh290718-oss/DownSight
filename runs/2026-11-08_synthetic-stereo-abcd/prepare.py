"""Extract cameras, Head anchors, D-only shape, and automatic 2D observations."""
import hashlib,itertools,json
import cv2
import numpy as np
from common import ROOT,RUN,OUT,OLD,read
from egosmplx_pipeline.body_common import JOINT_MAPPING
D=OLD/'dataset/samples/000001';cfg=read(RUN/'settings.json')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
names=read(OLD/'dataset/joint_names_egosmplx65.json')['names']
with np.load(D/'geometry.npz') as z:objectR=z['object_matrix_world'][:3,:3];stored_stereo=z['left_to_right_cv']
with np.load(D/'smplx_reconstruction.npz') as z:
    hr=cv2.Rodrigues(z['global_orient'].astype(float))[0]
    for j in [3,6,9,12,15]:hr=hr@cv2.Rodrigues(z['body_pose'].reshape(21,3)[j-1].astype(float))[0]
    oracle_shape=z['betas'].tolist()
with np.load(D/'left_vertices.npz') as z:world_left=z['world_to_camera_cv'];hp=z['joints_camera_cv_m'][15];headR=world_left[:3,:3]@objectR@hr
payload=dict(head_position_m=hp.tolist(),head_rotation_matrix=headR.tolist(),views={},source_hashes={n:sha(D/n) for n in ['geometry.npz','smplx_reconstruction.npz','left_vertices.npz','right_vertices.npz']},scope='Only cameras and Head frame from GT; D shape in separate file; no GT body pose, other joint locations or segmentation in fitting inputs')
for side,dirname in [('left','left'),('right','right_cpu')]:
    job=read(OLD/'fits'/dirname/'jobs'/(side+'.json'));obs_path=__import__('pathlib').Path(job['row']['predictions']['sapiens']);obs=read(obs_path);lookup={q['name']:q for q in obs['keypoints']}
    with np.load(D/(side+'_vertices.npz')) as z:
        T=z['world_to_camera_cv']@np.linalg.inv(world_left)
        camera=dict(K=z['K_zero_based'].tolist(),distortion=z['distortion'].tolist(),left_to_view=T.tolist())
    if side=='right':assert np.max(abs(T-stored_stereo))<1e-6
    selected=[n for n in JOINT_MAPPING if n!='nose' and n in lookup and np.isfinite([lookup[n]['x'],lookup[n]['y'],lookup[n]['score']]).all() and lookup[n]['score']>=.3 and 0<=lookup[n]['x']<1280 and 0<=lookup[n]['y']<720]
    points=[dict(name=n,index=names.index(JOINT_MAPPING[n]),xy=[lookup[n]['x'],lookup[n]['y']],weight=min(1,lookup[n]['score'])**2,group='shoulder' if n.endswith('_shoulder') else 'wrist' if n.endswith('_wrist') else 'body') for n in selected]
    wp=__import__('pathlib').Path(job['row']['predictions']['wilor'])
    with np.load(wp) as w:
        assert str(w['image_sha256'].item())==obs['image_sha256']
        candidates=[]
        for i in range(len(w['joints_2d'])):
            candidates.append(dict(candidate=i,side='right' if w['is_right'][i]>.5 else 'left',confidence=float(w['detector_confidence'][i]),xy=w['joints_2d'][i].tolist()))
    # Globally injective assignment, with explicit handedness and observed wrist gate.
    choices=[]
    for assign in itertools.product(range(-1,len(candidates)),repeat=2):
        used=[i for i in assign if i>=0]
        if len(used)!=len(set(used)):continue
        cost=0.;valid=True;matches={}
        for hand,i in zip(['left','right'],assign):
            if i<0:cost+=cfg['wilor_match_max_wrist_px'];continue
            c=candidates[i];q=lookup[hand+'_wrist'];distance=float(np.linalg.norm(np.array(c['xy'][0])-[q['x'],q['y']]))
            if c['side']!=hand or c['confidence']<cfg['wilor_min_confidence'] or distance>cfg['wilor_match_max_wrist_px']:valid=False;break
            cost+=distance;matches[hand]=dict(c,body_wrist_distance_px=distance)
        if valid:choices.append((cost,assign,matches))
    _,_,matches=min(choices,key=lambda x:(x[0],x[1]))
    payload['views'][side]=dict(camera=camera,body_points=points,hands=matches,image_sha256=obs['image_sha256'],observation_sha256=sha(obs_path),wilor_sha256=sha(wp),raw_source_sha256=sha(__import__('pathlib').Path(job['output'])/'raw/params.npz'))
# Epipolar check is diagnostic only: A must not receive stereo point-selection information.
cl,cr=[payload['views'][s]['camera'] for s in ['left','right']];T=np.array(cr['left_to_view']);R,t=T[:3,:3],T[:3,3]
tx=np.array([[0,-t[2],t[1]],[t[2],0,-t[0]],[-t[1],t[0],0]]);E=tx@R
for hand in ['left','right']:
    if not all(hand in payload['views'][s]['hands'] for s in ['left','right']):continue
    hs=[payload['views'][s]['hands'][hand] for s in ['left','right']]
    rays=[]
    for h,c in zip(hs,[cl,cr]):
        u=cv2.undistortPointsIter(np.array(h['xy'])[:,None,:],np.array(c['K']),np.array(c['distortion']),None,None,(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,50,1e-12)).reshape(-1,2);rays.append(np.column_stack([u,np.ones(len(u))]))
    l= rays[0]@E.T;r=rays[1]@E
    residual=np.abs((rays[1]*l).sum(1));err=.5*residual*(cr['K'][0][0]/np.linalg.norm(l[:,:2],axis=1)+cl['K'][0][0]/np.linalg.norm(r[:,:2],axis=1))
    mask=(err<=cfg['wilor_max_epipolar_px'])
    for h in hs:
        xy=np.array(h['xy']);mask&=np.isfinite(xy).all(1)&(xy[:,0]>=0)&(xy[:,0]<1280)&(xy[:,1]>=0)&(xy[:,1]<720)
    for h in hs:h.update(stereo_passed_indices=np.flatnonzero(mask).tolist(),epipolar_symmetric_px=err.tolist())
for view in payload['views'].values():
    for h in view['hands'].values():
        xy=np.array(h['xy']);good=np.isfinite(xy).all(1)&(xy[:,0]>=0)&(xy[:,0]<1280)&(xy[:,1]>=0)&(xy[:,1]<720)
        h['included_indices']=np.flatnonzero(good).tolist()
OUT.mkdir(exist_ok=True);(OUT/'inputs.json').write_text(json.dumps(payload,indent=2)+'\n');(RUN/'evidence/inputs.json').write_text(json.dumps(payload,indent=2)+'\n')
(OUT/'oracle_shape.json').write_text(json.dumps(dict(betas=oracle_shape,source_sha256=sha(D/'smplx_reconstruction.npz')))+'\n')
print('prepared', {s:{h:len(v['included_indices']) for h,v in q['hands'].items()} for s,q in payload['views'].items()})
