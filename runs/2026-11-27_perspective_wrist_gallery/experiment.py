"""Matched SMPL-X fits on four real stereo midpoints. Indices refer to the SMPL-X pose pack."""
from pathlib import Path
import sys
import cv2
import numpy as np
import torch
from scipy.optimize import least_squares

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
OUT = ROOT/'outputs/perspective_wrist_gallery_20261127'
RECENT = RUN.parent/'2026-11-27_vidihand2_comparison'
RECENT_OUT = ROOT/'outputs/vidihand2_comparison_20261127'
sys.path.insert(0,str(RUN.parent/'2026-11-27_stereo_hand_selection'))
from render_selection import stereo, title, rot, groups, free_render, o3d
from choose import old, SIDES, DIGITS, TIPS, JOINT_NAMES, read, write, sha
from compare_fit import FINGER_JOINTS, ROOTS, paired, ARM_OUT
ref = old.ref
IDS = np.array([[JOINT_NAMES.index(side+'_'+n) for n in ['wrist']+[d+s for d in DIGITS for s in ('1','2','3','')]] for side in SIDES])
KEYS = ('clip001','clip002','clip003','clip004')

def setup():
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    ref.prev.ref.camera_files = ref.prev.base.fit.camera_files = ref.cameras
    ref.compare.OUT = ref.OUT
    assert [sha(p) for p in ref.cameras()]==read(ref.PRIOR/'evidence/calibration.json')['converted']
    for file,expected in [('prepared',read(ref.PRIOR/'evidence/inputs.json')['shards']['0']),
                          ('protocol',read(ref.PRIOR/'evidence/inputs.json')['protocol'])]:
        assert sha(ref.OLD/'fit/0'/(file+'.json'))==expected
    return ref.prev.ref.common.setup_model()

def inputs(key):
    i = KEYS.index(key)
    if i<2:
        file = paired.palm.OUT/'poses'/(key+'.npz')
        meta = read(paired.PREV/'evidence'/(key+'.json'))
        assert sha(file)==meta['pose_hash']
    else:
        file = ARM_OUT/'palm/poses'/(key+'.npz')
        assert sha(file)==read(old.BASE/'evidence'/(key+'.json'))['seed_sha256']
        meta = read(ARM_OUT/'palm/evidence'/(key+'.json'))
    row = read(ref.OLD/'fit/0/prepared.json')['items'][i]
    assert row['pair']==key
    det_path = ref.OLD/'predictions.json'
    assert sha(det_path)==read(ref.PRIOR/'evidence/detection.json')['predictions_sha256']
    det = read(det_path)
    order = np.array([41]+[n for a in range(21,41,4) for n in range(a+3,a-1,-1)])
    obs = np.array([[np.array(det[key+'_'+eye+'.png']['xy'])[order+shift] for shift in (21,0)] for eye in SIDES])
    inside = ((obs>=0)&(obs<[1280,720])).all(-1)
    dist = np.linalg.norm(obs-obs[:,:,:1],axis=-1)
    used = (inside & (dist<3.2*np.median(dist[:,:,ROOTS],axis=2)[:,:,None]))[:,:,FINGER_JOINTS]
    return dict(np.load(file)),row,obs,inside,used,[meta['metrics']['after'][s]['quality'] for s in SIDES],sha(file)

def baseline(key,model,Camera):
    pack,row,obs,inside,used,quality,digest = inputs(key)
    x0 = torch.tensor(pack['locked'])
    start = dict(body_pose=pack['locked'][:21],delta=pack['locked'][21],translation=pack['locked'][22],
                 global0=pack['global0'],betas=pack['betas'])
    for eye in SIDES:
        name = key+'_'+eye
        assert sha(ref.OUT/'seg'/(name+'.npy'))==read(paired.palm.V4/'evidence/segmentation.json')['eyes'][name]['sampled_sha256']
    frame = ref.FixedFeet(key,row,start,model,Camera,read(ref.OLD/'fit/0/protocol.json'))
    movable = torch.tensor(paired.WRISTS+list(range(23,53)))
    free = x0[movable].clone().requires_grad_(True)
    solver = torch.optim.Adam([free],lr=.018)
    mask = torch.tensor(used)
    targets = x0.new_tensor(obs[:,:,FINGER_JOINTS])
    qualities = torch.tensor(quality)
    best_loss = float('inf')
    for step in range(paired.STEPS+1):
        x = x0.index_copy(0,movable,free)
        verts,j = paired.body(frame,x)
        points = j[torch.as_tensor(IDS)].reshape(-1,3)
        uv = torch.stack([paired.project(points@e.transform_t[:3,:3].T+e.transform_t[:3,3],e.cam).reshape(2,21,2) for e in frame.eyes])
        dist = (uv[:,:,FINGER_JOINTS]-targets).norm(dim=-1)
        robust = 2*(torch.sqrt(1+(dist/6)**2)-1)
        points_loss = (robust*mask).sum((0,2))/mask.sum((0,2)).clamp_min(1)
        edge = frame.observed(verts,True)[0]
        loss = .01*(qualities*points_loss).mean()+ref.WEIGHT*edge+.003*(x[paired.WRISTS]-x0[paired.WRISTS]).square().mean()+.0008*(x[23:]/.52).square().mean()
        assert torch.isfinite(loss)
        if loss.item()<best_loss:
            best_loss,best_step,best = loss.item(),step,x.detach().clone()
        if step<paired.STEPS:
            solver.zero_grad()
            loss.backward()
            solver.step()
            with torch.no_grad():
                cap = free.new_tensor(np.deg2rad([100]*2+[45]*30))[:,None]
                free.mul_((cap/free.norm(dim=1,keepdim=True).clamp_min(1e-8)).clamp_max(1))
    path = OUT/'poses'/(key+'_B.npz')
    np.savez_compressed(path,**dict(pack,locked=best.numpy()))
    write(RUN/'evidence'/(key+'_B.json'),dict(source_sha256=digest,pose_sha256=sha(path),loss=best_loss,step=best_step))
    print(key,'B',best_step,flush=True)

def choose_hand(key,i,layer,pack,means,rest,world,project,target,used):
    side = SIDES[i]
    seed = pack['locked']
    finger = slice(23+15*i,38+15*i)
    mirror = np.diag([-1,1,1]) if i==0 else np.eye(3)
    wrist = 20+i
    parent = world[18+i]
    init = old.mat(seed[wrist-1])
    candidates = {}
    params = {}
    if not used.any():
        return None,dict(source='no observations',candidates=candidates)
    new = key in KEYS[:2]
    receipts = read(RUN/'evidence/predictions.json') if new else read(RECENT/'evidence/wilor.json')['views']
    for n,eye in enumerate(SIDES):
        view = key+'_'+eye
        source = 'cam'+str(n+3)
        if side not in receipts[view]['sha256']:
            continue
        path = (OUT if new else RECENT_OUT)/'poses'/f'wilor_{view}_{side}.npz'
        assert sha(path)==receipts[view]['sha256'][side]
        vals = np.load(path)
        q = old.palm_basis(rest[IDS[i]])@old.palm_basis(vals['rest']).T
        local_rot = q@mirror@vals['pose']@mirror@q.T
        x = seed.copy()
        x[finger] = old.rv(local_rot)-means[15*i:15*(i+1)]
        assert np.max(abs(old.mat(x[finger]+means[15*i:15*(i+1)])-local_rot))<1e-4
        _,j = old.mesh(layer,x,pack)
        pivot = j[wrist]
        local = (j[IDS[i]]-pivot)@world[wrist]
        def residual(r):
            pred = project(local@(parent@old.mat(r)).T+pivot)
            delta = ((pred-target)[:,FINGER_JOINTS]/6)[used]
            return np.r_[delta.ravel(),old.rv(init.T@old.mat(r))/.65]
        sol = least_squares(residual,x[wrist-1],loss='soft_l1',max_nfev=180)
        assert sol.success
        x[wrist-1] = old.rv(old.mat(sol.x))
        _,j = old.mesh(layer,x,pack)
        err = np.linalg.norm(project(j[IDS[i]])-target,axis=-1)[:,FINGER_JOINTS]
        score = float(np.mean([np.mean(2*(np.sqrt(1+(e[m]/6)**2)-1)) for e,m in zip(err,used) if m.any()]))
        params[source] = x
        candidates[source] = dict(score=score,evaluations=sol.nfev,source_sha256=sha(path))
    if not candidates:
        return None,dict(source='no candidate',candidates=candidates)
    source = min(candidates,key=lambda s:candidates[s]['score'])
    return params[source],dict(source=source,candidates=candidates)

def evaluate(key,layer):
    pack,row,obs,inside,used,_,digest = inputs(key)
    seed = pack['locked']
    new = key in KEYS[:2]
    bfile = (OUT if new else old.BASE_OUT)/'poses'/(key+'_B.npz')
    brec = read(RUN/'evidence'/(key+'_B.json')) if new else read(old.BASE/'evidence'/(key+'.json'))['results']['B']
    assert sha(bfile)==brec['pose_sha256']
    base = dict(np.load(bfile))
    assert all(np.array_equal(base[k],pack[k]) for k in ('betas','global0'))
    means = layer.pose_mean.detach().numpy().reshape(-1,3)[25:]
    neutral = np.zeros_like(seed)
    neutral[23:] = -means
    _,rest = old.mesh(layer,neutral,dict(pack,global0=np.zeros(3)))
    world = [old.mat(seed[21])@old.mat(pack['global0'])]
    for n in range(1,22):
        world.append(world[int(layer.parents[n])]@old.mat(seed[n-1]))
    tr = [np.array(row['inputs']['views'][e]['camera']['left_to_view']) for e in SIDES]
    cams = list(map(read,ref.cameras()))
    def project(points):
        return np.stack([old.projection(points@t[:3,:3].T+t[:3,3],c) for c,t in zip(cams,tr)])
    selected = seed.copy()
    hands = {}
    for i,side in enumerate(SIDES):
        x,h = choose_hand(key,i,layer,pack,means,rest,world,project,obs[:,i],used[:,i])
        if x is not None:
            finger = slice(23+15*i,38+15*i)
            selected[finger] = x[finger]
            selected[19+i] = x[19+i]
        hands[side] = h
        h['fit_points_by_eye'] = used[:,i].sum(1).tolist()
    np.savez_compressed(OUT/'poses'/(key+'_wrist.npz'),**dict(pack,locked=selected))
    models = {k:old.mesh(layer,x,pack) for k,x in [('B',base['locked']),('wrist',selected)]}
    _,start = old.mesh(layer,seed,pack)
    drift = max(np.linalg.norm(j[:22]-start[:22],axis=1).max()*1000 for _,j in models.values())
    assert drift < .001
    for i,side in enumerate(SIDES):
        h = hands[side]
        h['tips_by_eye'] = inside[:,i,TIPS].sum(1).tolist()
        for mode,(v,j) in models.items():
            uv = project(j[IDS[i]])
            mask = inside[:,i][:,TIPS]
            errors = np.linalg.norm(uv-obs[:,i],axis=2)[:,TIPS][mask]
            h[mode] = dict(errors=errors.tolist(),rmse=old.rms(errors) if len(errors) else None)
    return models,tr,rot(seed[21])@rot(pack['global0']),dict(hands=hands,body_max_mm=float(drift),seed_sha256=digest,baseline_sha256=sha(bfile))

if __name__=='__main__' and sys.argv[1:] == ['baseline']:
    model,Camera = setup()
    for key in KEYS[:2]:
        baseline(key,model,Camera)

def render(key,models,transforms,root,record,rays,faces,parts,images):
    stereo.RUN,stereo.OUT = RUN,OUT
    for mode,(verts,joints) in list(models.items()):
        verts,file = stereo.export(key,mode,verts,faces)
        record.setdefault('obj_sha256',{})[mode] = sha(file)
        models[mode] = (verts,joints)
    labels = {'B':'Full B','wrist':'WiLoR + wrist fit'}
    body = []
    for angle,name in [(0,'Front'),(np.pi/2,'Side')]:
        axes = root@rot([0,angle,0])
        xyz = np.vstack([v@axes for v,j in models.values()])
        center = (xyz.min(0)+xyz.max(0))/2
        scale = min(315/np.ptp(xyz[:,0]),440/np.ptp(xyz[:,1]))
        for mode,(v,j) in models.items():
            body.append(title(free_render(v,faces,axes,center,scale),f'{key} | {name} | {labels[mode]}'))
    stereo.save(images,key+'_body',np.hstack(body))
    handrows = []
    for i,side in enumerate(SIDES):
        subset = faces[parts==side[0].upper()+'_hand']
        indices = np.unique(subset)
        across,up,normal = old.palm_basis(models['B'][1][IDS[i]]).T
        tiles = []
        for degrees in (12,78):
            a = np.deg2rad(degrees)
            axes = np.column_stack((np.cos(a)*across+np.sin(a)*normal,up,-np.cos(a)*normal+np.sin(a)*across))
            xyz = np.vstack([v[indices]@axes for v,j in models.values()])
            center = (xyz.min(0)+xyz.max(0))/2
            scale = 225/np.ptp(xyz[:,:2],axis=0).max()
            winding = subset[:,::-1] if np.linalg.det(axes)<0 else subset
            for mode,(v,j) in models.items():
                im = free_render(v,winding,axes,center,scale)[112:368,47:303]
                tiles.append(title(im,f'{side} {degrees} deg | {labels[mode]}'))
        handrows.append(np.hstack(tiles))
    stereo.save(images,key+'_hands',np.vstack(handrows))
    photos = read(ref.PRIOR/'evidence/frames.json')[KEYS.index(key)]['views']
    closeups = []
    for n,(eye,t,ray) in enumerate(zip(SIDES,transforms,rays)):
        file = ref.OLD/'images'/f'{key}_{eye}.png'
        assert sha(file)==photos[eye]['frame']
        photo = cv2.imread(str(file))
        view = f'cam{n+3}'
        masks = {}
        full = []
        for mode,(v,j) in models.items():
            part,valid,shade = stereo.raycast(v@t[:3,:3].T+t[:3,3],faces,parts,ray)
            masks[mode] = part
            overlay = photo.copy()
            overlay[valid] = (.52*overlay[valid]+.48*shade[valid,None]).astype('uint8')
            full.append(title(overlay,view+' | '+labels[mode]))
        stereo.save(images,key+'_'+view,np.hstack(full))
        for i,side in enumerate(SIDES):
            y,x = np.where((masks['B']==i+1)|(masks['wrist']==i+1))
            if not len(x):
                closeups.append(np.hstack([title(np.full((245,245,3),40,np.uint8),s) for s in (view+' '+side,'no visible mesh','no visible mesh')]))
                continue
            center = np.array([x.min()+x.max(),y.min()+y.max()])/2
            span = int(np.clip(max(np.ptp(x),np.ptp(y))*1.35+35,160,720))
            l,top = np.clip(np.rint(center-span/2).astype(int),0,[1280-span,720-span])
            def tile(im,label):
                return title(cv2.resize(im[top:top+span,l:l+span],(245,245)),view+' '+side+' | '+label)
            tiles = [tile(photo,'Photo')]
            for mode,color in [('B',(25,190,250)),('wrist',(60,220,95))]:
                im = photo.copy()
                stereo.edge(im,masks[mode]==i+1,color)
                tiles.append(tile(im,labels[mode]))
            closeups.append(np.hstack(tiles))
    stereo.save(images,key+'_photo',np.vstack(closeups))

def gallery():
    model,Camera = setup()
    layer = model.layer['neutral'].eval()
    faces = np.asarray(model.face,dtype=np.int32)
    parts = groups(layer,faces)
    y,x = np.indices((720,1280))
    xy = np.stack([x,y],axis=-1).reshape(-1,2)
    rays = []
    for cal in ref.cameras():
        ray = np.zeros((720,1280,6),np.float32)
        ray[:,:,3:] = Camera(cal).camera2world_ray(xy).reshape(720,1280,3)
        rays.append(o3d.core.Tensor(ray))
    images,metrics = {},{}
    for key in KEYS:
        models,t,root,rec = evaluate(key,layer)
        render(key,models,t,root,rec,rays,faces,parts,images)
        metrics[key] = rec
        print(key,[(h['source'],h['B']['rmse'],h['wrist']['rmse']) for h in rec['hands'].values()],flush=True)
    pooled = {}
    for tag,keys in [('001_002',KEYS[:2]),('003_004',KEYS[2:])]:
        pooled[tag] = {}
        for mode in ('B','wrist'):
            gaps = [e for key in keys for h in metrics[key]['hands'].values() for e in h[mode]['errors']]
            pooled[tag][mode] = dict(n=len(gaps),rmse=old.rms(gaps))
    write(RUN/'evidence/results.json',dict(metrics=metrics,pooled=pooled,images=images,exit_status=0))

if __name__=='__main__' and sys.argv[1:]==['gallery']:
    gallery()
