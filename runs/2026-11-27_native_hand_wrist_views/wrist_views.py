"""Compare one shared source per hand across both calibrated views."""
from pathlib import Path
import sys
import cv2
import numpy as np
import open3d as o3d
import torch

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
OUT = ROOT/'outputs/native_hand_wrist_views_20261127'
SELECT = RUN.parent/'2026-11-27_stereo_hand_selection'
NATIVE = RUN.parent/'2026-11-27_wilor_native_hands'
sys.path.insert(0,str(SELECT))
from render_selection import stereo, title, full_body, rot, groups
from choose import old, sha, read, write, SIDES, TIPS, DIGITS, JOINT_NAMES

MODES = ('mano','native_wrist','B_wrist','B')
LABEL = ('MANO @ B center','SMPL-X / WiLoR wrist','SMPL-X / B wrist','Original B')
COLOR = ((205,165,70),(185,70,205),(65,215,95),(25,180,250))

def gallery(run, key):
    file = run/'assets'/(key+'.jpg')
    assert sha(file)==read(run/'evidence/results.json')['images'][key]['sha256']
    return cv2.imread(str(file))

def body_models(layer,key,sources,results):
    def pose(mode):
        file = (old.BASE_OUT if mode=='B' else old.OUT)/'poses'/f'{key}_{mode}.npz'
        assert sha(file)==results[mode]['pose_sha256']
        return dict(np.load(file))
    base = pose('B')
    seed = base['locked']
    poses = {'B':seed}
    for mode,suffix in [('native_wrist',''),('B_wrist','_B_wrist')]:
        q = seed.copy()
        for i,source in enumerate(sources):
            archived = pose(source+suffix)
            x = archived['locked']
            for k in base.keys()-{'locked'}:
                assert np.array_equal(base[k],archived[k])
            fingers = slice(23+15*i,38+15*i)
            q[fingers] = x[fingers]
            q[19+i] = x[19+i]
        poses[mode] = q
    assert np.array_equal(poses['native_wrist'][23:],poses['B_wrist'][23:])
    prior = ROOT/'outputs/stereo_hand_selection_20261127/poses'/(key+'.npz')
    assert sha(prior)==read(SELECT/'evidence/metrics.json')[key]['pose_sha256']
    assert np.array_equal(np.load(prior)['locked'],poses['B_wrist'])
    models = {m:old.mesh(layer,q,base) for m,q in poses.items()}
    drift = max(np.linalg.norm(j[:22]-models['B'][1][:22],axis=1).max() for v,j in models.values())*1000
    assert drift < .001
    return models, float(drift), rot(seed[21])@rot(base['global0'])

def main():
    torch.set_num_threads(1)
    stereo.RUN,stereo.OUT = RUN,OUT
    model,Camera = old.ref.prev.ref.common.setup_model()
    layer = model.layer['neutral'].eval()
    faces = np.asarray(model.face,dtype=np.int32)
    labels = groups(layer,faces)
    cals = old.ref.cameras()
    assert [sha(p) for p in cals]==read(old.ref.PRIOR/'evidence/calibration.json')['converted']
    cams = list(map(read,cals))
    yy,xx = np.indices((720,1280))
    xy = np.stack((xx,yy),axis=-1).reshape(-1,2)
    rays = []
    for p in cals:
        ray = np.zeros((720,1280,6),np.float32)
        ray[:,:,3:] = Camera(p).camera2world_ray(xy).reshape(720,1280,3)
        rays.append(o3d.core.Tensor(ray))
    receipt = read(NATIVE/'evidence/results.json')['cases']
    selected = read(SELECT/'evidence/selection.json')
    prepared = old.ref.OLD/'fit/0/prepared.json'
    assert sha(prepared)==read(old.ref.PRIOR/'evidence/inputs.json')['shards']['0']
    results,images = {},{}
    pools = {m:[] for m in MODES}
    for row in read(prepared)['items'][2:4]:
        key = row['pair']
        sources = [selected[key]['hands'][side]['source'] for side in SIDES]
        rec = read(old.RUN/'evidence'/(key+'.json'))['results']
        models,drift,root = body_models(layer,key,sources,rec)
        transforms = [np.array(row['inputs']['views'][e]['camera']['left_to_view']) for e in SIDES]
        verts,nfaces,parts = [],[],[]
        hands = {}
        for i,(side,source) in enumerate(zip(SIDES,sources)):
            src = ('cam3','cam4').index(source)
            tag = f'{key}_{SIDES[src]}_{side[0].upper()}'
            mano = np.load(OUT/(tag+'.npz'))
            ids = [JOINT_NAMES.index(f'{side}_{n}') for n in ['wrist']+[d+s for d in DIGITS for s in ('1','2','3','')]]
            joints = {m:j[ids] for m,(v,j) in models.items()}
            wrist = joints['B'][0]
            R = transforms[src][:3,:3]
            joints['mano'] = mano['j']@R+wrist
            f = mano['faces']
            nfaces.append((f[:,::-1] if i==0 else f)+sum(map(len,verts)))
            parts.extend([side[0].upper()+'_hand']*len(f))
            verts.append(mano['v']@R+wrist)
            h = rec['B']['hands'][side]
            obs = np.array(h['observation'])
            ok = np.array(h['inside'])[:,TIPS]
            stats = {}
            for mode,j in joints.items():
                uv = np.stack([old.projection(j@t[:3,:3].T+t[:3,3],cam) for cam,t in zip(cams,transforms)])
                if mode!='mano':
                    archive_mode = {'B':'B','native_wrist':source,'B_wrist':source+'_B_wrist'}[mode]
                    archive_delta = float(np.linalg.norm(uv-rec[archive_mode]['hands'][side]['uv'],axis=-1).max())
                    if mode=='native_wrist':
                        original = dict(np.load(old.OUT/'poses'/f'{key}_{source}.npz'))
                        _,arch_joints = old.mesh(layer,original['locked'],original)
                        for n,(t,cam) in enumerate(zip(transforms,cams)):
                            single = old.projection(arch_joints[ids]@t[:3,:3].T+t[:3,3],cam)
                            assert np.max(abs(single-rec[source]['hands'][side]['uv'][n]))<.05
                    else:
                        assert archive_delta<.05
                err = np.linalg.norm(uv-obs,axis=2)[:,TIPS][ok]
                pools[mode].extend(err.tolist())
                stats[mode] = dict(tips_px=err.tolist(),rmse_px=old.rms(err),
                    pinky_pip_deg=float(old.angle(j[18]-j[17],j[19]-j[18])),
                    local_bone_mean_deg=float(old.angle(old.bones(j)@old.palm_basis(j),
                        old.bones(joints['mano'])@old.palm_basis(joints['mano'])).mean()),
                    normal_to_mano_deg=float(old.angle(old.palm_basis(j)[:,2],old.palm_basis(joints['mano'])[:,2])))
                if mode!='mano':
                    stats[mode]['archive_max_px'] = archive_delta
            hands[side] = dict(source=source,source_sha256=receipt[tag]['input_pack_sha256'],count=int(ok.sum()),modes=stats)
        drawable = {'mano':(np.vstack(verts),np.vstack(nfaces),np.array(parts))}
        obj_hashes = {}
        for mode,(v,j) in models.items():
            v,file = stereo.export(key,mode,v,faces)
            drawable[mode] = (v,faces,labels)
            obj_hashes[mode] = sha(file)
        native_gallery = gallery(NATIVE,key)
        local_gallery = gallery(old.RUN,key+'_native_transfer')
        closeups,source_views,local = [],[],[]
        for eye,(view,t) in enumerate(zip(SIDES,transforms)):
            file = old.ref.OLD/'images'/f'{key}_{view}.png'
            photo = cv2.imread(str(file))
            projected = {}
            for mode,(v,f,lbl) in drawable.items():
                part,_,_ = stereo.raycast(v@t[:3,:3].T+t[:3,3],f,lbl,rays[eye])
                projected[mode] = part
            for i,side in enumerate(SIDES):
                ref = receipt[f'{key}_{view}_{side[0].upper()}']
                assert sha(file)==ref['photo_sha256']
                x,y,w,h = ref['crop_xywh']
                crop = np.s_[y:y+h,x:x+w]
                def tile(im,caption):
                    return title(cv2.resize(im[crop],(256,256)),caption)
                prefix = f'{eye+3} {side[0].upper()} | '
                panels = [tile(photo,prefix+'Photo')]
                for mode,label,color in zip(MODES,LABEL,COLOR):
                    im = photo.copy()
                    stereo.edge(im,projected[mode]==i+1,color)
                    panels.append(tile(im,prefix+label))
                closeups.append(np.hstack(panels))
                if sources[i]==f'cam{eye+3}':
                    index = 2*i+eye
                    native_tile = native_gallery[index*285+29:(index+1)*285,256:512]
                    source_views.append(np.hstack([panels[0],title(native_tile,prefix+'Old native')]+panels[1:]))
                    local.append(local_gallery[index*282:(index+1)*282])
        for name,rows in [('stereo',closeups),('source',source_views),('shape',local)]:
            stereo.save(images,key+'_'+name,np.vstack(rows))
        bodies = [title(full_body(drawable[m][0],faces,root),label) for m,label in zip(MODES[1:],LABEL[1:])]
        stereo.save(images,key+'_body',np.hstack(bodies))
        results[key] = dict(hands=hands,max_body_joint_drift_mm=drift,body_obj_sha256=obj_hashes)
    results['pooled'] = {m:dict(count=len(values),rmse_px=old.rms(values)) for m,values in pools.items()}
    write(RUN/'evidence/results.json',dict(images=images,metrics=results,exit_status=0))

if __name__=='__main__':
    main()
