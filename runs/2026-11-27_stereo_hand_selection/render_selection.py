"""Render both eyes and local hand views from the round-tripped, selected full-body OBJ."""
import hashlib
import importlib.util
import zipfile
import cv2
import numpy as np
import open3d as o3d
import torch

from choose import RUN, OUT, CASES, SIDES, TIPS, DIGITS, JOINT_NAMES, old, read, write, sha
from gallery import free_render, rot
from common_run import groups

spec = importlib.util.spec_from_file_location('transfer_visuals',old.RUN/'visuals.py')
prior_views = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior_views)
stereo, title, mark = prior_views.stereo, prior_views.title, prior_views.mark

def local_views(models,faces,labels,hands):
    rows = []
    for side in SIDES:
        subset = faces[labels==side[0].upper()+'_hand']
        ids = np.unique(subset)
        names = ['wrist']+[d+s for d in DIGITS for s in ('1','2','3','')]
        reference = models['B'][1][[JOINT_NAMES.index(side+'_'+n) for n in names]]
        across,up,normal = old.palm_basis(reference).T
        panels = []
        for degrees in (12,78):
            a = np.deg2rad(degrees)
            basis = np.column_stack((np.cos(a)*across+np.sin(a)*normal,up,-np.cos(a)*normal+np.sin(a)*across))
            all_xyz = np.concatenate([v[ids]@basis for v,_ in models.values()])
            center = (all_xyz.min(0)+all_xyz.max(0))/2
            scale = 226/np.ptp(all_xyz[:,:2],axis=0).max()
            winding = subset[:,::-1] if np.linalg.det(basis)<0 else subset
            for method,(v,_) in models.items():
                im = free_render(v,winding,basis,center,scale)[112:368,47:303].copy()
                im[np.all(im==im[0,0],axis=2)] = (36,36,33)
                label = 'B' if method=='B' else hands[side]['source']
                panels.append(title(im,f'{side} | {label} | {degrees} deg'))
        rows.append(np.hstack(panels))
    return np.vstack(rows)

def full_body(v,faces,root):
    views = []
    for basis,label in ((root,'Front'),(root@rot([0,np.pi/2,0]),'Side')):
        xyz = v@basis
        center = (xyz.max(0)+xyz.min(0))*.5
        scale = min(315/np.ptp(xyz[:,0]),440/np.ptp(xyz[:,1]))
        views.append(title(free_render(v,faces,basis,center,scale),label))
    return np.hstack(views)

def main():
    torch.set_num_threads(1)
    stereo.RUN,stereo.OUT = RUN,OUT
    model,Camera = old.ref.prev.ref.common.setup_model()
    layer = model.layer['neutral'].eval()
    faces = np.asarray(model.face,dtype=np.int32)
    labels = groups(layer,faces)
    yy,xx = np.indices((720,1280))
    xy = np.stack((xx,yy),axis=-1).reshape(-1,2)
    rays = []
    for cal in old.ref.cameras():
        ray = np.zeros((720,1280,6),np.float32)
        ray[:,:,3:] = Camera(cal).camera2world_ray(xy).reshape(720,1280,3)
        rays.append(o3d.core.Tensor(ray))
    metrics = read(RUN/'evidence/metrics.json')
    frames = read(old.ref.PRIOR/'evidence/frames.json')
    rows = {row['pair']:row for row in read(old.ref.OLD/'fit/0/prepared.json')['items']}
    images,hashes = {},{}
    for key in CASES:
        info = metrics[key]
        models = {}
        for mode,file,digest in [('B',old.BASE_OUT/'poses'/(key+'_B.npz'),info['seed_sha256']),
                                 ('selected',OUT/'poses'/(key+'.npz'),info['pose_sha256'])]:
            assert sha(file)==digest
            pack = dict(np.load(file))
            x = pack['locked']
            v,j = old.mesh(layer,x,pack)
            if mode=='selected':
                v,path = stereo.export(key,mode,v,faces)
                hashes[path.name] = sha(path)
            models[mode] = v,j
        hands = info['hands']
        v = models['selected'][0]
        stereo.save(images,key+'_body',full_body(v,faces,rot(x[21])@rot(pack['global0'])))
        stereo.save(images,key+'_hand_OBJ',local_views(models,faces,labels,hands))
        closeups = []
        for eye,(view,ray) in enumerate(zip(SIDES,rays)):
            photo_path = old.ref.OLD/'images'/(key+'_'+view+'.png')
            assert sha(photo_path)==frames[int(key[-1])-1]['views'][view]['frame']
            photo = cv2.imread(str(photo_path))
            t = np.asarray(rows[key]['inputs']['views'][view]['camera']['left_to_view'])
            masks = {}
            for mode,(verts,_) in models.items():
                part,valid,shade = stereo.raycast(verts@t[:3,:3].T+t[:3,3],faces,labels,ray)
                masks[mode] = part
                if mode=='selected':
                    overlay = photo.copy()
                    overlay[valid] = (.45*overlay[valid]+.55*shade[valid,None]).astype('uint8')
            proj_name = key+'_cam'+str(3+eye)
            stereo.save(images,proj_name,overlay)
            with_points = overlay.copy()
            for side in SIDES:
                mark(with_points,hands[side],eye,range(21))
            stereo.save(images,proj_name+'_points',with_points)
            for i,side in enumerate(SIDES):
                h = hands[side]
                ok = np.asarray(h['inside'][eye])
                obs = np.asarray(h['observation'][eye])
                sy,sx = np.where((masks['B']==i+1)|(masks['selected']==i+1))
                fit = np.asarray(h['uv'][eye])
                fit = fit[((fit>=0)&(fit<[1280,720])).all(1)]
                extent = max(sx.max()-sx.min(), sy.max()-sy.min(), 60)
                low = np.array([sx.min(),sy.min()]) - .4*extent
                high = np.array([sx.max(),sy.max()]) + .4*extent
                near = ((obs>=low)&(obs<=high)).all(1)
                pts = np.vstack((obs[ok&near],fit,[sx.min(),sy.min()],[sx.max(),sy.max()]))
                span = int(min(720,max(145,np.ptp(pts,axis=0).max()+30)))
                l,top = np.clip(((pts.min(0)+pts.max(0)-span)*.5).astype(int),0,[1280-span,720-span])
                sl = np.s_[top:top+span,l:l+span]
                baseline = photo.copy()
                stereo.edge(baseline,masks['B']==i+1,(35,195,250))
                selected = photo.copy()
                stereo.edge(selected,masks['selected']==i+1,(60,220,95))
                fingertips = selected.copy()
                mark(fingertips,h,eye,[0]+list(TIPS))
                prefix = f'{eye+3} | {side} | '
                tiles = [(photo,'Photo'),(baseline,'B'),(selected,'Selected '+h['source']),(fingertips,'Wrist + tips')]
                closeups.append(np.hstack([title(cv2.resize(im[sl],(290,290)),prefix+text) for im,text in tiles]))
        stereo.save(images,key+'_hands',np.vstack(closeups))
    archive = OUT/'export'/'stereo_selected_hands_OBJ.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name in hashes:
            z.write(OUT/'export'/name,name)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and {n:hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist()}==hashes
    write(RUN/'evidence/attachment_audit.json',dict(files_sha256={archive.name:sha(archive)},members=hashes))
    write(RUN/'evidence/results.json',dict(images=images,exit_status=0))

if __name__ == '__main__':
    main()
