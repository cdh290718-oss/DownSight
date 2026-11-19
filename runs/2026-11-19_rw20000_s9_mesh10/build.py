"""Export only the ten archived RW20000 fits and draw the released surfaces."""
import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
OUT = ROOT / 'outputs/rw20000_s9_mesh10_20261119'
SOURCE = ROOT / 'outputs/rw_three_free12_20261119'
REFERENCE = ROOT / 'runs/2026-11-19_rw_three_free12/evidence'
sys.path.insert(0, str(ROOT / 'runs/2026-11-19_rw_landmarks_head_views'))
from export import geometry, rot
from render import free_render, load_obj, save
from study import OLD, common, read, write, sha
from evaluate import align

ASSETS = RUN / 'assets'
ARCHIVE = 'RW20000_S9_body01-10_free12_OBJ.zip'
TAG = 'rw20000-s9-mesh10-free12-20261119'


def overlay(v, faces, head, cam, rays, transform, image, raster):
    vv = (v @ transform[:3,:3].T + transform[:3,3]).astype(np.float32)
    uv = cam.world2camera(vv.astype(float)).astype(np.float32)
    depth = np.full(image.shape[:2], np.inf, np.float32)
    colors = np.zeros_like(image)
    drawn, rejected = raster(vv, faces, head, uv, rays, depth, colors, np.zeros_like(depth), 180.)
    visible = np.isfinite(depth)
    assert visible.any()
    result = image.copy()
    result[visible] = .55 * image[visible] + .45 * colors[visible]
    return result, dict(drawn=drawn, rejected=rejected, covered_pixels=int(visible.sum()))


def main():
    previous = read(REFERENCE / 'results.json')
    records = {r['pair']:r for r in previous['rows'] if r['model']=='rw20000'}
    checkpoint = SOURCE / 'budget_20000.npz'
    assert sha(checkpoint)==previous['fit_npz_sha256']
    assert sha(SOURCE / 'prepared.json')==read(REFERENCE / 'inputs.json')['solver_input_sha256']
    items = read(SOURCE / 'prepared.json')['items']
    done = read(SOURCE / 'fit_complete.json')
    diag = read(SOURCE / 'diagnostic_20000.json')
    assert done['exit_status']==0 and diag['ids']==done['ids']
    spec = importlib.util.spec_from_file_location('rasterizer', ROOT / 'runs/2026-11-04_mesh-projection-review/code/rasterizer.py')
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    model, camera = common.setup_model()
    layer = model.layer['neutral'].cpu().eval()
    cams = [camera(p) for p in common.camera_files()]
    faces = np.asarray(model.face, np.int32)
    head = (layer.lbs_weights.argmax(1).numpy()==15).astype(np.float32)
    y,x = np.indices((720,1280))
    rays = [c.camera2world_ray(np.stack([x.ravel(),y.ravel()],1)).reshape(720,1280,3).astype(np.float32) for c in cams]
    # Use the fixed viewing axes of the published S9 gallery, with a per-frame crop.
    basis = rot(np.load(OLD/'body01/D_left/final.npz')['global_orient'])
    bases = [basis,basis @ rot([0,np.pi/2,0])]
    gtfile = common.ARCH / 'body_gt_official.npz'
    assert sha(gtfile)==previous['gt_sha256']
    rows, members = [], {}
    objects = OUT / 'objects'
    objects.mkdir(exist_ok=True)
    with np.load(checkpoint) as z, np.load(gtfile) as gt:
        for i,(pair,version,_) in enumerate(done['ids']):
            if version!='rw20000':
                continue
            item = items[i]
            assert (pair,version)==(item['pair'],item['condition']) and diag['selected_feasible'][i]
            params = {k:z[n][i] for k,n in [('global_orient','global0'),('body_pose','body_pose'),('betas','betas'),('transl','translation')]}
            v,j,h,c = geometry(layer, params, z['delta'][i], False, None)
            assert np.allclose(h,c) and np.isfinite(v).all()
            g = list(gt['ids']).index(pair+'_left')
            target = gt['joints_m'][g,common.OBS].astype(float)
            pred = j[common.NATIVE_IDS][common.OBS]
            errors = [float(np.linalg.norm(a-target,axis=1).mean()*1000) for a in (pred,align(pred,target))]
            assert np.allclose(errors,[records[pair]['mpjpe12_mm'],records[pair]['pa12_mm']],atol=.001,rtol=0)
            obj = objects / (pair+'__rw20000__left_cv_m.obj')
            with obj.open('w') as f:
                f.write('# SMPL-X; left camera CV; meters\n')
                np.savetxt(f,v,fmt='v %.9f %.9f %.9f')
                np.savetxt(f,faces+1,fmt='f %d %d %d')
            vertices, triangles = load_obj(obj)
            assert np.max(np.abs(vertices-v)) < 1e-8 and np.array_equal(triangles,faces)
            members[obj.name] = sha(obj)
            projected = [vertices @ b for b in bases]
            centers = [(q.max(0)+q.min(0))/2 for q in projected]
            scale = min(min(295/np.ptp(q[:,0]),425/np.ptp(q[:,1])) for q in projected)
            free = np.hstack([free_render(vertices,triangles,b,center,scale) for b,center in zip(bases,centers)])
            pics, counts = [free], []
            for k,side in enumerate(('left','right')):
                src = common.ARCH / 'body_input' / (pair+'_'+side+'.png')
                image = cv2.imread(str(src))
                assert image.shape==(720,1280,3)
                T = np.asarray(item['inputs']['views'][side]['camera']['left_to_view'])
                pic, audit = overlay(vertices,triangles,head,cams[k],rays[k],T,image,renderer.raster)
                pics.append(pic)
                counts.append(dict(side=side,image_sha256=sha(src),**audit))
            images = {}
            for name,pic in zip(('obj','left','right'),pics):
                path = ASSETS / pair / (name+'.jpg')
                save(path,pic)
                images[path.name]=sha(path)
            panel = np.hstack([cv2.resize(free,(350,240),interpolation=cv2.INTER_AREA)]+
                              [cv2.resize(pic[:,280:1000],(240,240),interpolation=cv2.INTER_AREA) for pic in pics[1:]])
            label = np.full((36,830,3),246,np.uint8)
            cv2.putText(label,pair+'   RW20000          front / side                   left eye                     right eye',
                        (10,24),cv2.FONT_HERSHEY_SIMPLEX,.51,(30,30,30),1,cv2.LINE_AA)
            preview = ASSETS / (pair+'__rw20000.jpg')
            save(preview,np.vstack((label,panel)))
            rows.append(dict(pair=pair,source_obj_sha256=members[obj.name],obj=obj.name,
                             vertices=len(vertices),faces=len(triangles),selected_step=int(z['selected_step'][i]),
                             pa12_mm=errors[1],nonpa12_mm=errors[0],images=images,views=counts,preview_sha256=sha(preview)))
            print(pair, 'export and renders checked', flush=True)
    assert sorted(records)==sorted(r['pair'] for r in rows)==['body%02d'%i for i in range(1,11)]
    write(RUN/'evidence/renders_rw20000.json',dict(checkpoint_sha256=sha(checkpoint),rows=rows))
    archive = OUT/ARCHIVE
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name in sorted(members):
            z.write(objects/name,name)
        z.writestr('manifest.json',json.dumps(members,indent=2))
        z.writestr('README.txt','RW20000 S9 body01-body10. 12-point fit, torso-following head, free feet. Left-camera CV (X right, Y down, Z forward), meters, unaligned.\n')
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and set(z.namelist())==set(members)|{'manifest.json','README.txt'}
        assert all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in members.items())
    write(RUN/'evidence/attachment_audit.json',dict(archive=ARCHIVE,tag=TAG,files_sha256={ARCHIVE:sha(archive)},
          bytes=archive.stat().st_size,objects=len(members),jpeg=len(list(ASSETS.rglob('*.jpg'))),exit_status=0,
          command='<BODY_PYTHON> -Bu build.py',base='8d196f217f4a54b03412ee8918bd0ecf1a6e7561'))


if __name__=='__main__':
    main()
