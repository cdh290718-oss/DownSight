"""Export frozen RW20000 checkpoints and visualize the sampled stereo frames."""
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
sys.path.insert(0, str(ROOT/'runs/2026-11-19_rw20000_s9_mesh10'))
from build import common, geometry, rot, free_render, load_obj, save, read, write, sha, align
from runpy import run_path

SOURCE = ROOT/'outputs/rw_frozen200_free12_20261119'
REF = ROOT/'runs/2026-11-19_rw_frozen200_free12/evidence'
DATA = ROOT/'outputs/rw_frozen200_golden_d_20261118'
MANIFEST = ROOT/'outputs/rw_test_completion_20261117/test_manifest.json'
OUT = ROOT/'outputs/rw20000_frozen_mesh12_20261120'
ASSETS = RUN/'assets'
ARCHIVE = 'RW20000_frozen200_S1_S9_S11_12_OBJ.zip'
TAG = 'rw20000-frozen200-mesh12-20261120'


def sample(pairs):
    selected = []
    for subject in ('S1','S9','S11'):
        candidates = [p for p in pairs if p['subject']==subject]
        def order(p):
            key = 'rw20000-frozen-mesh12-v1/' + subject + '/' + p['pair']
            return hashlib.sha256(key.encode()).hexdigest()
        sequences = set()
        for p in sorted(candidates,key=order):
            if p['sequence'] in sequences:
                continue
            selected.append(p)
            sequences.add(p['sequence'])
            if len(sequences)==4:
                break
        assert len(sequences)==4
    return selected


def main():
    source = read(REF/'execution.json')
    inputs = read(REF/'inputs.json')
    checkpoint = SOURCE/'budget_20000.npz'
    for path,digest in [(checkpoint,source['params_sha256']),(MANIFEST,source['manifest_sha256']),
                         (SOURCE/'prepared.json',inputs['solver_input_sha256']),(DATA/'reference.json',source['reference_sha256'])]:
        assert sha(path)==digest
    provenance = read(ROOT/'runs/2026-11-18_rw_frozen200_golden_d/evidence/preparation.json')
    selected = sample(provenance['pairs'])
    images = {x['id']:x for x in read(MANIFEST)['records']}
    refs = read(DATA/'reference.json')
    metrics = {x['pair']:x for x in read(REF/'results.json')['rows'] if x['model']=='rw20000'}
    items = read(SOURCE/'prepared.json')['items']
    done = read(SOURCE/'fit_complete.json')
    assert done['exit_status']==0
    indices = {p:i for i,(p,m,_) in enumerate(done['ids']) if m=='rw20000'}
    model,camera = common.setup_model()
    layer = model.layer['neutral'].cpu().eval()
    faces = np.asarray(model.face,np.int32)
    head = (layer.lbs_weights.argmax(1).numpy()==15).astype(np.float32)
    calfiles = [DATA/('camera_'+s+'.json') for s in ('left','right')]
    assert all(sha(p)==inputs['cameras'][p.name] for p in calfiles)
    cams = [camera(p) for p in calfiles]
    adapters = [read(p)['input_adapter'] for p in calfiles]
    y,x = np.indices((872,872))
    pixels = np.stack([x.ravel(),y.ravel()],axis=1)
    rays = [c.camera2world_ray(pixels*a['scale']+a['offset']).reshape(872,872,3).astype(np.float32)
            for c,a in zip(cams,adapters)]
    raster = run_path(str(ROOT/'runs/2026-11-04_mesh-projection-review/code/rasterizer.py'))['raster']
    map_rows = read(ROOT/'runs/2026-11-16_sapiens-unrealego-audit/evidence/audit.json')['joint_mapping']
    rw_to_det = {m['rw_index']:m['sapiens308_id'] for m in map_rows}
    det_ids = [rw_to_det[ix] for ix in common.OBS]
    members, rows = {}, []
    with np.load(checkpoint) as z:
        for entry in selected:
            pair, subject, seq, frame = (entry[k] for k in ('pair','subject','sequence','frame'))
            i = indices[pair]
            row = items[i]
            record = metrics[pair]
            assert (row['pair'],row['condition'])==(pair,'rw20000')
            assert record['feasible'] and record['selected_step']==int(z['selected_step'][i])
            params = {k:z[n][i] for k,n in [('global_orient','global0'),('body_pose','body_pose'),('betas','betas'),('transl','translation')]}
            v,j,h,c = geometry(layer,params,z['delta'][i],False,None)
            assert np.allclose(h,c)
            pred = j[common.NATIVE_IDS][common.OBS]
            gt = np.asarray(refs[pair]['gt'])[common.OBS]
            values = [np.linalg.norm(a-gt,axis=1).mean()*1000 for a in (pred,align(pred,gt))]
            assert np.allclose(values,[record['mpjpe12_mm'],record['pa12_mm']],atol=.001,rtol=0)
            obj = OUT/'objects'/subject/f'{subject}_{pair}_{Path(seq).name}_f{frame}_rw20000.obj'
            obj.parent.mkdir(parents=True,exist_ok=True)
            with obj.open('w') as f:
                f.write('# left camera CV, meters\n')
                np.savetxt(f,v,fmt='v %.9f %.9f %.9f')
                np.savetxt(f,faces+1,fmt='f %d %d %d')
            v,tri = load_obj(obj)
            assert np.array_equal(faces,tri)
            digest = sha(obj)
            members[str(obj.relative_to(OUT/'objects'))]=digest
            basis = rot(z['delta'][i]) @ rot(z['global0'][i])
            bases = [basis,basis @ rot([0,np.pi/2,0])]
            coords = [v@b for b in bases]
            scale = min(min(300/np.ptp(q[:,0]),425/np.ptp(q[:,1])) for q in coords)
            pictures = [free_render(v,tri,b,(q.max(0)+q.min(0))/2,scale) for q,b in zip(coords,bases)]
            views = []
            for k,side in enumerate(('left','right')):
                m = images[entry['image_ids'][k]]
                assert (m['view'],m['subject'],m['sequence'],m['frame'])==(side,subject,seq,frame)
                path = MANIFEST.parent/m['image']
                assert sha(path)==m['image_sha256']==entry['image_sha256'][k]
                im = cv2.imread(str(path))
                assert im.shape==(872,872,3)
                T = np.asarray(row['inputs']['views'][side]['camera']['left_to_view'])
                a = adapters[k]
                def project(points):
                    return (cams[k].world2camera(points)-a['offset'])/a['scale']
                pxy = project(pred @ T[:3,:3].T + T[:3,3])
                valid = np.array([p['valid'][k] for p in row['points']],bool)
                gtxy = np.asarray(m['keypoints308_xy'])[det_ids]
                rmse = np.sqrt(np.square((pxy-gtxy)[valid]).sum()/valid.sum())
                prev = record['views'][k]
                assert prev['side']==side and prev['n']==valid.sum()
                assert abs(rmse-np.sqrt(prev['sum_sq_px2']['fit_vs_gt']/prev['n'])) < .001
                vv = (v@T[:3,:3].T + T[:3,3]).astype(np.float32)
                uv = project(vv.astype(float)).astype(np.float32)
                depth = np.full(im.shape[:2],np.inf,np.float32)
                color = np.zeros_like(im)
                drawn,rejected = raster(vv,tri,head,uv,rays[k],depth,color,np.zeros_like(depth),180/a['scale'])
                mask = np.isfinite(depth)
                assert mask.any()
                im[mask] = .55*im[mask]+.45*color[mask]
                pictures.append(im)
                views.append(dict(side=side,original_sha256=m['image_sha256'],drawn=drawn,rejected=rejected,
                                  covered_pixels=int(mask.sum()),fit_vs_gt_rmse_px=rmse))
            hashes = {}
            for name,pic in zip(('front','side','left','right'),pictures):
                path = ASSETS/pair/(name+'.jpg')
                save(path,pic)
                hashes[path.name]=sha(path)
            strip = np.full((36,830,3),245,np.uint8)
            for pos,title in zip((7,184,364,609),(subject+' '+pair+'  Front','Side','Left eye','Right eye')):
                cv2.putText(strip,title,(pos,25),cv2.FONT_HERSHEY_SIMPLEX,.54,(40,40,40),1,cv2.LINE_AA)
            cells = [cv2.resize(p,sz,interpolation=cv2.INTER_AREA) for p,sz in
                     zip(pictures,[(175,240),(175,240),(240,240),(240,240)])]
            preview = ASSETS/(pair+'__rw20000.jpg')
            save(preview,np.vstack((strip,np.hstack(cells))))
            rows.append(dict(subject=subject,pair=pair,sequence=seq,frame=frame,obj=obj.name,source_obj_sha256=digest,
                             vertices=len(v),faces=len(tri),selected_step=record['selected_step'],pa12_mm=values[1],
                             nonpa12_mm=values[0],images=hashes,preview_sha256=sha(preview),views=views))
            print(subject,pair,flush=True)
    write(RUN/'evidence/renders_rw20000.json',dict(checkpoint_sha256=sha(checkpoint),rows=rows))
    with zipfile.ZipFile(OUT/ARCHIVE,'w',zipfile.ZIP_DEFLATED) as z:
        for name in sorted(members):
            z.write(OUT/'objects'/name,name)
        z.writestr('manifest.json',json.dumps(members,indent=2))
        z.writestr('README.txt','RW20000, frozen 200. Four frames per subject. Left camera CV, meters (X right, Y down, Z forward). Free feet, torso-following head.\n')
    with zipfile.ZipFile(OUT/ARCHIVE) as z:
        assert len(members)==12 and set(z.namelist())==set(members)|{'manifest.json','README.txt'} and z.testzip() is None
        assert all(hashlib.sha256(z.read(n)).hexdigest()==s for n,s in members.items())
    write(RUN/'evidence/attachment_audit.json',dict(archive=ARCHIVE,tag=TAG,files_sha256={ARCHIVE:sha(OUT/ARCHIVE)},
          bytes=(OUT/ARCHIVE).stat().st_size,objects=len(rows),jpeg=len(list(ASSETS.rglob('*.jpg'))),exit_status=0,
          command='<BODY_PYTHON> -Bu gallery.py',base='97bd8ceee9c09083fb6db6ab2a22b1a551e47f45'))


if __name__=='__main__':
    main()
