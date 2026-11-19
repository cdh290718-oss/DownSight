"""Reopen the actual released OBJ files, then render two free views and two calibrated image overlays."""
import argparse
import importlib.util
from pathlib import Path

import cv2
import numpy as np

from export import CONTROL,CAM_GROUP,OLD_GROUP,AUTO,AUTO_HEAD,LABEL,rot
from study import RUN,OUT,ROOT,OLD,read,write,sha,common

ASSETS=RUN/'assets'
GROUPS={'old':OLD_GROUP,'current':CONTROL,'camera':CAM_GROUP,'automatic':AUTO,'automatic_head':AUTO_HEAD}
BG=245


def load_obj(path):
    lines=path.read_text().splitlines()
    v=np.array([s.split()[1:4] for s in lines if s.startswith('v ')],float)
    f=np.array([s.split()[1:] for s in lines if s.startswith('f ')],np.int32)-1
    assert np.isfinite(v).all() and f.min()==0 and f.max()==len(v)-1
    return v, f


def free_render(v, faces, basis, center, scale):
    w,h=350,480
    q=v@basis-center
    xy=np.rint(q[:,:2]*[scale,-scale]+[w/2,h/2]).astype(np.int32)
    tri=q[faces]
    normals=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    normals/=np.maximum(np.linalg.norm(normals,axis=1)[:,None],1e-12)
    lum=(133+94*np.maximum(0,normals@[-.35,.55,.75])).astype(int)
    im=np.full((h,w,3),BG,np.uint8)
    for fi in np.argsort(tri[:,:,2].mean(1)):
        c=int(lum[fi])
        cv2.fillConvexPoly(im,xy[faces[fi]],(c,c,c))
    return im


def save(path,im):
    path.parent.mkdir(parents=True,exist_ok=True)
    assert cv2.imwrite(str(path),im,[cv2.IMWRITE_JPEG_QUALITY,91])


def main(family):
    spec=importlib.util.spec_from_file_location('ray_raster',ROOT/'runs/2026-11-04_mesh-projection-review/code/rasterizer.py')
    renderer=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    model,cam_cls=common.setup_model()
    cams=[cam_cls(p) for p in common.camera_files()]
    head=(model.layer['neutral'].lbs_weights.argmax(1).numpy()==15).astype(np.float32)
    y,x=np.indices((720,1280))
    rays=[cam.camera2world_ray(np.stack([x.ravel(),y.ravel()],1)).reshape(720,1280,3).astype(np.float32) for cam in cams]
    basis=rot(np.load(OLD/'body01/D_left/final.npz')['global_orient'])
    bases=[basis,basis@rot([0,np.pi/2,0])]
    rows=read(RUN/'evidence'/('results_'+family+'.json'))['rows']
    manifest=[]
    for i in range(1,5):
        pair='body%02d'%i
        orig=np.load(OLD/pair/'D_left/final.npz')['smplx_mesh_cam']
        centers=[]
        for b in bases:
            pts=orig@b
            centers.append((pts.max(0)+pts.min(0))/2)
        # Fixed original-fit centers and scale ensure the same framing even across families.
        scale=min(min(288/np.ptp((orig@b)[:,0]),395/np.ptp((orig@b)[:,1])) for b in bases)
        images=[cv2.imread(str(common.ARCH/'body_input'/(pair+'_'+s+'.png'))) for s in ['left','right']]
        ts=read(OLD/pair/'inputs.json')['views']
        overview=[]
        for mode in GROUPS[family]:
            row=next(r for r in rows if r['condition']==mode and r['pair']==pair)
            obj=OUT/'objects'/pair/row['obj']
            assert sha(obj)==row['obj_sha256']
            v,f=load_obj(obj)
            assert np.array_equal(f,model.face)
            free=np.hstack([free_render(v,f,b,c,scale) for b,c in zip(bases,centers)])
            views=[]
            counts=[]
            for j,side in enumerate(['left','right']):
                T=np.array(ts[side]['camera']['left_to_view'])
                vv=(v@T[:3,:3].T+T[:3,3]).astype(np.float32)
                uv=cams[j].world2camera(vv.astype(float)).astype(np.float32)
                depth=np.full((720,1280),np.inf,np.float32)
                colors=np.zeros_like(images[j])
                hp=np.zeros_like(depth)
                drawn,rejected=renderer.raster(vv,f,head,uv,rays[j],depth,colors,hp,180.)
                mask=np.isfinite(depth)
                assert mask.any()
                im=images[j].copy()
                im[mask]=(.55*im[mask]+.45*colors[mask]).astype(np.uint8)
                views.append(im)
                counts.append(dict(view=side,drawn=drawn,rejected=rejected,covered_pixels=int(mask.sum())))
            dest=ASSETS/pair
            paths=[dest/(mode+'__'+name+'.jpg') for name in ['obj','left','right']]
            for path,im in zip(paths,[free]+views):
                save(path,im)
            manifest.append(dict(pair=pair,mode=mode,source_obj_sha256=sha(obj),views=counts,
                        images={p.name:sha(p) for p in paths}))
            cells=[cv2.resize(free,(350,240),interpolation=cv2.INTER_AREA)]
            cells.extend(cv2.resize(im[:,280:1000],(240,240),interpolation=cv2.INTER_AREA) for im in views)
            strip=np.full((36,830,3),246,np.uint8)
            cv2.putText(strip,pair+'   /   '+LABEL[mode],(12,25),cv2.FONT_HERSHEY_SIMPLEX,.70,(28,28,28),1,cv2.LINE_AA)
            overview.extend([strip,np.hstack(cells)])
        save(ASSETS/(pair+'__'+family+'.jpg'),np.vstack(overview))
    write(RUN/'evidence'/('renders_'+family+'.json'),dict(rows=manifest,
          source='OBJ reread and verified against intended export; triangle depth-tested in each original camera; no scene depth.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('family',choices=list(GROUPS))
    main(parser.parse_args().family)
