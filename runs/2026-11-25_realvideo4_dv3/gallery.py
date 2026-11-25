"""Render a reloaded cam3 OBJ and both calibrated overlays for each selected frame."""
from runpy import run_path
import cv2
import numpy as np
import torch
from solve import ROOT,RUN,OUT,SIDES,cameras,v3,read,write,sha
from render_result import geometry,rot,free_render,load_obj,annotated,text,DARK

ASSETS=RUN/'assets'

def save(name,im):
    path=ASSETS/(name+'.jpg')
    assert cv2.imwrite(str(path),im,[cv2.IMWRITE_JPEG_QUALITY,93])
    return dict(size=list(im.shape[1::-1]),sha256=sha(path))

def make_obj(key,vertices,faces,root):
    obj=OUT/(key+'_cam3_m.obj')
    with obj.open('w') as f:
        f.write('# '+key+' Golden D v3, cam3 CV, meters\n')
        np.savetxt(f,vertices,fmt='v %.9f %.9f %.9f')
        np.savetxt(f,faces+1,fmt='f %d %d %d')
    v,topology=load_obj(obj)
    assert np.array_equal(topology,faces) and np.max(abs(v-vertices))<1e-8
    bases=[root,root@rot([0.,np.pi/2,0.])]
    coords=[v@basis for basis in bases]
    scale=min(min(315/np.ptp(q[:,0]),440/np.ptp(q[:,1])) for q in coords)
    panels=[]
    for title,basis,q in zip(['Front','Side'],bases,coords):
        im=free_render(v,faces,basis,(q.max(0)+q.min(0))*.5,scale)
        im=cv2.copyMakeBorder(im,36,0,0,0,cv2.BORDER_CONSTANT,value=DARK)
        text(im,key+'   '+title,(14,25),.7)
        panels.append(im)
    return v,np.hstack(panels),dict(name=obj.name,sha256=sha(obj),faces=len(faces),vertices=len(v))

def main():
    torch.set_num_threads(1)
    folder=OUT/'fit/0'
    rows=read(folder/'prepared.json')['items']
    pack=dict(np.load(folder/'selected.npz'))
    selection=read(folder/'selected.json')
    diagnostic=read(folder/'diagnostic_20000.json')
    model,Camera=v3.ref.common.setup_model()
    layer=model.layer['neutral'].eval()
    faces=np.asarray(model.face,np.int32)
    head_label=(layer.lbs_weights.argmax(1).numpy()==15).astype('f')
    raster=run_path(str(ROOT/'runs/2026-11-04_mesh-projection-review/code/rasterizer.py'))['raster']
    models=[Camera(p) for p in cameras()]
    y,x=np.indices((720,1280))
    pix=np.stack([x.ravel(),y.ravel()],1)
    rays=[c.camera2world_ray(pix).reshape(720,1280,3).astype('f') for c in models]
    results={};images={};gallery=[]
    for n,(row,sel) in enumerate(zip(rows,selection)):
        key=row['pair']
        assert key==sel['pair']==diagnostic['ids'][n][0]
        if 'failed' in sel:
            results[key]=sel
            continue
        i=sel['offset']+2
        params={k:pack[f][i] for k,f in [('body_pose','body_pose'),('betas','betas'),('global_orient','global0'),('transl','translation')]}
        vertices,joints,head,chest=geometry(layer,params,pack['delta'][i],False,None)
        assert np.allclose(head,chest) and not pack['betas'][i].any()
        v,views,obj=make_obj(key,vertices,faces,rot(pack['delta'][i])@rot(pack['global0'][i]))
        images[key+'_obj']=save(key+'_obj',views)
        pts=row['points'];ids=[p['native_index'] for p in pts]
        details={};errors=[];thumbs=[]
        for eye,(side,cam,ray) in enumerate(zip(SIDES,models,rays)):
            im=cv2.imread(str(OUT/'images'/(key+'_'+side+'.png')))
            T=np.array(row['inputs']['views'][side]['camera']['left_to_view'])
            vv=(v@T[:3,:3].T+T[:3,3]).astype('f')
            uv=cam.world2camera(vv.astype(float)).astype('f')
            depth=np.full(im.shape[:2],np.inf,'f')
            colors=np.zeros_like(im)
            drawn,rejected=raster(vv,faces,head_label,uv,ray,depth,colors,np.zeros_like(depth),180.)
            mask=np.isfinite(depth)
            assert mask.any()
            mesh=im.copy()
            mesh[mask]=(.45*im[mask]+.55*colors[mask]).astype('uint8')
            fitted=cam.world2camera(joints[ids].astype(float)@T[:3,:3].T+T[:3,3])
            det=np.array([p['xy'][eye] for p in pts])
            valid=np.array([p['valid'][eye] for p in pts])
            dist=np.linalg.norm(fitted-det,axis=1)
            errors.extend(np.square(dist[valid]))
            tag='cam'+str(eye+3)
            with_points=annotated(mesh,fitted,det,pts,valid,key+' '+tag)
            for label,img in [('mesh',mesh),('points',annotated(im,fitted,det,pts,valid,key+' '+tag)),('both',with_points)]:
                name='_'.join([key,tag,label])
                images[name]=save(name,img)
            thumbs.append(cv2.resize(with_points,(640,411),interpolation=cv2.INTER_AREA))
            details[tag]=dict(rmse_px=float(np.sqrt(np.mean(dist[valid]**2))),visible_pixels=int(mask.sum()),drawn=int(drawn),rejected=int(rejected),
                points={p['name']:dict(valid=bool(ok),det=a.tolist(),fit=b.tolist(),gap_px=float(d)) for p,ok,a,b,d in zip(pts,valid,det,fitted,dist)})
        stereo=np.hstack(thumbs)
        images[key+'_stereo']=save(key+'_stereo',stereo)
        objthumb=cv2.resize(views,(558,411),interpolation=cv2.INTER_AREA)
        gallery.append(np.hstack([stereo,objthumb]))
        results[key]=dict(views=details,combined_rmse_px=float(np.sqrt(np.mean(errors))),valid_observations=len(errors),obj=obj,
            selected=sel,feasible=diagnostic['selected_feasible'][n],converged=diagnostic['convergence_pass'][n],
            max_joint_motion_mm=diagnostic['max_joint_motion_mm'][n],selected_step=diagnostic['selected_step'][n])
    images['overview']=save('overview',np.vstack(gallery))
    write(RUN/'evidence/results.json',dict(cases=results,images=images,fit=read(folder/'fit_complete.json'),
        selected_hash=sha(folder/'selected.npz'),exit_status=0))
if __name__=='__main__':
    main()
