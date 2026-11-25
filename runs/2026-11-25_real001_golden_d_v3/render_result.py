"""Export the selected mesh and render native fisheye and body-relative views."""
from runpy import run_path
import sys
import cv2
import numpy as np
import torch
from run_pair import RUN,ROOT,OUT,KEY,sides,cameras,v3,read,write,sha
sys.path.insert(0,str(ROOT/'runs/2026-11-19_rw_landmarks_head_views'))
from export import geometry,rot
from render import free_render,load_obj

ASSETS=RUN/'assets'
CROSS=(245,224,44)
RING=(24,157,255)
DARK=(28,27,23)


def save(name,im):
    path=ASSETS/(name+'.jpg')
    assert cv2.imwrite(str(path),im,[cv2.IMWRITE_JPEG_QUALITY,94])
    return {'sha256':sha(path),'size':list(im.shape[1::-1])}

def text(im,word,at,scale=.55,color=(245,245,245)):
    cv2.putText(im,word,at,cv2.FONT_HERSHEY_SIMPLEX,scale,color,1,cv2.LINE_AA)

def annotated(im,uv,det,pts,valid,eye):
    h,w=im.shape[:2]
    board=cv2.copyMakeBorder(im,48,54,0,0,cv2.BORDER_CONSTANT,value=DARK)
    xy=lambda p:tuple(np.rint(p+[0,48]).astype(int))
    text(board,f'{eye}   RW20000 detection: cyan X     Golden D v3 joint: orange O     Joined pairs enter loss',(16,32),.72)
    missing=[]
    for q,a,b,ok in zip(pts,det,uv,valid):
        if ok:
            cv2.line(board,xy(a),xy(b),(26,28,35),3,cv2.LINE_AA)
            cv2.line(board,xy(a),xy(b),(242,242,242),1,cv2.LINE_AA)
        else:
            missing.append(q['name'])
    for a,b,ok in zip(det,uv,valid):
        if ((b>=0)&(b<[w,h])).all():
            cv2.circle(board,xy(b),8,DARK,4,cv2.LINE_AA)
            cv2.circle(board,xy(b),8,RING,2,cv2.LINE_AA)
        if ((a>=0)&(a<[w,h])).all():
            c=CROSS if ok else (170,170,170)
            for k,sz in [(DARK,5),(c,2)]:
                cv2.drawMarker(board,xy(a),k,cv2.MARKER_TILTED_CROSS,13,sz,cv2.LINE_AA)
    text(board,'Excluded detection: '+', '.join(missing)+'     |     full native frame', (16,48+h+34),.66)
    return board

def main():
    ASSETS.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    folder=OUT/'fit/0'
    row=read(folder/'prepared.json')['items'][0]
    meta=read(folder/'selected.json')[0]
    assert meta['pair']==KEY and meta['losses'][-1]<=meta['losses'][0]+2e-6
    model,Camera=v3.ref.common.setup_model()
    layer=model.layer['neutral'].cpu().eval()
    layer.requires_grad_(False)
    pack=dict(np.load(folder/'selected.npz'));i=2
    assert not pack['betas'][i].any() and not pack['body_pose'][i,[11,14]].any()
    params={k:pack[n][i] for k,n in [('body_pose','body_pose'),('betas','betas'),('global_orient','global0'),('transl','translation')]}
    vertices,joints,head,spine=geometry(layer,params,pack['delta'][i],False,None)
    assert np.allclose(head,spine)
    assert np.max(abs(joints[:22]-v3.ref.full_joints(pack,layer,i)[:22]))<1e-5
    faces=np.asarray(model.face,np.int32)
    obj=OUT/'real001_t029p0_v3_rw20000_cam3_cv.obj'
    with obj.open('w') as f:
        f.write('# real001, Golden D v3, cam3 CV, meters\n')
        np.savetxt(f,vertices,fmt='v %.9f %.9f %.9f')
        np.savetxt(f,faces+1,fmt='f %d %d %d')
    v,faces2=load_obj(obj)
    assert np.array_equal(faces,faces2)
    delta=float(np.max(abs(v-vertices)))
    assert delta<1e-8
    root=rot(pack['delta'][i])@rot(pack['global0'][i])
    bases=[root,root@rot([0,np.pi/2,0])]
    bounds=[v@b for b in bases]
    scale=min(min(315/np.ptp(q[:,0]),440/np.ptp(q[:,1])) for q in bounds)
    panels=[]
    for b,q,tag in zip(bases,bounds,['Front','Side']):
        panel=free_render(v,faces,b,(q.min(0)+q.max(0))/2,scale)
        panel=cv2.copyMakeBorder(panel,36,0,0,0,cv2.BORDER_CONSTANT,value=DARK)
        text(panel,tag,(12,25),.68)
        panels.append(panel)
    images={'obj_front_side':save('obj_front_side',np.hstack(panels))}
    raster=run_path(str(ROOT/'runs/2026-11-04_mesh-projection-review/code/rasterizer.py'))['raster']
    head_label=(layer.lbs_weights.argmax(1).numpy()==15).astype(np.float32)
    pts=row['points'];ids=[p['native_index'] for p in pts]
    results={};squared=[]; previews=[]
    for n,(side,cal) in enumerate(zip(sides,cameras())):
        tag=['cam3','cam4'][n]
        cam=Camera(cal)
        im=cv2.imread(str(OUT/'images'/(KEY+'_'+side+'.png')))
        h,w=im.shape[:2]
        assert list(cam.img_size)==[w,h]
        y,x=np.indices((h,w))
        rays=cam.camera2world_ray(np.stack([x.ravel(),y.ravel()],1)).reshape(h,w,3).astype('f')
        T=np.array(row['inputs']['views'][side]['camera']['left_to_view'])
        vv=(v@T[:3,:3].T+T[:3,3]).astype('f')
        uv=cam.world2camera(vv.astype(float)).astype('f')
        depth=np.full((h,w),np.inf,np.float32)
        colors=np.zeros_like(im)
        drawn,rejected=raster(vv,faces,head_label,uv,rays,depth,colors,np.zeros_like(depth),180.)
        mask=np.isfinite(depth)
        assert mask.any()
        mesh=im.copy()
        mesh[mask]=(.45*im[mask]+.55*colors[mask]).astype('uint8')
        fitted=cam.world2camera(joints[ids].astype(float)@T[:3,:3].T+T[:3,3])
        det=np.array([p['xy'][n] for p in pts])
        valid=np.array([p['valid'][n] for p in pts])
        dist=np.linalg.norm(det-fitted,axis=1)
        squared.extend(np.square(dist[valid]))
        images[tag+'_mesh']=save(tag+'_mesh',mesh)
        images[tag+'_points']=save(tag+'_points',annotated(im,fitted,det,pts,valid,tag))
        images[tag+'_mesh_points']=save(tag+'_mesh_points',annotated(mesh,fitted,det,pts,valid,tag))
        previews.append(cv2.resize(annotated(mesh,fitted,det,pts,valid,tag),None,fx=.5,fy=.5,interpolation=cv2.INTER_AREA))
        results[tag]=dict(valid=int(valid.sum()),included_distance_px={p['name']:float(d) for p,d,ok in zip(pts,dist,valid) if ok},
                          rmse_px=float(np.sqrt(np.mean(dist[valid]**2))),visible_pixels=int(mask.sum()),drawn=int(drawn),rejected=int(rejected))
    images['stereo_overview']=save('stereo_overview',np.hstack(previews))
    write(RUN/'evidence/results.json',dict(views=results,images=images,combined_rmse_px=float(np.sqrt(np.mean(squared))),
        current_selection=meta,selection_sha256=sha(folder/'selected.npz'),obj=dict(name=obj.name,sha256=sha(obj),vertex_count=len(v),
        face_count=len(faces),reload_max_abs_m=delta,coordinates='cam3 CV metres'),diagnostic=read(folder/'diagnostic_20000.json'),
        fit_complete=read(folder/'fit_complete.json'),exit_status=0))
if __name__=='__main__':
    main()
