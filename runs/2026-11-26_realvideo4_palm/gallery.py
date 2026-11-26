"""Render a reloaded cam3 OBJ and both calibrated overlays for each selected frame."""
from runpy import run_path
import cv2
import numpy as np
import torch
import sys
from pathlib import Path
RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[1]
PRIOR=RUN.parent/'2026-11-25_realvideo4_dv3'
sys.path.insert(0,str(PRIOR))
from solve import OUT as OLD,SIDES,cameras,v3,read,write,sha
from render_result import geometry,rot,free_render,load_obj,annotated as annotate_old,text,DARK
OUT=ROOT/'outputs/realvideo4_palm_20261126'
ASSETS=RUN/'assets'

def annotated(im,uv,det,pts,valid,tag):
    pic=annotate_old(im,uv,det,pts,valid,tag)
    pic[:48]=DARK
    text(pic,tag+'   RW20000: cyan X     Palm fit: orange O     Joined pairs enter fit',(16,32),.72)
    return pic

def save(name,im):
    path=ASSETS/(name+'.jpg')
    assert cv2.imwrite(str(path),im,[cv2.IMWRITE_JPEG_QUALITY,93])
    return dict(size=list(im.shape[1::-1]),sha256=sha(path))

def make_obj(key,vertices,faces,root):
    obj=OUT/'export'/(key+'_palm_cam3_m.obj')
    with obj.open('w') as f:
        f.write('# '+key+' D v4 palm, cam3 CV, meters\n')
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

def hands_zoom(key,entry):
    from pathlib import Path
    old_assets=RUN.parent/'2026-11-25_realvideo4_dv4/assets'
    panels=[]
    detections=read(OLD/'predictions.json')
    for side in SIDES:
        for eye,cam in enumerate(['cam3','cam4']):
            old,new=[entry['metrics'][stage][side] for stage in ['before','after']]
            target=np.array(old['det'][eye])
            sources=[cv2.imread(str(OLD/'images'/(key+'_'+SIDES[eye]+'.png'))),
                     cv2.imread(str(old_assets/(key+'_'+cam+'_mesh.jpg'))),cv2.imread(str(ASSETS/(key+'_'+cam+'_mesh.jpg')))]
            tip_ids=np.array([21,25,29,33,37])+(21 if side=='left' else 0)
            tips=np.asarray(detections[key+'_'+SIDES[eye]+'.png']['xy'])[tip_ids]
            bounds=np.vstack([target,tips,old['uv'][eye],new['uv'][eye]])
            low,high=bounds.min(0),bounds.max(0)
            span=int(max(145,max(high-low)+115))
            left,top=np.clip(((low+high-span)/2).astype(int),0,[1280-span,720-span])
            row=[]
            for img,model,tag in zip(sources,[None,old,new],['Photo / detections','Golden D v4','Palm refinement']):
                canvas=cv2.resize(img[top:top+span,left:left+span],(280,280))
                def mark(points,color,cross):
                    xy=np.rint((np.array(points)-[left,top])*280/span).astype(int)
                    for idx,p in enumerate(xy):
                        if cross:cv2.drawMarker(canvas,p,color,cv2.MARKER_TILTED_CROSS,9,2,cv2.LINE_AA)
                        else:cv2.circle(canvas,p,5,color,2,cv2.LINE_AA)
                        if model is None and idx in (1,4):
                            cv2.putText(canvas,['I','P'][idx==4],tuple(p+[5,-5]),cv2.FONT_HERSHEY_SIMPLEX,.46,(20,20,20),3,cv2.LINE_AA)
                            cv2.putText(canvas,['I','P'][idx==4],tuple(p+[5,-5]),cv2.FONT_HERSHEY_SIMPLEX,.46,color,1,cv2.LINE_AA)
                mark(target,(245,224,44),True)
                if model is not None:mark(model['uv'][eye],(24,157,255),False)
                tile=cv2.copyMakeBorder(canvas,30,0,0,0,cv2.BORDER_CONSTANT,value=DARK)
                text(tile,f'{key} {side[0].upper()} {cam} | '+tag,(6,20),.43)
                row.append(tile)
            panels.append(np.hstack(row))
    return np.vstack(panels)

def main():
    torch.set_num_threads(1)
    folder=OLD/'fit/0'
    rows=read(folder/'prepared.json')['items']
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
    for row in rows:
        key=row['pair']
        sel=read(RUN/'evidence'/(key+'.json'))
        path=OUT/'poses'/(key+'.npz')
        assert sha(path)==sel['pose_hash']
        with np.load(path) as pack:
            x=pack['locked']
            params=dict(body_pose=x[:21],betas=pack['betas'],global_orient=pack['global0'],transl=x[22],
                        left_hand_pose=x[23:38],right_hand_pose=x[38:])
            vertices,joints,head,chest=geometry(layer,params,x[21],False,None)
            assert np.allclose(head,chest) and not x[23:].any() and not x[[6,7,9,10]].any()
            v,views,obj=make_obj(key,vertices,faces,rot(x[21])@rot(pack['global0']))
        images[key+'_obj']=save(key+'_obj',views)
        pts=row['points'];ids=[p['native_index'] for p in pts]
        details={};errors=[];thumbs=[]
        for eye,(side,cam,ray) in enumerate(zip(SIDES,models,rays)):
            im=cv2.imread(str(OLD/'images'/(key+'_'+side+'.png')))
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
            selected=sel['best'])
        closeup=hands_zoom(key,sel)
        images[key+'_hands']=save(key+'_hands',closeup)
    images['overview']=save('overview',np.vstack(gallery))
    write(RUN/'evidence/results.json',dict(cases=results,images=images,exit_status=0))
if __name__=='__main__':
    main()
