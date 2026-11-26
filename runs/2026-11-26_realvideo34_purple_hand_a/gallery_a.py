"""Publish stereo projections and external views from the same A-refined SMPL-X mesh."""
from zipfile import ZipFile, ZIP_DEFLATED
import hashlib
import cv2
import numpy as np
import open3d as o3d
import torch
from fit_hand_a import RUN, ROOT, OUT, PURPLE, METHOD, paired, sha, read, write
from gallery import geometry, rot, load_obj, free_render
from calibrate import rays, projection
from smplx.joint_names import JOINT_NAMES

CYAN=(255,225,0)
ORANGE=(0,135,255)
DIGITS=['thumb','index','middle','ring','pinky']

def title(img,msg):
    bar=np.full((32,img.shape[1],3),35,np.uint8)
    cv2.putText(bar,msg,(9,22),0,.59,(250,250,250),1,cv2.LINE_AA)
    return np.vstack((bar,img))

def save(name,img,hashes):
    path=RUN/'assets'/(name+'.jpg')
    assert cv2.imwrite(str(path),img,[cv2.IMWRITE_JPEG_QUALITY,93])
    assert cv2.imread(str(path)).shape==img.shape
    hashes[path.name]=sha(path)

def draw(img,data):
    canvas=img.copy()
    for item in data:
        a,b=[tuple(np.rint(item[n]).astype(int)) for n in ('det','fit')]
        cv2.line(canvas,a,b,(128,128,128),1,cv2.LINE_AA)
        cv2.circle(canvas,b,4,ORANGE,2,cv2.LINE_AA)
        cv2.drawMarker(canvas,a,CYAN,cv2.MARKER_TILTED_CROSS,9,2,cv2.LINE_AA)
    return canvas

def entry(name,a,b):
    return dict(name=name,det=np.asarray(a).tolist(),fit=np.asarray(b).tolist(),gap_px=float(np.linalg.norm(a-b)))

def rmse(rows):
    return float(np.sqrt(np.mean([a['gap_px']**2 for a in rows])))

def main():
    torch.set_num_threads(1)
    ref=paired.ref
    model,_=ref.prev.ref.common.setup_model()
    layer=model.layer['neutral'].eval()
    faces=np.asarray(model.face,dtype=np.int32)
    cats=layer.lbs_weights.argmax(1).numpy()
    ishead=np.isin(cats,[15,22,23,24])[faces].all(1)
    hand_categories=[np.isin(cats,[w]+list(range(i,i+15)))[faces].all(1) for w,i in [(20,25),(21,40)]]
    config=read(ref.OLD/'fit/0/prepared.json')['items']
    yy,xx=np.indices((720,1280))
    coords=np.column_stack([xx.ravel(),yy.ravel()])
    cams=list(map(read,ref.cameras()))
    outrows={}
    hashes={}
    overview=[]
    for key in ['clip003','clip004']:
        row=next(r for r in config if r['pair']==key)
        stored=read(RUN/'evidence'/(key+'.json'))['results']['control']
        pose=OUT/'poses'/(key+'_control.npz')
        assert sha(pose)==stored['pose_sha256']
        with np.load(pose) as pack:
            x=pack['locked']
            params=dict(global_orient=pack['global0'],body_pose=x[:21],transl=x[22],betas=pack['betas'],
                        left_hand_pose=x[23:38],right_hand_pose=x[38:])
            vertices,joints,_,_=geometry(layer,params,x[21],False,None)
            basis=rot(x[21])@rot(pack['global0'])
        path=OUT/(key+'_purple_A_cam3_m.obj')
        with path.open('w') as f:
            f.write('# Golden D v4 purple body, group A hands, cam3 meters\n')
            np.savetxt(f,vertices,fmt='v %.9f %.9f %.9f')
            np.savetxt(f,faces+1,fmt='f %d %d %d')
        v,topology=load_obj(path)
        assert np.max(abs(vertices-v))<1e-8 and np.array_equal(topology,faces)
        prev=read(PURPLE/'evidence/render.json')['rows'][key]['Purple']
        original=ROOT/'outputs/realvideo4_pitch_obj_20261126'/prev['obj']
        assert sha(original)==prev['sha256']
        purple,purple_f=load_obj(original)
        assert np.array_equal(purple_f,faces)
        head=np.unique(faces[ishead])
        head_mm=float(1000*np.linalg.norm(purple[head]-v[head],axis=1).max())
        head_source=ROOT/'outputs/realvideo4_head_mount_20261126'/(key+'.npz')
        assert sha(head_source)==prev['source_mesh_sha256']
        with np.load(head_source) as h:
            head_joint_mm=float(np.linalg.norm(h['head']+prev['translation_m']-joints[15])*1000)
        assert head_joint_mm<.001
        original_a=read(METHOD/'evidence'/(key+'.json'))['results']['control']['hands']
        previous={m:float(np.sqrt(np.mean([v[m]**2 for v in original_a.values()]))) for m in ['tips_px','wrist_px']}
        scene=o3d.t.geometry.RaycastingScene(nthreads=1)
        scene.add_triangles(o3d.core.Tensor(v.astype('float32')),o3d.core.Tensor(faces.astype('uint32')))
        bases=[basis,basis@rot([0,np.pi/2,0])]
        ext=[v@b for b in bases]
        scale=min(min(315/np.ptp(q[:,0]),440/np.ptp(q[:,1])) for q in ext)
        obj=np.hstack([title(free_render(v,faces,b,(q.max(0)+q.min(0))*.5,scale),key+'    '+label)
                       for b,q,label in zip(bases,ext,['Front','Side'])])
        save(key+'_obj',obj,hashes)
        views={}
        handzooms=[]
        combined=[]
        for n,(eye,cam) in enumerate(zip(paired.SIDES,cams)):
            tag='cam'+str(n+3)
            imfile=ref.OLD/'images'/(key+'_'+eye+'.png')
            assert sha(imfile)==read(paired.palm.V4/'evidence/segmentation.json')['eyes'][key+'_'+eye]['image_sha256']
            im=cv2.imread(str(imfile))
            T=np.array(row['inputs']['views'][eye]['camera']['left_to_view'])
            R,t=T[:3,:3],T[:3,3]
            origin=-R.T@t
            dirs=rays(coords,cam)@R
            dirs/=np.linalg.norm(dirs,axis=1,keepdims=True)
            beams=np.column_stack((np.broadcast_to(origin,dirs.shape),dirs))
            hits=scene.cast_rays(o3d.core.Tensor(beams.astype('float32')),nthreads=1)
            ids=hits['primitive_ids'].numpy()
            visible=ids<len(faces)
            norm=hits['primitive_normals'].numpy()[visible]
            diffuse=abs((norm*dirs[visible]).sum(1)).clip(0,1)
            rgb=np.tile(np.array([235,198,143]),(len(norm),1))
            rgb[ishead[ids[visible]]]=[213,146,228]
            for hand_label in hand_categories:
                rgb[hand_label[ids[visible]]]=[101,215,249]
            mesh=im.copy()
            flat=mesh.reshape(-1,3)
            flat[visible]=(.45*flat[visible]+.55*(.53+.47*diffuse[:,None])*rgb).astype('uint8')
            uv=projection(joints@R.T+t,cam)
            body=[entry(p['name'],np.array(p['xy'][n]),uv[p['native_index']]) for p in row['points'] if p['valid'][n]]
            hands={}
            for side in paired.SIDES:
                rec=stored['hands'][side]
                names=[side+'_'+d for d in ['wrist']+[d+'1' for d in DIGITS[1:]]+DIGITS]
                pred=uv[[JOINT_NAMES.index(name) for name in names]]
                assert np.max(abs(pred-np.array(rec['uv'][n])))<.01
                hands[side]=[entry(name,np.array(a),b) for name,a,b in zip(names,rec['target'][n],pred)]
            allpoints=body+hands['left']+hands['right']
            points=draw(im,allpoints)
            both=draw(mesh,allpoints)
            label=key+'    '+tag
            for mode,canvas in [('mesh',mesh),('points',points),('combined',both)]:
                save(key+'_'+tag+'_'+mode,title(canvas,label+('    Cyan X: detector    Orange O: fit' if mode!='mesh' else '    Projected mesh')),hashes)
            combined.append(title(cv2.resize(both,(560,315)),label))
            tiles=[]
            for side in paired.SIDES:
                data=hands[side]
                xy=np.array([a[c] for a in data for c in ['det','fit']])
                span=int(min(720,max(130,np.ptp(xy,axis=0).max()+45)))
                a,b=np.clip(np.rint((xy.min(0)+xy.max(0)-span)*.5).astype(int),[0,0],[1280-span,720-span])
                crop=draw(mesh,data)[b:b+span,a:a+span]
                tiles.append(title(cv2.resize(crop,(320,320)),key+'   '+tag+'   '+side))
            handzooms.append(np.hstack(tiles))
            views[tag]=dict(source_image_sha256=sha(imfile),visible_pixels=int(visible.sum()),body=body,hands=hands)
        save(key+'_hands',np.vstack(handzooms),hashes)
        overview.append(np.hstack(combined+[cv2.resize(obj,(474,347))]))
        outrows[key]=dict(obj=path.name,sha256=sha(path),pose_sha256=sha(pose),views=views,head_max_difference_mm=head_mm,purple_head_joint_difference_mm=head_joint_mm,previous_A_rmse_px=previous,
            body_rmse_px=rmse([d for eye in views.values() for d in eye['body']]),
            hand_wrist_rmse_px=rmse([hand[0] for eye in views.values() for hand in eye['hands'].values()]),
            heldout_tips_rmse_px=rmse([d for eye in views.values() for hand in eye['hands'].values() for d in hand[5:]]))
    save('overview',np.vstack(overview),hashes)
    archive=OUT/'clip003_004_purple_hand_A_OBJ.zip'
    with ZipFile(archive,'w',ZIP_DEFLATED) as z:
        for rec in outrows.values():
            z.write(OUT/rec['obj'],arcname=rec['obj'])
    with ZipFile(archive) as z:
        assert z.testzip() is None and len(z.namelist())==2
        for rec in outrows.values():
            assert hashlib.sha256(z.read(rec['obj'])).hexdigest()==rec['sha256']
    write(RUN/'evidence/gallery.json',dict(rows=outrows,image_sha256=hashes,
        bundle_sha256=sha(archive),exit_status=0))
if __name__=='__main__':main()
