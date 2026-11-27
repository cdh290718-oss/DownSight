from base import *
import numpy as np, cv2, torch, open3d as o3d
sys.path.insert(0,str(RUN.parent/'2026-11-27_wrist_only_default'))
import wrist_only as w
from render_selection import full_body, title, free_render,groups
from render_result import load_obj
from calibrate import projection,rays

def mark(img,a,b):
    aa,bb=[tuple(np.rint(p).astype(int)) for p in (a,b)]
    cv2.line(img,aa,bb,(127,127,127),1,cv2.LINE_AA)
    cv2.drawMarker(img,aa,(250,225,0),cv2.MARKER_TILTED_CROSS,9,2,cv2.LINE_AA)
    cv2.circle(img,bb,4,(0,138,255),1,cv2.LINE_AA)

def main(index):
    torch.set_num_threads(1);cv2.setNumThreads(1)
    index=int(index);row=read(OUT/'fit/0/prepared.json')['items'][index]
    key=row['pair'];frame=read(RUN/'evidence/frames.json')[index]
    assert frame['pair']==key
    rec=read(OUT/(key+'_hands.json'))
    path=OUT/'export'/rec['obj'];assert sha(path)==rec['obj_sha256']
    v,faces=load_obj(path)
    pack=dict(np.load(OUT/'poses'/(key+'_v5.npz')))
    model,_=w.ref.prev.ref.common.setup_model()
    layer=model.layer['neutral'].eval()
    vv,j=w.old.mesh(layer,pack['locked'],pack)
    assert np.max(abs(v-vv))<1e-8 and np.array_equal(faces,np.array(model.face))
    scene=o3d.t.geometry.RaycastingScene(nthreads=1)
    scene.add_triangles(o3d.core.Tensor(v.astype('float32')),o3d.core.Tensor(faces.astype('uint32')))
    hashes={}
    def save(label,img):
        p=RUN/'assets'/(key+'_'+label+'.jpg')
        assert cv2.imwrite(str(p),img,[cv2.IMWRITE_JPEG_QUALITY,88])
        hashes[p.name]=sha(p)
    root=w.old.mat(pack['locked'][21])@w.old.mat(pack['global0'])
    body=full_body(v,faces,root);save('obj',body)
    parts=groups(layer,faces)
    hand_views=[]
    for i,s in enumerate(SIDES):
        sub=faces[parts==s[0].upper()+'_hand'];ids=np.unique(sub)
        across,up,normal=w.old.palm_basis(j[w.prior.IDS[i]]).T
        tiles=[]
        for deg in [12,78]:
            a=np.deg2rad(deg)
            axes=np.column_stack((np.cos(a)*across+np.sin(a)*normal,up,-np.cos(a)*normal+np.sin(a)*across))
            q=v[ids]@axes
            tile=free_render(v,sub[:,::-1] if np.linalg.det(axes)<0 else sub,axes,(q.min(0)+q.max(0))*.5,225/np.ptp(q[:,:2],axis=0).max())[112:368,47:303]
            tiles.append(title(tile,f'{s}  {deg} deg'))
        hand_views.append(np.hstack(tiles))
    save('handobj',np.vstack(hand_views))
    views=dict(mesh=[],points=[],both=[]);hands=[];errors=dict(body=[],tips=[])
    yy,xx=np.indices((720,1280));pixels=np.column_stack([xx.ravel(),yy.ravel()])
    for eye,(side,cp) in enumerate(zip(SIDES,cameras())):
        cam=read(cp);T=np.array(row['inputs']['views'][side]['camera']['left_to_view']);R,t=T[:3,:3],T[:3,3]
        vec=rays(pixels,cam)@R;vec/=np.linalg.norm(vec,axis=1,keepdims=True)
        ray=np.column_stack([np.broadcast_to(-R.T@t,vec.shape),vec]).astype('float32')
        hits=scene.cast_rays(o3d.core.Tensor(ray),nthreads=1)
        fid=hits['primitive_ids'].numpy();mask=fid<len(faces);normal=hits['primitive_normals'].numpy()[mask]
        photo=OUT/'images'/(key+'_'+side+'.png');assert sha(photo)==frame['views'][side]['frame']
        im=cv2.imread(str(photo));mesh=im.copy()
        color=np.tile([229,191,144],(mask.sum(),1))
        color[np.isin(parts[fid[mask]],['L_hand','R_hand'])]=[105,206,252]
        shade=.55+.45*np.abs((vec[mask]*normal).sum(1)).clip(0,1)
        mesh.reshape(-1,3)[mask]=.5*im.reshape(-1,3)[mask]+.5*color*shade[:,None]
        points=im.copy();both=mesh.copy()
        uv=projection(j@R.T+t,cam)
        for p in row['points']:
            if not p['valid'][eye]:continue
            a=np.array(p['xy'][eye]);b=uv[p['native_index']]
            errors['body'].append(float(np.linalg.norm(a-b)))
            for img in (points,both):mark(img,a,b)
        for i,s in enumerate(SIDES):
            h=rec['hands'][s];a=np.array(h['obs'][eye]);b=uv[w.prior.IDS[i]]
            assert np.allclose(b,h['uv'][eye],atol=.01)
            good=np.array(h['valid'][eye]);tips=np.array(w.prior.TIPS);sel=tips[good[tips]]
            errors['tips'].extend(np.linalg.norm(a[sel]-b[sel],axis=1).tolist())
            for k in np.flatnonzero(good):
                for img in (points,both):mark(img,a[k],b[k])
        tag=key+'   cam'+str(eye+3)
        for label,pic in [('mesh',mesh),('points',points),('both',both)]:
            views[label].append(title(pic,tag+('    Cyan X: automatic     Orange O: model' if label!='mesh' else '    Dv5 mesh')))
        for i,s in enumerate(SIDES):
            h=rec['hands'][s]; good=np.array(h['valid'][eye])
            projected=uv[w.prior.IDS[i]]; validproj=((projected>=0)&(projected<[1280,720])).all(1)
            # Frame the visible model hand so a stray automatic fingertip on the
            # torso cannot zoom the hand panel out to the entire photograph.
            region=np.zeros(mask.shape,bool)
            region[mask]=parts[fid[mask]]==s[0].upper()+'_hand'
            hy,hx=np.where(region.reshape(yy.shape))
            q=projected[validproj]
            if len(hx):
                q=np.vstack([q,[hx.min(),hy.min()],[hx.max(),hy.max()]])
            if not len(q):
                q=np.array(h['obs'][eye])[good]
            if len(q)==0:
                hands.append(title(np.full((235,470,3),35,np.uint8),tag+'    '+s+' outside'));continue
            span=int(np.clip(max(np.ptp(q,axis=0))*1.15+35,135,720))
            l,top=np.clip(np.rint((q.max(0)+q.min(0)-span)*.5).astype(int),0,[1280-span,720-span])
            def crop(pic):return cv2.resize(pic[top:top+span,l:l+span],(235,235))
            hands.append(title(np.hstack([crop(im),crop(both)]),tag+'   '+s+'   |   Photo / mesh + points'))
    for label,panels in views.items():save(label,np.hstack(panels))
    save('hands',np.vstack([np.hstack(hands[:2]),np.hstack(hands[2:])]))
    thumb=np.hstack([cv2.resize(p,(428,252)) for p in views['both']]+[cv2.resize(body,(343,252))])
    save('overview',title(thumb,f"{key}    {100*frame['fraction']:.0f}%    t = {frame['relative_s']:.2f} s"))
    write(RUN/'evidence'/(key+'_render.json'),dict(hashes=hashes,source=rec['source'],obj=rec['obj'],obj_sha256=rec['obj_sha256'],drift_mm=rec['drift_mm'],
        hands={s:{k:rec['hands'][s][k] for k in ['source','candidates','fit_mask']} for s in SIDES},errors=errors,exit_status=0))
