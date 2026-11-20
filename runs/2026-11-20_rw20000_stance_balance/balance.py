"""Single-pair quasi-static balance refinement of archived camera-frame meshes."""
from pathlib import Path
import sys
import numpy as np
import torch
from scipy.spatial import ConvexHull

RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[1]
sys.path.insert(0,str(ROOT/'runs/2026-11-20_rw20000_frozen_mesh12'))
import gallery as g
from geometry import rotation, project
from render import load_obj

OUT=ROOT/'outputs/rw20000_stance_balance_20261120'
MAIN=['pair001','pair036','pair068','pair078']
AMB=['pair022','pair028','pair034','pair088']
PRIOR=np.array([0.,0.,-1.])
BAND=.005
STEPS=220


def center(j):
    """Dempster fractional masses; SMPL-X landmarks approximate segment ends."""
    hips=(j[...,1,:]+j[...,2,:])/2
    sh=(j[...,16,:]+j[...,17,:])/2
    c=.497*(hips+.495*(sh-hips))+.081*j[...,15,:]
    for a,b,w,k in [(16,18,.028,.436),(17,19,.028,.436),(18,20,.016,.43),(19,21,.016,.43),
                    (20,28,.006,.506),(21,43,.006,.506),(1,4,.1,.433),(2,5,.1,.433),
                    (4,7,.0465,.433),(5,8,.0465,.433),(7,10,.0145,.5),(8,11,.0145,.5)]:
        c=c+w*(j[...,a,:]+k*(j[...,b,:]-j[...,a,:]))
    return c


def axes(n):
    x=np.array([1.,0.,0.]);x=x-n*np.dot(n,x);x/=np.linalg.norm(x)
    return np.stack([x,np.cross(n,x)],axis=1)


def polygon(points):
    """Convex support boundary with outward unit half-space normals."""
    h=ConvexHull(points)
    return h.equations[:,:2],h.equations[:,2],h.vertices


def support(v,sole,side,n,d,tol=BAND):
    heights=v[sole]@n+d
    ix=sole[heights<=tol]
    return ix,sorted(set(side[ix].tolist()))


def plane_candidate(v,sole,side,double):
    # Supporting facets use geometry alone. Mass and evaluation labels have no role here.
    hull=ConvexHull(v[sole])
    normals=np.vstack([PRIOR,-hull.equations[:,:3]])
    normals=normals[(normals@PRIOR)>=np.cos(np.deg2rad(45))]
    for need_both in ([True,False] if double else [False]):
        for k in np.argsort(-(normals@PRIOR),kind='stable'):
            n=normals[k];d=-np.min(v@n)
            ids,feet=support(v,sole,side,n,d)
            if len(ids)<3 or (need_both and len(feet)!=2):continue
            xy=v[ids]@axes(n)
            if np.linalg.matrix_rank(xy-xy.mean(0))<2:continue
            return n,d,ids,feet
    raise ValueError('No plausible supporting plane')


def stats(v,j,sole,side,n,d):
    ids,feet=support(v,sole,side,n,d)
    xy=axes(n);pts=v[ids]@xy
    if len(pts)<3 or np.linalg.matrix_rank(pts-pts.mean(0))<2:
        return dict(margin_mm=None,supported=feet,contacts=len(ids),penetration_mm=max(0.,-1000*(v@n+d).min()))
    A,b,h=polygon(pts)
    com=center(j);margin=-float(np.max(A@(com@xy)+b))
    # Vary the applied load direction at fixed geometry and contacts.
    height=com@n+d
    pert=[-float(np.max(A@(com@xy+height*np.tan(np.deg2rad(5))*np.array(dr))+b))
          for dr in [(1,0),(-1,0),(0,1),(0,-1)]]
    return dict(margin_mm=margin*1000,supported=feet,contacts=len(ids),
                area_cm2=ConvexHull(pts).volume*1e4,margin_at_5deg_mm=(np.array(pert)*1000).tolist(),
                penetration_mm=max(0.,-1000*(v@n+d).min()))


def tensor(x):
    return torch.as_tensor(x,dtype=torch.float32)


def posed(layer,body,glob,delta,tr):
    b=len(body)
    zero={k:torch.zeros(b,n) for k,n in [('left_hand_pose',45),('right_hand_pose',45),
          ('expression',10),('jaw_pose',3),('leye_pose',3),('reye_pose',3)]}
    res=layer(body_pose=body.reshape(b,-1),global_orient=glob,betas=torch.zeros(b,10),**zero)
    rot=rotation(delta).transpose(1,2);p=res.joints[:,:1]
    return tuple((v-p)@rot+p+tr[:,None] for v in (res.vertices,res.joints))


def main(strong=False, keep_contacts=False):
    torch.set_num_threads(1)
    checkpoint=g.SOURCE/'budget_20000.npz'
    assert g.sha(checkpoint)==g.read(g.REF/'execution.json')['params_sha256']
    inputs=g.read(g.SOURCE/'prepared.json')['items']; done=g.read(g.SOURCE/'fit_complete.json')
    archives={r['pair']:r for r in g.read(g.RUN/'evidence/renders_rw20000.json')['rows']}
    model,cam=g.common.setup_model();layer=model.layer['neutral'].cpu().eval();layer.requires_grad_(False)
    cams=[cam(g.DATA/('camera_'+s+'.json')) for s in ['left','right']]
    skin=layer.lbs_weights.detach().numpy()
    rest=layer.v_template.detach().numpy()
    side=np.full(len(rest),-1,int);sole=[]
    for k,idx in enumerate([[7,10],[8,11]]):
        ix=np.flatnonzero(skin[:,idx].sum(1)>.8)
        ix=ix[rest[ix,1]<rest[ix,1].min()+.034]
        sole.extend(ix);side[ix]=k
    sole=np.array(sole)
    names=MAIN if strong else MAIN+AMB
    suffix='_contact' if keep_contacts else ('_strong' if strong else '')
    ids=[done['ids'].index([p,'rw20000','2d']) for p in names]
    records=[];best_params=[]
    with np.load(checkpoint) as src:
        for p,i in zip(names,ids):
            item=inputs[i];assert item['pair']==p and item['condition']=='rw20000'
            body0=tensor(src['body_pose'][i]).reshape(1,21,3);glob=tensor(src['global0'][i:i+1])
            dt=tensor(src['delta'][i:i+1]);tr=tensor(src['translation'][i:i+1])
            v0,j0=posed(layer,body0,glob,dt,tr);old=v0[0].numpy().astype(float)
            a=archives[p];path=g.OUT/'objects'/a['subject']/a['obj']
            assert g.sha(path)==a['source_obj_sha256']
            vv,_=load_obj(path);assert np.max(np.abs(vv-old))<.000003
            n,d,contact,feet=plane_candidate(old,sole,side,p in MAIN)
            basis=axes(n);A,b,boundary=polygon(old[contact]@basis)
            anchor=contact[boundary]
            before=stats(old,j0[0].numpy(),sole,side,n,d)
            T=[tensor(item['inputs']['views'][s]['camera']['left_to_view']) for s in ['left','right']]
            ix=[q['native_index'] for q in item['points']]
            def projections(j):
                return torch.stack([project(j[:,ix]@t[:3,:3].T+t[:3,3],c) for t,c in zip(T,cams)],1)
            uv0=projections(j0)
            valid=tensor(np.array([q['valid'] for q in item['points']]).T)[None]
            db=torch.zeros_like(body0,requires_grad=True); dr=torch.zeros_like(dt,requires_grad=True)
            shift=torch.zeros_like(tr,requires_grad=True)
            mask=torch.ones_like(db);mask[:,[11,14]]=0
            opt=torch.optim.Adam([db,dr,shift],lr=.001)
            best=float('inf'); saved=None
            count=(650 if strong else STEPS) if before['margin_mm']<0 else 0
            for step in range(count+1):
                body=body0+db*mask
                v,j=posed(layer,body,glob,dt+dr,tr+shift)
                violation=(tensor(A)@(center(j)[0]@tensor(basis))+tensor(b)).max()
                stability=(violation+(.012 if strong else 0.)).relu()
                uv=projections(j)
                pixel=((uv-uv0).square().sum(-1)*valid).sum()/valid.sum()
                contact_loss=((v[:,anchor]-v0[:,anchor]).square().sum(-1)).mean()
                pen=(-(v[:,sole]@tensor(n)+d)).relu().max()
                head=(j[:,15]-j0[:,15]).square().sum()
                loss=(stability/(.001 if strong else .01)).square()+pixel/4+contact_loss/.002**2+(pen/.002)**2+head/.008**2+db.square().mean()/.05**2+dr.square().mean()/.08**2+shift.square().mean()/.02**2
                if keep_contacts:
                    # Keep the boundary vertices eligible when the post-fit contact patch is rebuilt.
                    excess=(v[:,anchor]@tensor(n)+d-.0045).relu().max()
                    loss=loss+(excess/.0002).square()
                value=float(loss.detach())
                if value<best:
                    best=value
                    saved=[q.detach().numpy()[0].copy() for q in (body,glob,dt+dr,tr+shift,v,j)]
                if step<count:
                    opt.zero_grad();loss.backward();opt.step()
            body,root,delta,translation,v,j=saved
            after=stats(v.astype(float),j,sole,side,n,d)
            rec=dict(pair=p,group='standing' if p in MAIN else 'uncertain',source_obj_sha256=a['source_obj_sha256'],
                     candidate_up=n.tolist(),plane_offset_m=float(d),tilt_from_camera_prior_deg=float(np.rad2deg(np.arccos(np.clip(n@PRIOR,-1,1)))),
                     before=before,after=after,steps=count,loss=best,
                     frozen_polygon_margin_after_mm=-float((A@(center(j)@basis)+b).max())*1000,
                     max_anchor_displacement_mm=float(np.linalg.norm(v[anchor]-old[anchor],axis=1).max()*1000))
            records.append(rec);best_params.append((body,root,delta,translation))
            print(p,feet,round(before['margin_mm'],1),after['margin_mm'],flush=True)
    np.savez_compressed(OUT/('adjusted'+suffix+'.npz'),**dict(zip(['body','root','delta','translation'],map(np.stack,zip(*best_params)))))
    g.write(RUN/'evidence'/('balance'+suffix+'.json'),dict(rows=records,checkpoint_sha256=g.sha(checkpoint),
            adjusted_sha256=g.sha(OUT/('adjusted'+suffix+'.npz')),contact_band_m=BAND,command='<BODY_PYTHON> -Bu balance.py'+(' --contact' if keep_contacts else (' --strong' if strong else '')),exit_status=0))


if __name__=='__main__':main('--strong' in sys.argv or '--contact' in sys.argv,'--contact' in sys.argv)
