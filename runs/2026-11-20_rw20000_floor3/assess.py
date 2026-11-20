"""Calibrated stereo reconstruction of bare carpet; no body-derived plane."""
from pathlib import Path
import sys
import cv2
import numpy as np

RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[1]
sys.path.insert(0,str(ROOT/'runs/2026-11-20_rw20000_frozen_mesh12'))
import gallery as g
from render import load_obj,free_render,save
sys.path.insert(0,str(ROOT/'runs/2026-11-19_rw_flat_feet'))
from plane import unit,angle,ransac,refine
read,write,sha=g.read,g.write,g.sha
# Half-resolution left-eye pixel coordinates, manually selected on the raw photographs.
REGIONS={
 'pair045':[[(117,237),(215,239),(242,244),(295,226),(332,216),(318,261),(288,293),(227,311),(170,306),(124,286)],[(306,166),(329,124),(359,149),(380,172),(373,203),(333,211),(312,216)]],
 'pair055':[[(112,219),(133,227),(148,269),(164,270),(177,252),(246,253),(280,251),(299,247),(304,261),(283,287),(250,319),(211,320),(175,314),(147,301),(127,278),(115,255)],[(317,228),(340,230),(362,221),(375,214),(369,241),(353,256),(322,262)]],
 'pair078':[[(115,259),(157,248),(199,260),(227,251),(268,242),(297,239),(320,238),(334,252),(325,300),(297,323),(233,343),(179,331),(143,316)],[(303,176),(326,185),(341,199),(353,206),(344,230),(310,226),(288,228)]]}
class RawCamera:
    def __init__(self,cam,adapter):
        self.cam,self.scale,self.offset=cam,adapter['scale'],np.array(adapter['offset'])
    def camera2world_ray(self,uv):
        return self.cam.camera2world_ray(uv*self.scale+self.offset)
    def world2camera(self,xyz):
        return (self.cam.world2camera(xyz)-self.offset)/self.scale


def sparse(images,mask,cams,R,t):
    sift=cv2.SIFT_create(contrastThreshold=.002)
    kl,dl=sift.detectAndCompute(images[0],mask)
    kr,dr=sift.detectAndCompute(images[1],None)
    matcher=cv2.BFMatcher()
    right=matcher.match(dr,dl)
    seen=set(); pts=[]; uv=[]
    for m,n in matcher.knnMatch(dl,dr,k=2):
        l=kl[m.queryIdx].pt
        cell=tuple(np.rint(l))
        if cell in seen or m.distance>=.86*n.distance or right[m.trainIdx].trainIdx!=m.queryIdx:continue
        r=kr[m.trainIdx].pt
        u,v=[unit(cam.camera2world_ray(np.array([point]))[0]) for cam,point in zip(cams,[l,r])]
        origin=-R.T@t
        rays=np.column_stack([u,-R.T@v]); ds=np.linalg.lstsq(rays,origin,rcond=None)[0]
        pos=(ds[0]*u+origin+ds[1]*(R.T@v))/2
        if min(ds)<.45 or max(ds)>4:continue
        err=[np.linalg.norm(cam.world2camera(x[None])[0]-p) for cam,x,p in zip(cams,[pos,R@pos+t],[l,r])]
        if max(err)>1.5:continue
        # Reject repetitive carpet features unless their local texture agrees in both eyes.
        patches=[cv2.getRectSubPix(im,(13,13),xy).astype(float) for im,xy in zip(images,[l,r])]
        a,b=[(q-q.mean()).ravel() for q in patches]
        if (a@b)/max(1e-8,np.linalg.norm(a)*np.linalg.norm(b)) < .65:continue
        pts.append(pos);uv.append(l);seen.add(cell)
    return np.array(pts),np.array(uv)


def estimate(images,cams,T,polys):
    gray=[cv2.cvtColor(a,cv2.COLOR_BGR2GRAY) for a in images]
    mask=np.zeros_like(gray[0]);cv2.fillPoly(mask,polys,255)
    R,t=T[:3,:3],T[:3,3]
    xyz,pixels=sparse(gray,mask,cams,R,t)
    print('points',len(xyz),flush=True)
    start,inside=ransac(xyz)
    yy,xx=np.nonzero(mask); uv=np.column_stack([xx,yy])[::5]
    half=(uv[:,0]//36+uv[:,1]//36)%2==0
    gray=[im.astype(float) for im in gray]
    p,a,b=[refine(start,q,gray,cams,R,t) for q in (uv,uv[half],uv[~half])]
    normal=-unit(p); dist=1/np.linalg.norm(p)
    check_angle=angle(a,b); check_dist=abs(1/np.linalg.norm(a)-1/np.linalg.norm(b))
    err=np.abs(xyz@normal+dist)
    print('inlier/angle/distance',inside.sum(),check_angle,dist,check_dist,flush=True)
    ok=bool(inside.sum()>=8 and .45<dist<3 and normal[2]<0 and check_angle<5 and check_dist<.05 and np.median(err)<.05)
    check=dict(sparse_matches=len(xyz),ransac_inliers=int(inside.sum()),dense_pixels=len(uv),
               checker_angle_deg=check_angle,checker_distance_diff_mm=check_dist*1000,
               sparse_median_residual_mm=float(np.median(err)*1000),sparse_vs_dense_deg=angle(start,p),accepted=ok)
    return normal,dist,check,pixels[inside]


def draw(pair,vertices,faces,joints,gt,up,d,forward,im,polys,inliers,accepted):
    horizontal=unit(forward-up*(up@forward))
    basis=np.column_stack([horizontal,up,np.cross(horizontal,up)])
    union=np.vstack([vertices,joints,gt])@basis
    lo,hi=union.min(0),union.max(0); lo[1]=min(lo[1],-d)-.055; hi[1]=max(hi[1],-d)+.05
    center=(hi+lo)/2
    scale=min(305/(hi[0]-lo[0]),435/(hi[1]-lo[1]))
    mesh=free_render(vertices,faces,basis,center,scale)
    def screen(points):
        return np.rint(((points@basis-center)[:,:2])*[scale,-scale]+[175,240]).astype(int)
    line=round(240+(d+center[1])*scale)
    for left in range(4,345,14):
        cv2.line(mesh,(left,line),(min(345,left+(14 if accepted else 6)),line),(50,145,63),2)
    if not accepted:
        cv2.putText(mesh,'Tentative floor',(8,28),0,.55,(25,70,155),1,cv2.LINE_AA)
    edges=[(0,1),(1,2),(2,3),(0,4),(4,5),(5,6),(1,7),(4,11),(7,11),(7,8),(8,9),(9,10),(11,12),(12,13),(13,14)]
    points=screen(gt)
    for a,b in edges:cv2.line(mesh,tuple(points[a]),tuple(points[b]),(25,95,227),2,cv2.LINE_AA)
    for p in points:cv2.circle(mesh,tuple(p),3,(22,68,192),-1,cv2.LINE_AA)
    patch=im.copy(); cv2.polylines(patch,polys,True,(0,245,255),3)
    for q in inliers:cv2.circle(patch,tuple(np.rint(q).astype(int)),4,(0,215,30),-1)
    patch=cv2.copyMakeBorder(cv2.resize(patch,(430,430)),25,25,0,0,cv2.BORDER_CONSTANT,value=(245,245,245))
    title=np.full((50,780,3),245,np.uint8)
    cv2.putText(title,pair+'   Orange: GT   Green: carpet',(8,31),0,.48,(35,35,35),1,cv2.LINE_AA)
    cv2.putText(title,'Stereo carpet samples (left eye)',(362,31),0,.50,(35,35,35),1,cv2.LINE_AA)
    path=RUN/'assets'/(pair+'.jpg');save(path,np.vstack([title,np.hstack([mesh,patch])]))
    return sha(path)


def main():
    rows={r['pair']:r for r in read(g.RUN/'evidence/renders_rw20000.json')['rows']}
    provenance=read(ROOT/'runs/2026-11-18_rw_frozen200_golden_d/evidence/preparation.json')
    pairs={r['pair']:r for r in provenance['pairs']}
    assert sha(g.MANIFEST)==provenance['manifest_sha256']
    images={r['id']:r for r in read(g.MANIFEST)['records']}
    items=read(g.SOURCE/'prepared.json')['items']
    truth=read(g.DATA/'reference.json')
    model,cam=g.common.setup_model()
    paths=[g.DATA/('camera_'+s+'.json') for s in ('left','right')]
    cams=[RawCamera(cam(path),read(path)['input_adapter']) for path in paths]
    results=[]
    with np.load(g.SOURCE/'budget_20000.npz') as z:
        for pair,regions in REGIONS.items():
            row=rows[pair];entry=pairs[pair]
            i=next(i for i,it in enumerate(items) if (it['pair'],it['condition'])==(pair,'rw20000'))
            orig=[]
            for k,key in enumerate(entry['image_ids']):
                path=g.MANIFEST.parent/images[key]['image'];assert sha(path)==entry['image_sha256'][k]
                orig.append(cv2.imread(str(path)))
            T=np.array(items[i]['inputs']['views']['right']['camera']['left_to_view'])
            polys=[np.array(poly,np.int32)*2 for poly in regions]
            up,d,check,inliers=estimate(orig,cams,T,polys)
            # Preserve rejected estimates as provisional diagnostics, with no contact conclusion.
            obj=g.OUT/'objects'/row['subject']/row['obj'];assert sha(obj)==row['source_obj_sha256']
            verts,faces=load_obj(obj)
            par={k:z[n][i] for k,n in [('global_orient','global0'),('body_pose','body_pose'),('betas','betas'),('transl','translation')]}
            v,j,_,_=g.geometry(model.layer['neutral'],par,z['delta'][i],False,None)
            assert np.max(np.abs(v-verts))<1.e-8
            pred=j[g.common.NATIVE_IDS];gt=np.array(truth[pair]['gt'])
            def axis(x):return x[[1,4]].mean(0)-x[[7,11]].mean(0)
            forward=(g.rot(z['delta'][i])@g.rot(z['global0'][i]))[:,2]
            legs=model.layer['neutral'].lbs_weights.argmax(1).numpy()
            sole={side:float(np.min(verts[np.isin(legs,idx)]@up+d)*1000) for side,idx in [('left',[7,10]),('right',[8,11])]}
            results.append(dict(pair=pair,obj_sha256=sha(obj),up_left_cv=up.tolist(),camera_height_m=d,
                validation=check,pred_trunk_from_up_deg=angle(axis(pred),up),gt_trunk_from_up_deg=angle(axis(gt),up),
                trunk_axis_error_deg=angle(axis(pred),axis(gt)),foot_lowest_mm=sole,
                mesh_lowest_mm=float((verts@up+d).min()*1000),
                pred_ankle_height_mm=((pred[[13,9]]@up+d)*1000).tolist(),
                gt_ankle_height_mm=((gt[[13,9]]@up+d)*1000).tolist(),
                side_image_sha256=draw(pair,verts,faces,pred,gt,up,d,forward,orig[0],polys,inliers,check['accepted'])))
    write(RUN/'evidence/planes.json',dict(rows=results,base='0c8780119a54874daf7cdff7c6a862b130050c3b',command='<BODY_PYTHON> -Bu assess.py',exit_status=0))
if __name__=='__main__':main()
