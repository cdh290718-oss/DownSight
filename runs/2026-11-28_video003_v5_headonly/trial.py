"""Fit a single calibrated stereo pair to a detected nose and visible facial outline."""
import sys
from pathlib import Path
import cv2
import numpy as np
import open3d as o3d
import torch
from scipy.optimize import least_squares
from smplx.joint_names import JOINT_NAMES
from smplx.vertex_ids import vertex_ids

RUN = Path(__file__).resolve().parent
OLD = RUN.parent / '2026-11-27_v003_04_free_neck'
PRIVATE = RUN.parents[1] / 'outputs/video003_v5_headonly_20261128'
sys.path.insert(0, str(RUN.parent / '2026-11-27_video34_elbow_audit'))
import audit as a
from calibrate import rays

KEY = 'v003_%02d' % (int(sys.argv[1])+1)
SIDES = ('left', 'right')
ROT = [11, 14]
torch.set_num_threads(1)
cv2.setNumThreads(1)
previous = a.read(OLD / 'evidence/results.json')
ref = a.original.w.ref
ref.OLD = ref.OUT = ref.compare.OUT = a.OUT
ref.PRIOR = a.PRIOR
ref.cameras = ref.prev.ref.camera_files = ref.prev.base.fit.camera_files = a.cameras
model, camera = ref.prev.ref.common.setup_model()
layer = model.layer['neutral'].eval()
faces = np.asarray(model.face, np.int32)
topology = o3d.core.Tensor(faces.astype(np.uint32))
face_ids = np.isin(layer.lbs_weights.argmax(1).numpy(), [15,22,23,24])[faces].all(1)
nose_vertex = vertex_ids['smplx']['nose']
pose_file = RUN.parents[1] / 'outputs/video003_strict_v5_20261127/poses' / (KEY+'_v5.npz')
prep = a.OUT / 'fit/0/prepared.json'
det = a.OUT / 'predictions.json'
strict_receipt=a.read(RUN.parent/'2026-11-27_video003_strict_v5/evidence'/(KEY+'_render.json'))
strict_obj=RUN.parents[1]/'outputs/video003_strict_v5_20261127/export'/strict_receipt['obj']
assert a.sha(strict_obj)==strict_receipt['obj_sha256']
pack = dict(np.load(pose_file))
row = next(r for r in a.read(prep)['items'] if r['pair'] == KEY)
cams = list(map(a.read, a.cameras()))
assert list(map(a.sha,a.cameras())) == previous['cameras_sha256']
tfs = [np.array(row['inputs']['views'][s]['camera']['left_to_view']) for s in SIDES]
pred = a.read(det)
target = a.elbow_observations(row,['L','R'],pred,KEY)
valid = ((target>=0)&(target<[1280,720])).all(2)
base = pack['locked'].copy()
nose = np.array([pred[KEY+'_'+s+'.png']['xy'][178] for s in SIDES])
legacy = np.array(previous['observed_xy'])
mouth = np.array(previous['near_mouth_xy'])
seg_records = a.read(a.PRIOR/'evidence/segmentation.json')['eyes']
images, masks, edges, seg_hashes = [], [], [], []
for e,side in enumerate(SIDES):
    name = KEY+'_'+side
    photo = a.OUT/'images'/(name+'.png')
    seg_file = a.OUT/'seg'/(name+'.npy')
    assert a.sha(photo) == seg_records[name]['image_sha256']
    seg_hashes.append(a.sha(seg_file))
    assert seg_hashes[-1] == seg_records[name]['sampled_sha256']
    images.append(cv2.imread(str(photo)))
    labels = np.load(seg_file).argmax(0)
    skin = np.isin(labels,[3,24,25,26,27,28]).astype('uint8')
    _,components = cv2.connectedComponents(skin)
    nx,ny = np.rint((nose[e]-2)/4).astype(int)
    selected = components[ny,nx]
    assert selected > 0
    region = components == selected
    masks.append(region)
    glasses = cv2.dilate((labels==2).astype('uint8'),np.ones((5,5),np.uint8))
    points = []
    for x in range(2,region.shape[1]-2,2):
        ys = np.flatnonzero(region[:,x])
        if len(ys)<3: continue
        y = ys[0]
        if y<2 or glasses[y,x] or labels[y-1,x] not in (0,13,22,23): continue
        points.append([4*x+2,4*y])  # interface halfway between adjacent sampled pixel centers
    edges.append(np.array(points))
    assert len(points)>=8

def project(j):
    return np.stack([a.projection(j@tf[:3,:3].T+tf[:3,3],c) for tf,c in zip(tfs,cams)])

def geometry(degrees):
    p = base.copy()
    p[ROT] = np.deg2rad(degrees).reshape(2,3)
    return a.pack_pose(layer,pack,p)

def ray_grid(eye,xs,ys):
    xx,yy = np.meshgrid(xs,ys)
    tf=tfs[eye]
    d=rays(np.column_stack([xx.ravel(),yy.ravel()]),cams[eye])@tf[:3,:3]
    d/=np.linalg.norm(d,axis=1,keepdims=True)
    origin=-tf[:3,:3].T@tf[:3,3]
    return o3d.core.Tensor(np.column_stack([np.broadcast_to(origin,d.shape),d]).astype('float32'))

def scene(v):
    s=o3d.t.geometry.RaycastingScene(nthreads=1)
    s.add_triangles(o3d.core.Tensor(np.ascontiguousarray(v,np.float32)),topology)
    return s

def face_mask(s,grid,width):
    idx=s.cast_rays(grid,nthreads=1)['primitive_ids'].numpy()
    hit=idx<len(faces)
    hit[hit]=face_ids[idx[hit]]
    return hit.reshape(-1,width)

def make_sampler(points,rows):
    return [(ray_grid(e,edge[:,0],rows),len(edge)) for e,edge in enumerate(points)]

def sample(s,sampler,rows):
    values=[]
    for grid,width in sampler:
        hits=face_mask(s,grid,width)
        values.append(np.where(hits.any(0),rows[hits.argmax(0)],720))
    return values

fit_rows=np.arange(0,720,4)
eval_rows=np.arange(350,720)
auto_grid=make_sampler(edges,fit_rows)
old_grid=make_sampler(legacy,eval_rows)
def tip(v): return project(v[[nose_vertex]])[:,0]
def rmse(diff): return float(np.sqrt((diff**2).sum(1).mean()))
def gap(lines,points): return float(np.mean([np.abs(y-p[:,1]).mean() for y,p in zip(lines,points)]))
cache={}
def loss(x):
    key=tuple(x)
    if max(np.linalg.norm(x.reshape(2,3),axis=1))>35: return np.inf
    if key not in cache:
        v,_=geometry(x)
        y=sample(scene(v),auto_grid,fit_rows)
        cache[key]=rmse(tip(v)-nose)+gap(y,edges)+.003*(x*x).sum()
    return cache[key]

def nose_error(x): return np.r_[(tip(geometry(x)[0])-nose).ravel(), .45*x]
def jac(x): return np.column_stack([(nose_error(x+d)-nose_error(x-d))/.2 for d in .1*np.eye(6)])
def refine(x):
    for step in (8,4,2,1,.5):
        for _ in range(20):
            opts=[x]+[x+d for d in step*np.vstack([np.eye(6),-np.eye(6)])]
            best=min(opts,key=loss)
            if np.array_equal(x,best): break
            x=best
    return x
zero=np.zeros(6)
init=least_squares(nose_error,zero,jac=jac,bounds=(-35,35)).x
rot=init[:6].reshape(2,3)
rot*=np.minimum(1,34.5/np.maximum(np.linalg.norm(rot,axis=1,keepdims=True),.01))
best=min([refine(z) for z in [zero,init,np.array([12.,0,0,-8,0,0])]],key=loss)
pose=base.copy()
pose[ROT]=np.deg2rad(best).reshape(2,3)
v,j=a.pack_pose(layer,pack,pose)
_,before_joints=a.pack_pose(layer,pack,base)
body_ids=[i for i in range(22) if i not in (12,15)]
joint_drift=float(np.linalg.norm(j[body_ids]-before_joints[body_ids],axis=1).max()*1000)
assert joint_drift < .001
fixed=[i for i in range(len(base)) if i not in ROT]
assert np.array_equal(pose[fixed],base[fixed])
s=scene(v)
before_v,before_j=a.pack_pose(layer,pack,base)
old_vertices,old_faces=__import__('render_result').load_obj(strict_obj)
assert np.array_equal(old_faces,faces) and np.max(abs(old_vertices-before_v))<1e-8
auto_y=sample(s,auto_grid,fit_rows)
arm_uv=project(j[[18,19,20,21]])
err=np.linalg.norm(arm_uv-target,axis=2)
arm=[float(np.sqrt(np.mean(err[:,sl][valid[:,sl]]**2))) if valid[:,sl].any() else None for sl in (slice(2),slice(2,4))]
out=PRIVATE/'export';out.mkdir(exist_ok=True)
obj=out/(KEY+'.obj')
with obj.open('x') as f:
 np.savetxt(f,v,fmt='v %.9f %.9f %.9f')
 np.savetxt(f,faces+1,fmt='f %d %d %d')
readback,topo=__import__('render_result').load_obj(obj)
assert np.max(abs(readback-v))<1e-8 and np.array_equal(topo,faces)
np.savez_compressed(PRIVATE/'poses'/(KEY+'.npz'),**dict(pack,locked=pose))
from render_selection import full_body
root=a.original.w.old.mat(pose[21])@a.original.w.old.mat(pack['global0'])
body=full_body(v,faces,root)
assert cv2.imwrite(str(RUN/'assets'/(KEY+'_obj.jpg')),body)
full_tiles=[]; debug_tiles=[]
for e in range(2):
 xs=np.arange(0,1280,2);ys=np.arange(0,720,2)
 idx=s.cast_rays(ray_grid(e,xs,ys),nthreads=1)['primitive_ids'].numpy()
 hit=idx<len(faces); facehit=np.zeros_like(hit);facehit[hit]=face_ids[idx[hit]]
 hit=hit.reshape(len(ys),len(xs));facehit=facehit.reshape(len(ys),len(xs))
 image=images[e].copy()
 mask_full=cv2.resize(hit.astype('uint8'),(1280,720),interpolation=cv2.INTER_NEAREST).astype(bool)
 face_full=cv2.resize(facehit.astype('uint8'),(1280,720),interpolation=cv2.INTER_NEAREST).astype(bool)
 overlay=image.copy();overlay[mask_full]=[200,155,120];overlay[face_full]=[200,80,190]
 blend=cv2.addWeighted(image,.6,overlay,.4,0)
 uv=project(j);det=project(before_j)
 for p in row['points']:
  if p['valid'][e]:
   x,y=np.rint(p['xy'][e]).astype(int);q=tuple(np.rint(uv[e,p['native_index']]).astype(int))
   cv2.drawMarker(blend,(x,y),(255,255,0),cv2.MARKER_TILTED_CROSS,9,1);cv2.circle(blend,q,4,(0,140,255),1)
 full_tiles.append(a.title(blend,KEY+' cam'+str(e+3)))
 x0=[710,0][e];tile=image[400:,x0:x0+570].copy()
 xx,yy=np.meshgrid(np.arange(x0,x0+570),np.arange(400,720))
 directions=rays(np.column_stack((xx.ravel(),yy.ravel())),cams[e])@tfs[e][:3,:3]
 directions/=np.linalg.norm(directions,axis=1,keepdims=True)
 origin=-tfs[e][:3,:3].T@tfs[e][:3,3]
 beam=np.column_stack((np.broadcast_to(origin,directions.shape),directions)).astype('float32')
 idx=s.cast_rays(o3d.core.Tensor(beam),nthreads=1)['primitive_ids'].numpy()
 visible=idx<len(faces);visible[visible]=face_ids[idx[visible]]
 line=cv2.morphologyEx(visible.reshape(320,570).astype('uint8'),cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))
 tile[line.astype(bool)]=(60,175,255)
 for p in edges[e]:cv2.circle(tile,tuple(p-[x0,400]),2,(220,235,0),-1)
 for p,c in [(nose[e],(220,235,0)),(tip(v)[e],(60,175,255))]:
  cv2.drawMarker(tile,tuple(np.rint(p-[x0,400]).astype(int)),c,cv2.MARKER_CROSS,12,2)
 debug_tiles.append(a.title(tile,KEY+' cam'+str(e+3)))
assert cv2.imwrite(str(PRIVATE/(KEY+'_overlay.jpg')),np.hstack(full_tiles))
assert cv2.imwrite(str(PRIVATE/(KEY+'_head.jpg')),np.hstack(debug_tiles))
result=dict(pair=KEY,source_pose_sha256=a.sha(pose_file),obj_sha256=a.sha(obj),nose_rmse=rmse(tip(v)-nose),outline_mae=gap(auto_y,edges),arm_rmse=arm,translation_mm=[0.,0.,0.],joint_drift_mm=joint_drift,local_deg=best.reshape(2,3).tolist(),nose_uv=nose.tolist(),projected_nose=tip(v).tolist(),edge_points=[e.tolist() for e in edges],evaluations=len(cache),exit_status=0)
a.write(RUN/'evidence'/(KEY+'.json'),result)
print(KEY,result['nose_rmse'],result['outline_mae'],flush=True)
