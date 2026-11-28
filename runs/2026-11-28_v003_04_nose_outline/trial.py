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
PRIVATE = RUN.parents[1] / 'outputs/v003_04_nose_outline_20261128'
sys.path.insert(0, str(RUN.parent / '2026-11-27_video34_elbow_audit'))
import audit as a
from calibrate import rays

KEY = 'v003_04'
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
pose_file = a.OUT / 'palm/poses' / (KEY+'.npz')
prep = a.OUT / 'fit/0/prepared.json'
det = a.OUT / 'predictions.json'
for path, field in [(pose_file,'palm_sha256'),(prep,'prepared_sha256'),(det,'predictions_sha256')]:
    assert a.sha(path) == previous[field]
pack = dict(np.load(pose_file))
row = next(r for r in a.read(prep)['items'] if r['pair'] == KEY)
cams = list(map(a.read, a.cameras()))
assert list(map(a.sha,a.cameras())) == previous['cameras_sha256']
tfs = [np.array(row['inputs']['views'][s]['camera']['left_to_view']) for s in SIDES]
pred = a.read(det)
target = a.elbow_observations(row,['L','R'],pred,KEY)
valid = ((target>=0)&(target<[1280,720])).all(2)
base = a.replay(pack,target,valid,tfs,model,camera,row,KEY,1)
nose = np.array([pred[KEY+'_'+s+'.png']['xy'][178] for s in SIDES])
legacy = np.array(previous['observed_xy'])
mouth = np.array(previous['near_mouth_xy'])
seg_records = a.read(a.PRIOR/'evidence/segmentation.json')['eyes']
images, masks, edges, seg_hashes = [], [], [], []
for e,side in enumerate(SIDES):
    name = KEY+'_'+side
    photo = a.OUT/'images'/(name+'.png')
    seg_file = a.OUT/'seg'/(name+'.npy')
    assert a.sha(photo) == previous['image_sha256'][e]
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
init=least_squares(nose_error,zero,jac=jac,bounds=(-35,35)).x.reshape(2,3)
init*=np.minimum(1,34.5/np.maximum(np.linalg.norm(init,axis=1,keepdims=True),.01))
best=min([refine(z) for z in [zero,init.ravel(),np.array([12.,0,0,-8,0,0])]],key=loss)
old=np.array(previous['results']['local_joint']['local_rotvec_deg']).ravel()
deg={'previous':old,'nose_outline':best}
mouth_ids=[JOINT_NAMES.index(side+'_mouth_1') for side in SIDES]
base_j=geometry(zero)[1]
results={}
public, private=[],[]
for name,angles in deg.items():
    v,j=geometry(angles)
    s=scene(v)
    lines=sample(s,auto_grid,fit_rows)
    old_y=sample(s,old_grid,eval_rows)
    oral=project(j[mouth_ids])[[0,1],[0,1]]
    held=np.array(old_y)[:,1::2]-legacy[:,1::2,1]
    arm_shift=np.linalg.norm(j[16:22]-base_j[16:22],axis=1)*1000
    assert arm_shift.max()<.001
    results[name]=dict(nose_rmse=rmse(tip(v)-nose),outline_mae=gap(lines,edges),
        legacy_held_mae=float(np.abs(held).mean()),mouth_rmse=rmse(oral-mouth),
        nose_uv=tip(v).tolist(),mouth_uv=oral.tolist(),outline_y=[l.tolist() for l in lines],
        legacy_y=[l.tolist() for l in old_y],local_deg=angles.reshape(2,3).tolist(),arm_shift_mm=arm_shift.tolist())
    print(name,[(k,round(results[name][k],2)) for k in ['nose_rmse','outline_mae','legacy_held_mae','mouth_rmse']],flush=True)
    tiles=[[],[]]
    for e,x0 in enumerate([710,0]):
        xs=np.arange(x0,x0+570);ys=np.arange(400,720)
        rendered=face_mask(s,ray_grid(e,xs,ys),len(xs)).astype('uint8')
        line=cv2.morphologyEx(rendered,cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8)).astype(bool)
        skin=cv2.resize(masks[e].astype('uint8'),(1280,720),interpolation=cv2.INTER_NEAREST)[400:,x0:x0+570]
        for m,tile in enumerate([np.full((320,570,3),45,np.uint8),images[e][400:,x0:x0+570].copy()]):
            if m==0:tile[skin.astype(bool)]=(80,80,80)
            tile[line]=(60,175,255)
            for pt in edges[e]:
                cv2.circle(tile,tuple(pt-[x0,400]),2,(220,235,0),-1)
            for pt,col in [(nose[e],(220,235,0)),(tip(v)[e],(60,175,255))]:
                cv2.drawMarker(tile,tuple(np.rint(pt-[x0,400]).astype(int)),col,cv2.MARKER_CROSS,14,2)
            tiles[m].append(a.title(tile,name+' | cam'+str(e+3)))
    public.append(np.hstack(tiles[0]));private.append(np.hstack(tiles[1]))
for key,field in [('legacy_held_mae','held_mae'),('mouth_rmse','mouth_rmse')]:
    assert abs(results['previous'][key]-previous['results']['local_joint'][field])<.1
assert cv2.imwrite(str(RUN/'assets/comparison.png'),np.vstack(public))
assert cv2.imwrite(str(PRIVATE/'photo_comparison.jpg'),np.vstack(private))
a.write(RUN/'evidence/results.json',dict(pair=KEY,model_nose_vertex=int(nose_vertex),det_channel=178,
    nose_uv=nose.tolist(),observed_outline=[edge.tolist() for edge in edges],legacy_xy=legacy.tolist(),
    mouth_xy=mouth.tolist(),results=results,images=previous['image_sha256'],segmentation=seg_hashes,
    predictions_sha256=a.sha(det),palm_sha256=a.sha(pose_file),baseline_receipt=a.sha(OLD/'evidence/results.json'),
    evaluations=len(cache),exit_status=0))
