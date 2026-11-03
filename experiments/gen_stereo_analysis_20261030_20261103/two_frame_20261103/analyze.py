"""Read-only two-time-point Gen stereo / DFM / experiment-C diagnostic.

Writes only previously nonexistent files under this script's unique directory.
Distances are camera-centre ranges; DFM is a reference, not ground truth.
"""
import csv
import hashlib
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = Path('/vepfs-mlp2/mlp-public/huyaoqing/gen_egosmplx_run')
OLD = OUT.parent / 'gen_egosmplx_two_frames_20261030'
C = OLD / 'experiment_C/output'
PREV = OLD / 'stereo_diagnostic_20261031_rkk9qawi'
names = ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
         'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee',
         'right_knee', 'left_ankle', 'right_ankle', 'nose']
dfm_index = dict(left_shoulder=16, right_shoulder=17, left_elbow=18, right_elbow=19,
                 left_wrist=20, right_wrist=21, left_hip=1, right_hip=2,
                 left_knee=4, right_knee=5, left_ankle=7, right_ankle=8)
c_index = dict(left_shoulder=8, right_shoulder=9, left_elbow=10, right_elbow=11,
               left_wrist=12, right_wrist=13, left_hip=1, right_hip=2,
               left_knee=3, right_knee=4, left_ankle=5, right_ankle=6, nose=24)
dfm = np.load(ROOT / 'dfm_reference_virtual.npz', allow_pickle=False)
body = np.load(C / 'frames/gen_00000_cam3/body/params.npz', allow_pickle=False)
cj = np.asarray(body['smplx_joint_cam'], dtype=np.float64)
assert cj.shape == (137, 3)
c_head = cj[66]  # 137-joint mapping index 66 corresponds to the original Head joint.
cal = [json.loads((ROOT / 'calibration' / n).read_text())['sourceParams'] for n in
       ('gen_cam2_virtual_cam3.egosmplx.json', 'gen_cam3_virtual_cam4.egosmplx.json')]
trans = json.loads((PREV / 'transform.json').read_text())
R = np.array(trans['R_cam4_from_cam3'], dtype=float)
t = np.array(trans['t_cam4_from_cam3_m'], dtype=float)
center4 = -R.T @ t
assert np.allclose(R.T @ R, np.eye(3), atol=1e-9) and np.linalg.det(R) > .9999


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ray(x, c):
    fx, fy, cx, cy, xi, alpha = c
    mx, my = (x[0]-cx)/fx, (x[1]-cy)/fy
    r2 = mx*mx+my*my
    radical = 1-(2*alpha-1)*r2
    if radical < 0:
        return np.full(3, np.nan)
    mz = (1-alpha*alpha*r2)/(alpha*np.sqrt(radical)+1-alpha)
    radical2 = mz*mz+(1-xi*xi)*r2
    if radical2 < 0:
        return np.full(3, np.nan)
    k = (mz*xi+np.sqrt(radical2))/(mz*mz+r2)
    v = np.array([k*mx, k*my, k*mz-xi])
    return v/np.linalg.norm(v)


def project(x, c):
    fx, fy, cx, cy, xi, alpha = c
    d1 = np.linalg.norm(x)
    z = xi*d1+x[2]
    den = alpha*np.sqrt(x[0]*x[0]+x[1]*x[1]+z*z)+(1-alpha)*z
    return np.array([fx*x[0]/den+cx, fy*x[1]/den+cy]) if den > 0 else np.full(2, np.nan)


def solve(z):
    a, b = ray(z[:2], cal[0]), R.T @ ray(z[2:], cal[1])
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return np.full(3, np.nan), np.full(2, np.nan), np.nan, np.nan
    q = np.linalg.lstsq(np.column_stack((a, -b)), center4, rcond=None)[0]
    x = a*q[0]
    gap = np.linalg.norm(x-(center4+b*q[1]))
    angle = np.degrees(np.arccos(np.clip(a @ b, -1, 1)))
    return x, q, gap, angle


def epi(z):
    return np.dot(ray(z[2:], cal[1]), np.cross(t, R @ ray(z[:2], cal[0])))


def grad(f, z, h=.01):
    result = []
    for i in range(len(z)):
        d = np.zeros_like(z); d[i] = h
        result.append((f(z+d)-f(z-d))/(2*h))
    return np.asarray(result)


def segment_details(labels, uv, classes):
    x, y = np.rint(uv).astype(int)
    if not (0 <= x < labels.shape[1] and 0 <= y < labels.shape[0]):
        return {'inside': False}
    patch = labels[max(0,y-10):min(labels.shape[0],y+11), max(0,x-10):min(labels.shape[1],x+11)]
    ids, counts = np.unique(patch, return_counts=True)
    order = np.argsort(counts)[::-1][:4]
    return {'inside': True, 'center_class': classes[str(int(labels[y,x]))]['name'],
            'neighborhood_21x21': [{'class': classes[str(int(ids[i]))]['name'],
                                    'fraction': float(counts[i]/patch.size)} for i in order]}


def safe(v):
    if isinstance(v, np.ndarray): return [safe(x) for x in v.tolist()]
    if isinstance(v, (np.floating, float)): return float(v) if np.isfinite(v) else None
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (list, tuple)): return [safe(x) for x in v]
    if isinstance(v, dict): return {k: safe(x) for k, x in v.items()}
    return v


def write_json(name, data):
    with (OUT / name).open('x') as f:
        json.dump(safe(data), f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def get_inputs(frame):
    outputs = []
    classes = json.loads((C / 'observations/segmentation/classes.json').read_text())
    for i, cam in enumerate(('cam3','cam4')):
        image = ROOT / 'images' / cam / ('gen_%05d.png' % frame)
        pose = ((C / 'observations' / ('gen_%05d_%s' % (frame,cam)) / 'sapiens/keypoints.json')
                if frame == 0 else
                (OUT / 'new_sapiens' / ('gen_%05d_%s' % (frame,cam)) / 'keypoints.json'))
        data = json.loads(pose.read_text())
        assert data['image_sha256'] == digest(image), (image, pose)
        labels_path = C / 'observations/segmentation/frames' / ('gen_%05d_%s' % (frame,cam)) / 'labels.npy'
        outputs.append(dict(image=image, pose=pose, keypoints={k['name']: k for k in data['keypoints']},
                            labels=np.load(labels_path) if labels_path.exists() else None, classes=classes))
    return outputs


def make_frame(frame):
    idx = int(np.flatnonzero(dfm['frames'] == frame)[0])
    refs = [np.asarray(dfm['cam3_body_xyz'][idx]), np.asarray(dfm['cam4_body_xyz'][idx])]
    uv_refs = [np.asarray(dfm['cam3_body_uv'][idx]), np.asarray(dfm['cam4_body_uv'][idx])]
    inputs = get_inputs(frame)
    reproj_err = [max(np.linalg.norm(project(refs[k][n], cal[k])-uv_refs[k][n]) for n in range(22)) for k in (0,1)]
    assert max(reproj_err) < 1e-5
    assert np.max(np.linalg.norm(refs[0] @ R.T + t - refs[1],axis=1)) < 1e-8
    rows = []
    for name in names:
        k3,k4 = (q['keypoints'][name] for q in inputs)
        z = np.array([k3['x'],k3['y'],k4['x'],k4['y']], dtype=float)
        x, dist, gap, angle = solve(z)
        grad_epi = grad(epi,z)
        sampson = abs(epi(z))/np.linalg.norm(grad_epi)
        forward = bool(np.isfinite(dist).all() and np.all(dist > 0))
        sigma = float(3*np.linalg.norm(grad(lambda zz: np.linalg.norm(solve(zz)[0]),z))) if forward else np.nan
        sigma3 = float(3*np.linalg.norm(np.vstack([grad(lambda zz: solve(zz)[0][i],z) for i in range(3)]),'fro')) if forward else np.nan
        di = dfm_index.get(name)
        ref3 = refs[0][di] if di is not None else None
        refuv = [uv_refs[k][di] for k in (0,1)] if di is not None else [None,None]
        cpt = cj[c_index[name]] if frame == 0 else None
        residual3 = np.linalg.norm(x-ref3) if forward and ref3 is not None else np.nan
        labels = []
        for k in (0,1):
            labels.append(dict(sapiens=segment_details(inputs[k]['labels'],z[2*k:2*k+2],inputs[k]['classes']),
                               dfm=segment_details(inputs[k]['labels'],refuv[k],inputs[k]['classes']) if refuv[k] is not None else None)
                          if inputs[k]['labels'] is not None else None)
        row = dict(frame=frame,joint=name,uv_cam3=z[:2],uv_cam4=z[2:],score_cam3=k3['score'],score_cam4=k4['score'],
                   inside_cam3=k3['inside_image'],inside_cam4=k4['inside_image'],
                   epi_sampson_px=sampson,epi_flag_gt9px=bool(sampson>9),angle_deg=angle,
                   forward=forward,ray_gap_cm=gap*100,tri_xyz_cam3_m=x if forward else None,
                   tri_range_cm=np.linalg.norm(x)*100 if forward else np.nan,
                   tri_range_sigma_cm_assuming_3px_independent=sigma*100,
                   tri_3d_rms_sigma_cm_assuming_3px_independent=sigma3*100,
                   dfm_xyz_cam3_m=ref3,dfm_uv_cam3=refuv[0],dfm_uv_cam4=refuv[1],
                   dfm_range_cm=np.linalg.norm(ref3)*100 if ref3 is not None else np.nan,
                   tri_minus_dfm_range_cm=(np.linalg.norm(x)-np.linalg.norm(ref3))*100 if forward and ref3 is not None else np.nan,
                   tri_dfm_3d_error_cm=residual3*100,
                   sapiens_dfm_uv_error_cam3_px=np.linalg.norm(z[:2]-refuv[0]) if refuv[0] is not None else np.nan,
                   sapiens_dfm_uv_error_cam4_px=np.linalg.norm(z[2:]-refuv[1]) if refuv[1] is not None else np.nan,
                   c_xyz_cam3_m=cpt,c_range_cm=np.linalg.norm(cpt)*100 if cpt is not None else np.nan,
                   c_uv_cam3=project(cpt,cal[0]) if cpt is not None else None,
                   c_uv_cam4=project(R@cpt+t,cal[1]) if cpt is not None else None,
                   tri_to_dfm_head_cm=np.linalg.norm(x-refs[0][15])*100 if forward else np.nan,
                   dfm_to_dfm_head_cm=np.linalg.norm(ref3-refs[0][15])*100 if ref3 is not None else np.nan,
                   c_to_c_head_cm=np.linalg.norm(cpt-c_head)*100 if cpt is not None else np.nan,
                   tri_to_c_head_cm=np.linalg.norm(x-c_head)*100 if forward and frame == 0 else np.nan,
                   segmentation=labels)
        rows.append(row)
    return dict(frame=frame, dfm_index=idx, rows=rows, inputs=inputs,
                dfm_head=refs[0][15], projection_max_error_px=reproj_err)


def write_tables(frames):
    fields = ['frame','joint','inside_cam3','inside_cam4','score_cam3','score_cam4',
              'epi_sampson_px','epi_flag_gt9px','forward','angle_deg','ray_gap_cm',
              'tri_range_cm','dfm_range_cm','tri_minus_dfm_range_cm','tri_dfm_3d_error_cm',
              'tri_range_sigma_cm_assuming_3px_independent','tri_3d_rms_sigma_cm_assuming_3px_independent',
              'sapiens_dfm_uv_error_cam3_px','sapiens_dfm_uv_error_cam4_px','c_range_cm',
              'tri_to_dfm_head_cm','dfm_to_dfm_head_cm','tri_to_c_head_cm','c_to_c_head_cm']
    with (OUT/'joints.csv').open('x', newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for item in frames:
            for r in item['rows']:
                q=safe(r);w.writerow({k:q.get(k) for k in fields})
    bones = [('left_upper_arm','left_shoulder','left_elbow'),('right_upper_arm','right_shoulder','right_elbow'),
             ('left_forearm','left_elbow','left_wrist'),('right_forearm','right_elbow','right_wrist'),
             ('shoulder_width','left_shoulder','right_shoulder')]
    result = []
    for item in frames:
        r={x['joint']:x for x in item['rows']}
        for n,a,b in bones:
            qa,qb=r[a],r[b]
            tri=np.linalg.norm(qa['tri_xyz_cam3_m']-qb['tri_xyz_cam3_m'])*100 if qa['forward'] and qb['forward'] else np.nan
            d=np.linalg.norm(qa['dfm_xyz_cam3_m']-qb['dfm_xyz_cam3_m'])*100
            c=np.linalg.norm(cj[c_index[a]]-cj[c_index[b]])*100
            result.append(dict(frame=item['frame'],bone=n,tri_cm=tri,dfm_cm=d,c_frame0_reference_cm=c,
                 tri_minus_c_cm=tri-c,dfm_minus_c_cm=d-c,one_or_both_endpoints_outside=not all([qa['inside_cam3'],qa['inside_cam4'],qb['inside_cam3'],qb['inside_cam4']]),
                 one_or_both_endpoints_epi_gt9=qa['epi_flag_gt9px'] or qb['epi_flag_gt9px']))
    with (OUT/'bones.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader()
        for row in result:w.writerow(safe(row))
    return result


def save_fig(fig,name):
    p=OUT/name
    with p.open('xb') as f:fig.savefig(f, format='png', dpi=140, bbox_inches='tight')
    plt.close(fig)


def plot_all(item):
    fig, axes = plt.subplots(1,2,figsize=(19,6.5))
    for k,ax in enumerate(axes):
        im=cv2.cvtColor(cv2.imread(str(item['inputs'][k]['image'])),cv2.COLOR_BGR2RGB)
        ax.imshow(im);ax.set_xlim(0,1280);ax.set_ylim(720,0);ax.axis('off')
        ax.set_title('frame %05d cam%d: cyan Sapiens; orange DFM; magenta C (only frame 0)'%(item['frame'],k+3))
        for r in item['rows']:
            s=r['uv_cam3' if k==0 else 'uv_cam4'];d=r['dfm_uv_cam3' if k==0 else 'dfm_uv_cam4'];c=r['c_uv_cam3' if k==0 else 'c_uv_cam4']
            if 0<=s[0]<1280 and 0<=s[1]<720:
                ax.scatter(*s,c='cyan',s=30,edgecolor='black',zorder=3)
                ax.annotate(r['joint'].replace('left_','L_').replace('right_','R_'),s,xytext=(4,-5),textcoords='offset points',fontsize=6,color='white',bbox={'facecolor':'black','alpha':.65,'pad':1})
            if d is not None and 0<=d[0]<1280 and 0<=d[1]<720:ax.scatter(*d,c='orange',s=30,marker='+',linewidth=1.5,zorder=4)
            if c is not None and 0<=c[0]<1280 and 0<=c[1]<720:ax.scatter(*c,c='magenta',s=18,marker='x',linewidth=1,zorder=3)
    save_fig(fig,'all_joints_%05d.png'%item['frame'])


def plot_elbows(frames):
    fig, axes=plt.subplots(4,2,figsize=(13,18))
    for row,(item,side) in enumerate([(x,s) for x in frames for s in ('left','right')]):
        rec=next(r for r in item['rows'] if r['joint']==side+'_elbow')
        for k,ax in enumerate(axes[row]):
            inp=item['inputs'][k];im=cv2.cvtColor(cv2.imread(str(inp['image'])),cv2.COLOR_BGR2RGB);ax.imshow(im)
            if inp['labels'] is not None:
                mask=np.isin(inp['labels'],[7,11,16,20]).astype(np.uint8)
                contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    if len(contour)>3:
                        xy=contour[:,0,:];ax.plot(xy[:,0],xy[:,1],color='lime',linewidth=.7,alpha=.8)
            s=rec['uv_cam3' if k==0 else 'uv_cam4'];d=rec['dfm_uv_cam3' if k==0 else 'dfm_uv_cam4'];c=rec['c_uv_cam3' if k==0 else 'c_uv_cam4']
            ax.scatter(*s,c='cyan',s=80,marker='o',edgecolor='black',zorder=5,label='Sapiens')
            ax.scatter(*d,c='orange',s=120,marker='+',linewidth=2.5,zorder=5,label='DFM projected')
            if c is not None:ax.scatter(*c,c='magenta',s=85,marker='x',linewidth=2,zorder=5,label='C projected')
            valid=[p for p in (s,d,c) if p is not None and 0<=p[0]<1280 and 0<=p[1]<720]
            lo=np.min(valid,axis=0)-90;hi=np.max(valid,axis=0)+90
            ax.set_xlim(max(0,lo[0]),min(1280,hi[0]));ax.set_ylim(min(720,hi[1]),max(0,lo[1]));ax.axis('off')
            ax.set_title('frame %05d %s elbow cam%d | separation %.1f px | %s'%
                         (item['frame'],side,k+3,rec['sapiens_dfm_uv_error_cam3_px' if k==0 else 'sapiens_dfm_uv_error_cam4_px'],
                          'green = Sapiens arm mask' if inp['labels'] is not None else 'no segmentation cache'),fontsize=9)
            ax.legend(loc='lower left',fontsize=7)
    fig.tight_layout()
    save_fig(fig,'elbow_evidence.png')


if __name__=='__main__':
    frames=[make_frame(x) for x in (0,270)]
    bones=write_tables(frames)
    for item in frames:plot_all(item)
    plot_elbows(frames)
    serial=[]
    for item in frames:
        serial.append(dict(frame=item['frame'],dfm_index=item['dfm_index'],rows=item['rows'],dfm_head=item['dfm_head'],
                    projection_max_error_px=item['projection_max_error_px'],
                    source_files=[dict(image=str(x['image']), image_sha256=digest(x['image']),
                                       sapiens=str(x['pose']),sapiens_sha256=digest(x['pose'])) for x in item['inputs']]))
    write_json('measurements.json',dict(frames=serial,bones=bones,dfm_index_assumption='standard SMPL 22 ordering, no name metadata in NPZ; coordinate definitions may differ',
        stereo_transform=trans,dfm_is_ground_truth=False,
        epi_threshold='9 px illustrative; sigma=3 px independent, no calibration/sync bias',
        head_distance='triangles relative to same-frame DFM Head; C relative to C Head; no C frame 270',
        c_frame0_body_file=str(C/'frames/gen_00000_cam3/body/params.npz'),c_head_cam3_m=c_head,
        c_head_vs_dfm_frame0_head_cm=np.linalg.norm(c_head-frames[0]['dfm_head'])*100,
        segmentation='existing Sapiens 29-class labels for frame 0 only; labels are predictions, not anatomical truth',
        range_definition='Euclidean distance from cam3 centre; closest-point on cam3 ray',
        point_3d_uncertainty='linearized independent 3px standard deviation, excludes systematics and DFM error'))
    for item in frames:
        print('FRAME',item['frame'])
        for r in item['rows']:
            if r['joint'] in ('left_elbow','right_elbow','left_wrist','right_wrist','left_hip','right_hip','left_knee','left_ankle'):
                print(r['joint'],'tri',round(r['tri_range_cm'],1),'DFM',round(r['dfm_range_cm'],1),'3d-error',round(r['tri_dfm_3d_error_cm'],1),'epi',round(r['epi_sampson_px'],1),'seg',safe(r['segmentation']))
