"""Read selected v3 outputs, measure full-model geometry, render diagnostically relevant views."""
from pathlib import Path
import runpy
import cv2
import numpy as np

# Load the local experiment without shadowing v3's own module named experiment.
settings = runpy.run_path(str(Path(__file__).with_name('experiment.py')), run_name='head_mount')
from render_result import annotated, free_render
RUN, ROOT, OUT, SOURCE, original, v3, read, write, sha, geometry, rot, CASES = (
    settings[k] for k in ('RUN','ROOT','OUT','SOURCE','original','v3','read','write','sha','geometry','rot','CASES'))
LABELS = ['Left history', 'Right history', 'History mean', 'Mount', 'Mount -10 deg', 'Mount +10 deg']


def save(name, img):
    path = RUN / 'assets' / (name+'.jpg')
    assert cv2.imwrite(str(path),img,[cv2.IMWRITE_JPEG_QUALITY,93])
    return dict(sha256=sha(path),size=list(img.shape[1::-1]))


def main():
    inputs = read(RUN/'evidence/inputs.json')
    previous = read(original.RUN/'evidence/results.json')
    assert previous['selection_sha256'] == sha(SOURCE/'selected.npz')
    model, Camera = v3.ref.common.setup_model()
    layer = model.layer['neutral'].eval()
    faces = np.asarray(model.face,np.int32)
    head_faces = (layer.lbs_weights.argmax(1).numpy()==15).astype(np.float32)
    raster = runpy.run_path(str(ROOT/'runs/2026-11-04_mesh-projection-review/code/rasterizer.py'))['raster']
    cameras = [Camera(cal) for cal in original.cameras()]
    row = read(SOURCE/'prepared.json')['items'][0]
    points = row['points']
    ids = [q['native_index'] for q in points]
    views = []
    for side,camera in zip(original.sides,cameras):
        im = cv2.imread(str(original.OUT/'images'/('real001_'+side+'.png')))
        h,w = im.shape[:2]
        y,x = np.indices((h,w))
        rays = camera.camera2world_ray(np.stack([x.ravel(),y.ravel()],1)).reshape(h,w,3).astype('f')
        views.append((np.array(row['inputs']['views'][side]['camera']['left_to_view']),im,rays))
    results,images,panels = {},{},[]
    for n,(name,label) in enumerate(zip(CASES,LABELS)):
        folder = SOURCE if n==0 else OUT/'fit/0'
        di = 0 if n==0 else n-1
        meta = read(folder/'selected.json')[di]
        assert meta['pair'] == ('real001' if n==0 else name)
        diagnostic = read(folder/'diagnostic_20000.json')
        assert diagnostic['ids'][di][0] == meta['pair']
        if 'failed' in meta:
            results[name] = meta
            continue
        pack = dict(np.load(folder/'selected.npz'))
        i = meta['offset']+2
        assert not pack['body_pose'][i,[11,14]].any()
        parameters = {k:pack[key][i] for k,key in [('body_pose','body_pose'),('betas','betas'),('global_orient','global0'),('transl','translation')]}
        v,j,head,chest = geometry(layer,parameters,pack['delta'][i],False,None)
        root = rot(pack['delta'][i]) @ rot(pack['global0'][i])
        assert np.allclose(head,chest)
        up = (root.T@chest)[:,1]
        pitch = float(np.rad2deg(np.arctan2(-up[2],up[1])))
        dist = []
        for eye,(camera,(T,im,rays)) in enumerate(zip(cameras,views)):
            det = np.asarray([p['xy'][eye] for p in points])
            valid = np.asarray([p['valid'][eye] for p in points])
            assert valid.sum()==11
            uv = camera.world2camera(j[ids].astype(float)@T[:3,:3].T+T[:3,3])
            dist.append(np.linalg.norm(uv-det,axis=1)[valid])
            if name=='mount':
                vv = (v@T[:3,:3].T+T[:3,3]).astype('f')
                projected = camera.world2camera(vv.astype(float)).astype('f')
                colors = np.zeros_like(im)
                depth = np.full(im.shape[:2],np.inf,dtype='f')
                raster(vv,faces,head_faces,projected,rays,depth,colors,np.zeros_like(depth),180.)
                mask = np.isfinite(depth)
                assert mask.any()
                overlay = im.copy()
                overlay[mask] = (.45*im[mask]+.55*colors[mask]).astype('uint8')
                tag = ['cam3','cam4'][eye]
                images['mount_'+tag] = save('mount_'+tag,annotated(overlay,uv,det,points,valid,tag))
        rmse = float(np.sqrt(np.mean(np.square(dist))))
        metrics = dict(pitch_back_deg=pitch,rmse_px=rmse,by_eye_rmse_px=np.sqrt(np.mean(np.square(dist),axis=1)).tolist(),
            head_residual_mm=float(1000*np.linalg.norm(j[15]-inputs['anchors_cam3_m'][name])),
            brow_to_mount_nominal_mm=float(1000*np.linalg.norm(j[inputs['landmarks']].mean(0)-inputs['brows_cam3_m']['mount'])),
            selected=meta['selected'][-1],probes=meta['probes'],step=diagnostic['selected_step'][di],
            feasible=diagnostic['selected_feasible'][di],convergence=diagnostic['convergence_pass'][di],
            end_window_motion_mm=diagnostic['max_joint_motion_mm'][di],selected_loss=meta['losses'][-1])
        if name in inputs['brows_cam3_m']:
            metrics['brow_to_own_assumed_mount_mm'] = float(1000*np.linalg.norm(j[inputs['landmarks']].mean(0)-inputs['brows_cam3_m'][name]))
        if n==0:
            assert abs(rmse-previous['combined_rmse_px'])<1e-5
        results[name] = metrics
        front_side = []
        for basis in (root, root@rot([0,np.pi/2,0])):
            q = v@basis
            panel = free_render(v,faces,basis,(q.min(0)+q.max(0))/2,235.)
            assert np.ptp(q,axis=0)[1]*235.<480 and np.ptp(q,axis=0)[0]*235.<350
            front_side.append(panel)
        view = cv2.copyMakeBorder(np.hstack(front_side),66,0,0,0,cv2.BORDER_CONSTANT,value=(38,35,34))
        cv2.putText(view,label,(12,27),cv2.FONT_HERSHEY_SIMPLEX,.73,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(view,'Chest pitch %+.1f deg    Stereo RMSE %.2f px'%(pitch,rmse),(12,54),cv2.FONT_HERSHEY_SIMPLEX,.59,(230,230,230),1,cv2.LINE_AA)
        panels.append(view)
    if len(panels)==6:
        images['historical_vs_mount'] = save('historical_vs_mount',np.hstack([panels[0],panels[3]]))
        images['all_front_side'] = save('all_front_side',np.vstack([np.hstack(panels[:3]),np.hstack(panels[3:])]))
    write(RUN/'evidence/results.json',dict(cases=results,images=images,previous_selection=sha(SOURCE/'selected.npz'),
        new_selection=sha(OUT/'fit/0/selected.npz'),fit_complete=read(OUT/'fit/0/fit_complete.json'),exit_status=0))


if __name__=='__main__':
    main()
