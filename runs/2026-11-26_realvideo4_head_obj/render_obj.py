"""Render four refitted meshes and matched before/after views in the old camera frame."""
from pathlib import Path
import sys
import zipfile
import cv2
import numpy as np
import torch

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
OUT = ROOT/'outputs/realvideo4_head_obj_20261126'
NEW = ROOT/'outputs/realvideo4_head_mount_20261126'
OLD = RUN.parent/'2026-11-25_realvideo4_dv4'
sys.path.insert(0,str(OLD))
import gallery as g


def titled(im,label):
    im=cv2.copyMakeBorder(im,34,0,0,0,cv2.BORDER_CONSTANT,value=g.DARK)
    g.text(im,label,(12,24),.62)
    return im


def save(name,img):
    path=RUN/'assets'/name
    path.parent.mkdir(exist_ok=True)
    assert cv2.imwrite(str(path),img,[cv2.IMWRITE_JPEG_QUALITY,92])
    assert cv2.imread(str(path)).shape==img.shape
    return g.sha(path)


def main():
    torch.set_num_threads(1)
    model,_=g.v3.ref.common.setup_model()
    faces=np.asarray(model.face,np.int32)
    layer=model.layer['neutral'].eval()
    checks=g.read(RUN.parent/'2026-11-26_realvideo4_head_mount/evidence/body.json')['rows']
    baseline=g.read(OLD/'evidence/results.json')['cases']
    rows={}
    tiles=[]
    for key in sorted(checks):
        param=NEW/'poses'/(key+'.npz')
        assert g.sha(param)==checks[key]['v4_pose_sha256']
        with np.load(param) as z:
            x=z['locked']
            body=dict(global_orient=z['global0'],body_pose=x[:21],transl=x[22],betas=z['betas'],
                      left_hand_pose=x[23:38],right_hand_pose=x[38:])
        v,_,_,_=g.geometry(layer,body,x[21],False,None)
        path=OUT/(key+'_refit_cam3_m.obj')
        with path.open('w') as f:
            f.write('# Refitted Golden D v4, cam3 OpenCV coordinates, meters\n')
            np.savetxt(f,v,fmt='v %.9f %.9f %.9f')
            np.savetxt(f,faces+1,fmt='f %d %d %d')
        refit,topology=g.load_obj(path)
        diff=float(np.max(abs(v-refit)))
        assert diff<1e-8 and np.array_equal(faces,topology)
        old_path=g.OUT/'export'/baseline[key]['obj']['name']
        assert g.sha(old_path)==baseline[key]['obj']['sha256']
        old,oldfaces=g.load_obj(old_path)
        assert np.array_equal(oldfaces,faces)
        old_pose=g.OUT/'poses'/(key+'.npz')
        assert g.sha(old_pose)==g.read(OLD/'evidence'/(key+'.json'))['pose_hash']
        with np.load(old_pose) as z:
            root=g.rot(z['locked'][21])@g.rot(z['global0'])
        bases=[root,root@g.rot([0.,np.pi/2,0.])]
        union=np.vstack([old,refit])
        coords=[union@basis for basis in bases]
        scale=min(min(315/np.ptp(q[:,0]),440/np.ptp(q[:,1])) for q in coords)
        centers=[(q.min(0)+q.max(0))/2 for q in coords]
        panels=[]
        for view,basis,center in zip(('Front','Side'),bases,centers):
            for name,mesh in [('Original',old),('Refit',refit)]:
                im=g.free_render(mesh,faces,basis,center,scale)
                assert np.count_nonzero(im[:,:,0]<230)>300
                panels.append(titled(im,f'{key}   {view}   {name}'))
        tiles.append(cv2.resize(np.hstack([panels[1],panels[3]]),(560,411),interpolation=cv2.INTER_AREA))
        rows[key]=dict(pose_sha256=g.sha(param),original_obj_sha256=g.sha(old_path),obj=path.name,
                     obj_sha256=g.sha(path),max_vertex_readback_error_m=diff,vertices=len(v),faces=len(faces),
                     comparison_sha256=save(key+'.jpg',np.hstack(panels)))
    assert len(tiles)==4
    overview=save('refits.jpg',np.vstack([np.hstack(tiles[:2]),np.hstack(tiles[2:])]))
    bundle=OUT/'four_refitted_OBJ_cam3_m.zip'
    with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_DEFLATED) as f:
        for row in rows.values():f.write(OUT/row['obj'],arcname=row['obj'])
    with zipfile.ZipFile(bundle) as f:
        import hashlib
        assert f.testzip() is None and len(f.namelist())==4
        for row in rows.values():assert hashlib.sha256(f.read(row['obj'])).hexdigest()==row['obj_sha256']
    g.write(RUN/'evidence/render.json',dict(base_commit='6c068dab49905b2f93334b77fb177dca06df32e2',dirty_worktree=True,
            rows=rows,overview_sha256=overview,bundle_sha256=g.sha(bundle),command='bash <RUN>/run.sh',exit_status=0))

if __name__=='__main__':main()
