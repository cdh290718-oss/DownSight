"""Render the two archived temporal-overlay transforms directly from exported meshes."""
import hashlib
from pathlib import Path
import sys
from zipfile import ZipFile, ZIP_DEFLATED
import cv2
import numpy as np
import open3d as o3d

RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[1]
PITCH=RUN.parent/'2026-11-26_realvideo4_mount_pitch'
V4=RUN.parent/'2026-11-25_realvideo4_dv4'
OUT=ROOT/'outputs/realvideo4_pitch_obj_20261126'
sys.path.insert(0,str(PITCH))
from pitch import Case, inverse_beam, pivot, rot, OUT as GEOMETRY
sys.path.insert(0,str(V4))
from gallery import free_render, load_obj, read, write, sha, rot as body_rotation, OUT as BODY
COLORS={'Purple':(203,102,255),'Green':(94,230,84)}

def header(image,label,color):
    bar=np.full((35,image.shape[1],3),35,np.uint8)
    bar[:,:8]=color
    cv2.putText(bar,label,(18,25),cv2.FONT_HERSHEY_SIMPLEX,.64,(250,250,250),1,cv2.LINE_AA)
    return np.vstack([bar,image])

def save(name,image):
    dest=RUN/'assets'/name
    assert cv2.imwrite(str(dest),image,[cv2.IMWRITE_JPEG_QUALITY,93])
    assert cv2.imread(str(dest)).shape==image.shape
    return sha(dest)

def stereo(v,faces,case,R,t,color):
    scene=o3d.t.geometry.RaycastingScene(nthreads=1)
    scene.add_triangles(o3d.core.Tensor(v.astype('float32')),o3d.core.Tensor(faces.astype('uint32')))
    panels=[]
    mismatch=[]
    for eye,start in enumerate([710,0]):
        yy,xx=np.mgrid[550:720,start:start+570]
        beam=case.beams(np.column_stack([xx.ravel(),yy.ravel()]),eye)
        origin,directions=beam
        rays=np.column_stack((np.broadcast_to(origin,directions.shape),directions)).astype('float32')
        hit=scene.cast_rays(o3d.core.Tensor(rays),nthreads=1)
        ids=hit['primitive_ids'].numpy()
        vis=ids<len(faces)
        vis[vis]=case.labels[ids[vis]]
        ref=case.hit(inverse_beam(beam,R,t),np.zeros(3))
        mismatch.append(int(np.count_nonzero(vis!=ref)))
        assert mismatch[-1]<=max(3,.001*ref.sum())
        lighting=abs((directions[vis]*hit['primitive_normals'].numpy()[vis]).sum(1)).clip(0,1)
        shaded=(.45+.55*lighting[:,None])*np.array(color)
        crop=case.images[eye][550:,start:start+570].copy()
        pixels=crop.reshape(-1,3)
        pixels[vis]=.62*shaded+.38*pixels[vis]
        line=cv2.morphologyEx(vis.reshape(xx.shape).astype('uint8'),cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))
        crop[line.astype(bool)]=color
        for x,y in case.edges[eye]:
            cv2.circle(crop,(int(x)-start,int(y)-550),2,(0,245,255),-1)
        x,y=case.corners[eye]
        cv2.drawMarker(crop,(x-start,y-550),(0,245,255),cv2.MARKER_CROSS,12)
        panels.append(crop)
    return np.hstack(panels),mismatch

def main():
    params=PITCH/'evidence/temporal.json'
    angles=PITCH/'evidence/angles.json'
    hashes=read(PITCH/'evidence/execution.json')['sha256']
    for p in (params,angles):
        assert sha(p)==hashes['evidence/'+p.name]
    settings=read(params)
    candidates={'Purple':settings['zero'],'Green':settings['selected']}
    assert [c['deg'] for c in candidates.values()]==[0,-10]
    rows={}
    images={}
    full=[]
    bundle=OUT/'clip003_004_purple_green_obj.zip'
    with ZipFile(bundle,'w',ZIP_DEFLATED) as archive:
        for key in ('clip003','clip004'):
            path=GEOMETRY/(key+'.npz')
            assert sha(path)==read(angles)['inputs']['mesh_sha256'][key]
            with np.load(path) as pack:
                base,faces=pack['v'],pack['f']
            pose=BODY/'poses'/(key+'.npz')
            assert sha(pose)==read(V4/'evidence'/(key+'.json'))['pose_hash']
            with np.load(pose) as z:
                root=body_rotation(z['locked'][21])@body_rotation(z['global0'])
            case=Case(key)
            rows[key]={}
            meshes={}
            closeups=[]
            for label,c in candidates.items():
                R=rot(c['deg'])
                t=np.array(c['translation_m'])
                target=(base-pivot)@R.T+pivot+t
                obj=OUT/(key+'_'+label.lower()+'_cam3_m.obj')
                with obj.open('w') as f:
                    f.write('# cam3 coordinates, meters\n')
                    np.savetxt(f,target,fmt='v %.9f %.9f %.9f')
                    np.savetxt(f,faces+1,fmt='f %d %d %d')
                v,topology=load_obj(obj)
                diff=float(np.max(abs(v-target)))
                assert diff<1e-8 and np.array_equal(faces,topology)
                face,errors=stereo(v,faces,case,R,t,COLORS[label])
                closeups.append(header(face,key+'    '+label+'    |    cam3 (left)                                             cam4 (right)',COLORS[label]))
                rows[key][label]=dict(obj=obj.name,sha256=sha(obj),vertices=len(v),faces=len(faces),
                    max_vertex_readback_error_m=diff,degree=c['deg'],translation_m=t.tolist(),source_mesh_sha256=sha(path),
                    pose_sha256=sha(pose),left_right_mask_disagreement_pixels=errors)
                archive.write(obj,arcname=obj.name)
                meshes[label]=v
            images[key+'_stereo.jpg']=save(key+'_stereo.jpg',np.vstack(closeups))
            union=np.vstack(list(meshes.values()))
            bases=[root,root@body_rotation([0.,np.pi/2,0.])]
            coords=[union@b for b in bases]
            scale=min(min(315/np.ptp(q[:,0]),440/np.ptp(q[:,1])) for q in coords)
            panels=[]
            for name,b,q in zip(['Front','Side'],bases,coords):
                center=(q.max(0)+q.min(0))*.5
                for label,v in meshes.items():
                    view=free_render(v,faces,b,center,scale)
                    panels.append(header(view,key+'    '+name+'    '+label,COLORS[label]))
            full.append(np.hstack(panels))
    with ZipFile(bundle) as archive:
        assert archive.testzip() is None and len(archive.namelist())==4
        for variants in rows.values():
            for item in variants.values():
                assert hashlib.sha256(archive.read(item['obj'])).hexdigest()==item['sha256']
    images['full_body.jpg']=save('full_body.jpg',np.vstack(full))
    write(RUN/'evidence/render.json',dict(base_commit='88d405716e1add5628a3538ca8a5f5d642f398c5',dirty_worktree=True,
        settings_sha256=sha(params),angle_evidence_sha256=sha(angles),rows=rows,image_sha256=images,bundle_sha256=sha(bundle),
        command='bash runs/2026-11-26_realvideo4_pitch_obj/run.sh',exit_status=0))
if __name__=='__main__':main()
