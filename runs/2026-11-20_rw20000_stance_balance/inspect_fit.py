"""Evaluate frozen fits against independently loaded labels and render actual OBJ surfaces."""
import hashlib
import io
import json
import zipfile
import cv2
import numpy as np
from scipy.spatial import ConvexHull
from balance import g, RUN, ROOT, OUT, MAIN, AMB, center, support, axes

ARCHIVE='RW20000_stance_balance_OBJ.zip'
TAG='rw20000-single-pair-stance-balance-20261120'


def unit(x):
    return x/np.linalg.norm(x)


def angle(x,y):
    return float(np.rad2deg(np.arccos(np.clip(unit(x)@unit(y),-1,1))))


def draw_pair(vertices,joints,faces,basis,n,d,contacts,margins,labels):
    side=basis[:,[2,1,0]].copy()
    side[:,2]*=-1
    all_q=np.concatenate(vertices)@side
    mid=(all_q.max(0)+all_q.min(0))/2
    scale=min(300/np.ptp(all_q[:,0]),405/np.ptp(all_q[:,1]))
    plane_xy=axes(n)
    clouds=[v[ix]@plane_xy for v,ix in zip(vertices,contacts)]
    mass=[center(j)@plane_xy for j in joints]
    both=np.vstack(clouds+mass)
    offset=(both.max(0)+both.min(0))/2
    patch_scale=min(280/max(.1,np.ptp(both[:,0])),320/max(.1,np.ptp(both[:,1])))
    panels=[]
    for v,j,title in zip(vertices,joints,labels):
        panel=g.free_render(v,faces,side,mid,scale)
        c=center(j)@side
        xy=np.rint((c[:2]-mid[:2])*[scale,-scale]+[175,240]).astype(int)
        floor=int(round(240+scale*(d+mid[1])))
        cv2.line(panel,(0,floor),(349,floor),(65,140,65),2)
        cv2.line(panel,tuple(xy),(xy[0],floor),(35,100,240),2)
        cv2.circle(panel,tuple(xy),5,(35,100,240),-1)
        panels.append(panel)
    top=np.full((480,350,3),248,np.uint8)
    def xy(a):return np.rint((a-offset)*[patch_scale,-patch_scale]+[175,210]).astype(np.int32)
    for pts,c,color in zip(clouds,mass,[(130,115,95),(30,95,225)]):
        cv2.polylines(top,[xy(pts[ConvexHull(pts).vertices])],True,color,2,cv2.LINE_AA)
        cv2.drawMarker(top,tuple(xy(c)),color,cv2.MARKER_CROSS,14,2)
    for k,(tag,m,col) in enumerate(zip(labels,margins,[(130,115,95),(30,95,225)])):
        cv2.putText(top,tag+'  %+.1f mm'%m,(10,425+24*k),0,.57,col,1,cv2.LINE_AA)
    bar=np.full((38,1050,3),250,np.uint8)
    for x,s in zip([10,360,710],[labels[0]+' | side',labels[1]+' | side','Top: support hull + COM']):
        cv2.putText(bar,s,(x,26),0,.62,(45,45,45),1,cv2.LINE_AA)
    return np.vstack([bar,np.hstack(panels+[top])])


def main():
    model,cam=g.common.setup_model()
    layer=model.layer['neutral'].cpu().eval()
    faces=np.asarray(model.face,np.int32)
    skin=layer.lbs_weights.detach().numpy()
    rest=layer.v_template.detach().numpy()
    left_right=np.full(len(rest),-1,int)
    for k,idx in enumerate([[7,10],[8,11]]):
        ix=np.flatnonzero(skin[:,idx].sum(1)>.8)
        left_right[ix[rest[ix,1]<rest[ix,1].min()+.034]]=k
    sole=np.flatnonzero(left_right>=0)
    head=(skin.argmax(1)==15).astype(np.float32)
    inputs=g.read(g.SOURCE/'prepared.json')['items']
    done=g.read(g.SOURCE/'fit_complete.json')['ids']
    truth=g.read(g.DATA/'reference.json')
    images={r['id']:r for r in g.read(g.MANIFEST)['records']}
    entries={r['pair']:r for r in g.read(ROOT/'runs/2026-11-18_rw_frozen200_golden_d/evidence/preparation.json')['pairs']}
    published={r['pair']:r for r in g.read(g.REF/'results.json')['rows'] if r['model']=='rw20000'}
    maprows=g.read(ROOT/'runs/2026-11-16_sapiens-unrealego-audit/evidence/audit.json')['joint_mapping']
    det_id={m['rw_index']:m['sapiens308_id'] for m in maprows}
    det=[det_id[q] for q in g.common.OBS]
    modes={'soft':('','Soft'),'strong':('_strong','Strong'),'contact':('_contact','Contact')}
    ref=g.read(ROOT/'runs/2026-11-20_rw20000_floor3/evidence/planes.json')['rows']
    floor=next(x for x in ref if x['pair']=='pair078')
    src=np.load(g.SOURCE/'budget_20000.npz')
    camera_files=[g.DATA/('camera_'+s+'.json') for s in ['left','right']]
    cams=[cam(p) for p in camera_files]
    adapters=[g.read(p)['input_adapter'] for p in camera_files]
    yy,xx=np.indices((872,872))
    px=np.stack([xx.ravel(),yy.ravel()],1)
    rays=[c.camera2world_ray(px*a['scale']+a['offset']).reshape(872,872,3).astype(np.float32) for c,a in zip(cams,adapters)]
    raster=g.run_path(str(ROOT/'runs/2026-11-04_mesh-projection-review/code/rasterizer.py'))['raster']
    rows=[]
    members={}
    zpath=OUT/ARCHIVE
    with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED) as archive:
        for mode,(suffix,label) in modes.items():
            evidence=g.read(RUN/'evidence'/('balance'+suffix+'.json'))
            assert g.sha(OUT/('adjusted'+suffix+'.npz'))==evidence['adjusted_sha256']
            result=np.load(OUT/('adjusted'+suffix+'.npz'))
            for k,r in enumerate(evidence['rows']):
                p=r['pair'];e=entries[p];i=done.index([p,'rw20000','2d']);item=inputs[i]
                n=np.array(r['candidate_up']);d=r['plane_offset_m']
                old={t:src[u][i] for t,u in [('global_orient','global0'),('body_pose','body_pose'),('betas','betas'),('transl','translation')]}
                new={t:result[u][k] for t,u in [('global_orient','root'),('body_pose','body'),('transl','translation')]}
                new['betas']=old['betas']
                meshes=[];joints=[];contacts=[];metrics=[];hashes={}
                gt=np.asarray(truth[p]['gt'])[g.common.OBS]
                fwd=(g.rot(src['delta'][i])@g.rot(src['global0'][i]))[:,2]
                x=unit(np.cross(n,fwd));basis=np.stack([x,n,np.cross(x,n)],axis=1)
                for t,(pars,delta) in enumerate([(old,src['delta'][i]),(new,result['delta'][k])]):
                    v,j,h,c=g.geometry(layer,pars,delta,False,None)
                    assert np.allclose(h,c)
                    meshes.append(v);joints.append(j)
                    ix,_=support(v,sole,left_right,n,d)
                    contacts.append(ix)
                    native=j[g.common.NATIVE_IDS][g.common.OBS]
                    errors=[np.linalg.norm(pos-gt,axis=1).mean()*1000 for pos in [native,g.align(native,gt)]]
                    if not t:
                        assert np.allclose(errors,[published[p]['mpjpe12_mm'],published[p]['pa12_mm']],atol=.001,rtol=0)
                    views=[]
                    for eye,side in enumerate(['left','right']):
                        m=images[e['image_ids'][eye]];a=adapters[eye]
                        T=np.asarray(item['inputs']['views'][side]['camera']['left_to_view'])
                        def project(raw):return (cams[eye].world2camera(raw)-a['offset'])/a['scale']
                        pred=project(native@T[:3,:3].T+T[:3,3])
                        valid=np.array([q['valid'][eye] for q in item['points']],bool)
                        observed=(np.array([q['xy'][eye] for q in item['points']])-a['offset'])/a['scale']
                        scores=[np.square((pred-y)[valid]).sum() for y in [np.asarray(m['keypoints308_xy'])[det],observed]]
                        views.append(dict(n=int(valid.sum()),gt_sq=scores[0],det_sq=scores[1]))
                        if not t:
                            old_v=published[p]['views'][eye]
                            for val,name in zip(scores,['fit_vs_gt','fit_vs_detection']):
                                assert abs(np.sqrt(val/valid.sum())-np.sqrt(old_v['sum_sq_px2'][name]/old_v['n']))<.001
                        if t:
                            source=g.MANIFEST.parent/m['image']
                            assert g.sha(source)==m['image_sha256']==e['image_sha256'][eye]
                            im=cv2.imread(str(source))
                            vv=(v@T[:3,:3].T+T[:3,3]).astype(np.float32)
                            depth=np.full(im.shape[:2],np.inf,np.float32);color=np.zeros_like(im)
                            raster(vv,faces,head,project(vv.astype(float)).astype(np.float32),rays[eye],depth,color,np.zeros_like(depth),180/a['scale'])
                            mask=np.isfinite(depth);assert mask.any()
                            im[mask]=.55*im[mask]+.45*color[mask]
                            path=RUN/'assets'/p/(mode+'_'+side+'.jpg')
                            g.save(path,im);hashes[path.name]=g.sha(path)
                    metrics.append(dict(nonpa=errors[0],pa=errors[1],views=views))
                path=RUN/'assets'/p/(mode+'_comparison.jpg')
                g.save(path,draw_pair(meshes,joints,faces,basis,n,d,contacts,[r['before']['margin_mm'],r['after']['margin_mm']],['Baseline',label]))
                hashes[path.name]=g.sha(path)
                world=(v+d*n)@basis
                assert np.max(np.abs(world@basis.T-d*n-v))<1e-6 and np.linalg.det(basis)>.99999
                for frame,verts in [('left_camera',v),('support_Y_up',world)]:
                    key='/'.join([r['group'],mode,p,frame+'.obj'])
                    stream=io.StringIO()
                    stream.write('# '+p+' '+frame+' meters\n')
                    np.savetxt(stream,verts,fmt='v %.9f %.9f %.9f')
                    np.savetxt(stream,faces+1,fmt='f %d %d %d')
                    raw=stream.getvalue().encode()
                    archive.writestr(key,raw);members[key]=hashlib.sha256(raw).hexdigest()
                a=dict(pair=p,group=r['group'],subject=e['subject'],mode=mode,images=hashes,before=metrics[0],after=metrics[1],
                       head_shift_mm=float(np.linalg.norm(joints[0][15]-j[15])*1000),up_basis=basis.tolist(),source_image_sha256=e['image_sha256'])
                if p=='pair078':
                    a['prior_floor_direction_difference_deg']=angle(n,np.array(floor['up_left_cv']))
                    a['trunk_angle_to_carpet_up_deg']=[angle((js[16]+js[17]-js[1]-js[2])/2,np.array(floor['up_left_cv'])) for js in joints]
                rows.append(a)
                print(mode,p,flush=True)
        archive.writestr('manifest.json',json.dumps(members,indent=2))
        archive.writestr('README.txt','All units meters. Same adjusted mesh in left-camera CV and hypothesized Y-up support frames. Standing: soft, strong and contact-preserving strong fits. Uncertain: soft fit only; contact state unverified. No measured gravity.\n')
    with zipfile.ZipFile(zpath) as archive:
        assert archive.testzip() is None and len(members)==32 and all(hashlib.sha256(archive.read(key)).hexdigest()==val for key,val in members.items())
    totals={}
    for group,mode in [('standing','soft'),('standing','strong'),('standing','contact'),('uncertain','soft')]:
        selected=[r for r in rows if (r['group'],r['mode'])==(group,mode)]
        totals[group+'_'+mode]={}
        for stage in ['before','after']:
            values=[r[stage] for r in selected]
            views=[v for a in values for v in a['views']]
            totals[group+'_'+mode][stage]=dict(pa=np.mean([a['pa'] for a in values]),nonpa=np.mean([a['nonpa'] for a in values]),
                **{k:np.sqrt(sum(v[k] for v in views)/sum(v['n'] for v in views)) for k in ['gt_sq','det_sq']})
    g.write(RUN/'evidence/renders_all.json',dict(rows=rows,groups=totals,command='<BODY_PYTHON> -Bu inspect_fit.py',exit_status=0))
    g.write(RUN/'evidence/attachment_audit.json',dict(archive=ARCHIVE,tag=TAG,files_sha256={ARCHIVE:g.sha(zpath)},bytes=zpath.stat().st_size,objects=len(members),object_hashes=members,jpeg=sum(len(r['images']) for r in rows),exit_status=0))


if __name__=='__main__':main()
