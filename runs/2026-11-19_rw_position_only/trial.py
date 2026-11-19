"""Fit a torso-locked model with a position-only head observation."""
from pathlib import Path
import sys
import cv2
import numpy as np
import torch
RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[1]
PREV=ROOT/'runs/2026-11-19_rw_landmarks_head_views'
sys.path.insert(0,str(PREV))
from study import common,read,write,sha
import fit
OUT=ROOT/'outputs/rw_position_only_20261119'
PRIOR=ROOT/'outputs/rw_landmarks_head_views_20261119'
# Model up is -Z in a downward-looking camera, model left is +X.
FIXED=np.array([[1.,0.,0.],[0.,0.,1.],[0.,-1.,0.]])
LABELS={'baseline':'auto_fixed','auto_body':'auto_body','gt_fixed':'gt_fixed','gt_body':'gt_body'}


def body_seed(item,cams):
    origins=[]
    rotations=[]
    for side in ['left','right']:
        t=np.asarray(item['inputs']['views'][side]['camera']['left_to_view'])
        rotations.append(t[:3,:3].T)
        origins.append(-rotations[-1]@t[:3,3])
    joints={}
    for point in item['points']:
        if point['name'] not in ['L_Shoulder','R_Shoulder','L_Hip','R_Hip']:
            continue
        if not all(point['valid']):
            return FIXED.copy(),'missing bilateral landmarks'
        rays=[r@cam.camera2world_ray(np.array([xy]))[0] for xy,r,cam in zip(point['xy'],rotations,cams)]
        rays=[r/np.linalg.norm(r) for r in rays]
        m=np.column_stack([rays[0],-rays[1]])
        distances,_,rank,_=np.linalg.lstsq(m,origins[1]-origins[0],rcond=None)
        if rank<2 or np.linalg.cond(m)>150 or min(distances)<=0:
            return FIXED.copy(),'unstable stereo rays'
        joints[point['name']]=np.mean([o+r*d for o,r,d in zip(origins,rays,distances)],axis=0)
    ls,rs,lh,rh=[joints[k] for k in ['L_Shoulder','R_Shoulder','L_Hip','R_Hip']]
    up=(ls+rs-lh-rh)/2
    if np.linalg.norm(up)<1e-4:
        return FIXED.copy(),'degenerate up'
    up/=np.linalg.norm(up)
    lat=(ls-rs+lh-rh)/2
    lat-=up*(lat@up)
    if np.linalg.norm(lat)<1e-4:
        return FIXED.copy(),'degenerate lateral'
    lat/=np.linalg.norm(lat)
    return np.column_stack([lat,up,np.cross(lat,up)]),None


def prepare():
    _,cam_cls=common.setup_model()
    cams=[cam_cls(p) for p in common.camera_files()]
    cfg=read(PRIOR/'protocol.json')
    cfg.pop('camera_R_head_proxy')
    cfg.update(conditions=list(LABELS),head_rotation_weight=0.,scope='Position-only head; rigid head relative to upper chest')
    items=[]
    audit=[]
    inputs={}
    for source,path,variant in [('auto',PRIOR/'automatic/prepared.json','baseline'),('gt',PRIOR/'prepared.json','15history')]:
        inputs[source]=sha(path)
        for item in read(path)['items']:
            if item['condition']!=variant:
                continue
            points=[{k:p[k] for k in ['name','index','native_index','group','xy','valid']} for p in item['points']]
            assert not any(p['accepted'] for p in item['points'])
            for p in points:
                p['accepted']=False
            position=item['anchor']['position']
            est,reason=body_seed(item,cams)
            for mode,seed in [('fixed',FIXED),('body',est)]:
                name=source+'_'+mode
                assert np.allclose(seed.T@seed,np.eye(3)) and np.isclose(np.linalg.det(seed),1)
                minimal=dict(pair=item['pair'],points=points,inputs=item['inputs'],head_position=position,
                    initial_body_rotation=seed.tolist(),condition='baseline' if name=='auto_fixed' else name)
                items.append(minimal)
                audit.append(dict(pair=item['pair'],mode=name,active_points=sum(any(p['valid']) for p in points),
                    initial_body_rotation=seed.tolist(),fallback=reason if mode=='body' else None))
    assert len(items)==16
    write(OUT/'prepared.json',dict(items=items))
    write(OUT/'protocol.json',cfg)
    write(RUN/'evidence/inputs.json',dict(seeds=audit,input_sha256=inputs,
          solver_input_sha256=sha(OUT/'prepared.json'),protocol_sha256=sha(OUT/'protocol.json'),
          head_fields=['head_position'],fixed_convention='model up = left-camera -Z, model left = left-camera +X',
          masks='Inherited in-frame masks; four S9 pairs; no facial, hand or orientation measurements.'))


class TorsoOnly(fit.Problem):
    def __init__(self,prepared,cfg,model,camera_cls):
        # The legacy constructor takes root initialization in its anchor.rotation slot.
        # Adapt our body seeds here; the public solver input has no head rotation.
        assert cfg['head_rotation_weight']==0 and all('anchor' not in r for r in prepared)
        rows=[dict(r,anchor=dict(position=r['head_position'],rotation=r['initial_body_rotation'])) for r in prepared]
        super().__init__(rows,cfg,model,camera_cls)
        assert all(r['rank']==0 for r in self.initialization.values())
        self.mask=self.t(np.ones((1,21,1)))
        self.mask[:,[11,14]]=0
        self.ratios=cfg['landmark_ratios']

    def parameters(self):
        body,beta=super().parameters()
        return body*self.mask,beta

    def forward(self):
        native,rh,beta=super().forward()
        virtual=[native[:,a]+k*(native[:,b]-native[:,a])
                 for a,b,k in zip([15,8,7],[12,11,10],self.ratios)]
        return torch.cat([native,torch.stack(virtual,dim=1)],dim=1),rh,beta

if __name__=='__main__':
    if sys.argv[1]=='prepare':
        prepare()
    else:
        assert sys.argv[1]=='fit'
        fit.RUN=fit.OUT=OUT
        fit.Problem=TorsoOnly
        sys.argv=[__file__,'--threads','1']
        fit.main()
