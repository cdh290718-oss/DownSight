"""Horizontal sole frames during fitting with differentiated foot/toe readouts."""
import sys
from pathlib import Path
import numpy as np
import torch
from torch.nn.functional import normalize
from floor import RUN,ROOT,OUT,trial,common,read,write,sha
from geometry import rotation
import fit

KEYS={'baseline':'free12','flat12':'flat12','free15':'free15','flat15':'flat15'}


def sole_frame(vertices,patches):
    heel,fore_a,fore_b=[vertices[a].mean(0) for a in patches]
    forward=(fore_a+fore_b)/2-heel
    forward/=np.linalg.norm(forward)
    up=np.cross(forward,fore_a-fore_b)
    up/=np.linalg.norm(up)
    return np.column_stack((np.cross(up,forward),up,forward))


def support_patches(layer):
    vs=layer.v_template.numpy()
    w=layer.lbs_weights.numpy()
    feet=[]
    for ankle,toe in [(7,10),(8,11)]:
        ids=np.flatnonzero(w[:,[ankle,toe]].sum(1)>.85)
        z=vs[ids,2]
        heel=ids[z<np.quantile(z,.25)]
        front=ids[z>np.quantile(z,.60)]
        x=vs[front,0]
        regions=[heel,front[x>np.median(x)],front[x<=np.median(x)]]
        patches=[a[np.argsort(vs[a,1])[:4]].tolist() for a in regions]
        frame=sole_frame(vs,patches)
        assert frame[1,1]>.85
        feet.append(patches)
    return feet


def prepare():
    model,cam=common.setup_model()
    cams=[cam(p) for p in common.camera_files()]
    floors={v['pair']:v['up'] for v in read(RUN/'evidence/floor.json')['rows']}
    cfg=read(trial.OUT/'protocol.json')
    cfg.update(conditions=list(KEYS),scope='free vs horizontal feet, RW20000 12/15 auto observations',sole_patches=support_patches(model.layer['neutral']))
    raw=trial.PRIOR/'automatic/prepared.json'
    items=[];audit=[]
    for row in read(raw)['items']:
        count=12 if row['condition']=='baseline' else 15
        assert row['condition'] in ['baseline','15history']
        points=[dict({k:p[k] for k in ['name','index','native_index','group','xy','valid']},accepted=False) for p in row['points']]
        assert sum(any(p['valid']) for p in points)==count
        seed,reason=trial.body_seed(row,cams)
        assert reason is None
        for lock in [False,True]:
            mode=('flat' if lock else 'free')+str(count)
            item=dict(pair=row['pair'],inputs=row['inputs'],points=points,head_position=row['anchor']['position'],
                initial_body_rotation=seed.tolist(),condition=next(k for k,v in KEYS.items() if v==mode),
                flat=lock,floor_up=floors[row['pair']])
            items.append(item)
            audit.append(dict(pair=row['pair'],mode=mode,points=count,shoulder_hip_fallback=reason))
    assert len(items)==16
    for pair in floors:
        rows=[r for r in items if r['pair']==pair]
        assert all(r['initial_body_rotation']==rows[0]['initial_body_rotation'] and r['head_position']==rows[0]['head_position'] for r in rows)
        assert all(r['points'][:12]==rows[0]['points'][:12] for r in rows)
    write(OUT/'protocol.json',cfg);write(OUT/'prepared.json',dict(items=items))
    write(RUN/'evidence/inputs.json',dict(rows=audit,source_rw20000_sha256=sha(raw),
          prepared_sha256=sha(OUT/'prepared.json'),protocol_sha256=sha(OUT/'protocol.json'),
          neutral_sole_vertex_patches=cfg['sole_patches']))


def world_ankles(body,root,delta):
    parent=rotation(delta)@rotation(root)
    knees=parent[:,None]@rotation(body[:,[0,1]])@rotation(body[:,[3,4]])
    return knees,knees@rotation(body[:,[6,7]])


def flat_targets(world,frames,normals):
    forward=(world@frames)[...,2]
    up=normals[:,None,:].expand_as(forward)
    direction=forward-(forward*up).sum(-1,keepdim=True)*up
    # Only relevant for near-vertical toe headings; preserve lateral heading there.
    alt=torch.cross((world@frames)[...,0],up,dim=-1)
    direction=normalize(torch.where(direction.norm(dim=-1,keepdim=True)>.02,direction,alt),dim=-1)
    return torch.stack([torch.cross(up,direction,dim=-1),up,direction],dim=-1)@frames.transpose(-1,-2)


class FlatProblem(trial.TorsoOnly):
    def __init__(self,prepared,cfg,model,camera_cls):
        super().__init__(prepared,cfg,model,camera_cls)
        self.lock=torch.tensor([r['flat'] for r in self.items])[:,None,None]
        self.normals=self.t([r['floor_up'] for r in self.items])
        self.frames=self.t(np.stack([sole_frame(self.layer.v_template.numpy(),p) for p in cfg['sole_patches']]))
        self.offset=self.fk.rest[[10,11]]-self.fk.rest[[7,8]]

    def forward(self):
        native,rh,beta=fit_base_forward(self)
        body,_=self.parameters()
        _,world=world_ankles(body,self.global0,self.delta)
        target=flat_targets(world,self.frames,self.normals)
        foot=native[:,[7,8]]+(target@self.offset[...,None])[...,0]
        # Only forefoot descendants change when ankle orientation is replaced.
        joints=torch.cat([native[:,:10],torch.where(self.lock,foot,native[:,10:12]),native[:,12:]],dim=1)
        virtual=[joints[:,a]+k*(joints[:,b]-joints[:,a]) for a,b,k in zip([15,8,7],[12,11,10],self.ratios)]
        return torch.cat([joints,torch.stack(virtual,dim=1)],dim=1),rh,beta

fit_base_forward=fit.Problem.forward

if __name__=='__main__':
    prepare()
    fit.RUN=fit.OUT=OUT
    fit.Problem=FlatProblem
    sys.argv=[__file__,'--threads','1']
    fit.main()
