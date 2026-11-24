"""Rebuild wrist and Golden D v3 candidates, then run the frozen elbow selector."""
import cv2
from pixel_intervention import FIT, OUT, RUN, elbow, np, prev, search, torch, wrist


def make():
    torch.set_num_threads(1)
    prev.base.fit.camera_files=prev.ref.camera_files
    cfg=prev.read(FIT/'protocol.json')
    items=prev.read(FIT/'prepared.json')['items']
    base=dict(np.load(FIT/'budget_20000.npz'))
    progress=prev.read(FIT/'fit_complete.json')
    assert all(base['selected_step']>=0)
    model,camera=prev.ref.common.setup_model()
    adapters=[prev.read(p)['input_adapter'] for p in prev.ref.camera_files()]
    poses,records=[],{}
    for i,item in enumerate(items):
        name=item['pair']
        assert progress['ids'][i]==[name,'baseline','2d']
        images=[]
        for view in prev.read(OUT/'image_inputs.json')[name.split('_')[0]]:
            path=prev.VM.parent/view['image']
            assert prev.sha(path)==view['sha256']
            images.append(cv2.imread(str(path)))
        arms=wrist.frame_candidates(item,base,i,model,camera,images,adapters,cfg)
        b={f:base[f][i] for f in search.FIELDS}
        w={f:v.copy() for f,v in b.items()}
        for (s,e,_),options in zip(wrist.ARMS,arms):
            w['body_pose'][[s-1,e-1]]=min(options,key=lambda c:c['loss'])['pose']
        full,trials=search.frame(item,w,b,model,camera,cfg)
        losses=[c['loss'] for c in full]
        records[name]=dict(index=i,offset=len(poses),names=[c['name'] for c in full],
                           loss=losses,best=int(np.argmin(losses)),chain_probes=trials)
        for c in full:
            poses.append(dict(w,**{f:c[f] for f in search.FIELDS[:3]}))
    par=OUT/'validation_bank.npz'; choices=OUT/'validation_choices.json'
    np.savez_compressed(par,**{f:np.stack([c[f] for c in poses]) for f in search.FIELDS})
    prev.write(choices,records)
    prev.write(RUN/'evidence/validation_search.json',dict(parameter_sha256=prev.sha(par),choices_sha256=prev.sha(choices),
        prepared_sha256=prev.sha(FIT/'prepared.json'),protocol_sha256=prev.sha(FIT/'protocol.json'),
        fit_sha256=prev.sha(FIT/'budget_20000.npz'),fit=progress,rows=records))


if __name__=='__main__':
    with torch.no_grad():
        make()
        # Run the published selector on the freshly regenerated, same-format experiment archive.
        elbow.REF=elbow.RUN=RUN
        elbow.OUT=search.OUT=OUT
        prev.VAL=FIT
        elbow.select('validation')
