"""Post-hoc diagnostic only. Score already generated oracle-input candidates on detector pixels."""
from pixel_intervention import CASES, FIT, OUT, RUN, key, np, prev, torch


def main():
    params=dict(np.load(OUT/'validation_bank.npz'))
    choices=prev.read(OUT/'validation_choices.json')
    cfg=prev.read(FIT/'protocol.json')
    items={r['pair']:r for r in prev.read(FIT/'prepared.json')['items']}
    model,cam=prev.ref.common.setup_model()
    prev.base.fit.camera_files=prev.ref.camera_files
    result={}
    for frame in CASES:
        name=key(frame,'detector')
        problem=prev.base.trial.TorsoOnly([items[name]],cfg,model,cam)
        native=[p['native_index'] for p in items[name]['points']]
        gt=np.array(prev.read(prev.OUT/'validation_truth.json')[frame]['gt'])[prev.ref.common.OBS]
        dst=result[frame]={}
        for mode in ['detector','projected_wrists','projected_all']:
            r=choices[key(frame,mode)]
            ids=np.arange(r['offset'],r['offset']+len(r['names']))
            b,beta,root,d,t=[torch.tensor(params[k][ids]).float() for k in ['body_pose','betas','global0','delta','translation']]
            j,h=problem.fk(b,beta,root,d,t)
            problem.bodyvar=b[:,problem.body_ids]
            loss,valid,terms=problem.objective((j,h,beta))
            assert valid.any()
            best=int(torch.where(valid,loss,torch.full_like(loss,float('inf'))).argmin())
            if mode=='detector':
                assert best==r['best'] and abs(float(loss[best])-r['loss'][best])<2e-6
            points=j.numpy()[:,native]
            dst[mode]={}
            for lbl,i in [('selected_for_own_inputs',r['best']),('lowest_detector_loss',best)]:
                dst[mode][lbl]=dict(feasible=bool(valid[i]),detector_loss=float(loss[i]),
                    non_pa_mm=float(1000*np.linalg.norm(points[i]-gt,axis=1).mean()),name=r['names'][i])
    prev.write(RUN/'evidence/rescore.json',result)

if __name__=='__main__':
    torch.set_num_threads(1)
    with torch.no_grad():
        main()
