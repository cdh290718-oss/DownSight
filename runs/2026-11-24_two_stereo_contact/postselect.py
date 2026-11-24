"""Apply the same frame objective and hard gate to composed v3 limb/trunk candidates."""
import sys
import numpy as np
import torch
from common_run import EV,OUT,prev,search,inputs
from fit_probe import Frame,WEIGHT


def main(pair):
    from pathlib import Path
    def guard(event,args):
        if event=='open' and args and isinstance(args[0],str):
            if Path(args[0]).name in ('reference.json','validation_truth.json','samples.npz','test_manifest.json'):
                raise PermissionError('Body evaluation labels are excluded during selection')
    sys.addaudithook(guard)
    torch.set_num_threads(1)
    model,cam,cfg,data=inputs()
    item,orig=data[pair]
    frame=Frame(pair,item,orig,model,cam,cfg)
    early=prev.read(EV/(pair+'_selection.json'))
    path=OUT/(pair+'_states.npz')
    assert prev.sha(path)==early['state_sha256']
    archive=dict(np.load(search.OUT/'test_bank.npz'))
    manifest=prev.read(search.OUT/'test_choices.json')[pair]
    states=dict(np.load(path))
    candidates={mode+'_init':torch.from_numpy(state) for mode,state in states.items()}
    for i,tag in enumerate(manifest['names']):
        pos=manifest['offset']+i
        xx=frame.v3.clone()
        for idx,key in [(slice(21),'body_pose'),(21,'delta'),(22,'translation')]:
            xx[idx]=torch.from_numpy(archive[key][pos].reshape(xx[idx].shape))
        candidates['archive_'+tag]=xx
    for mode in ['outline','hard','joint']:
        x=candidates[mode+'_init']
        seed=dict(orig,body_pose=x[:21].numpy(),delta=x[21].numpy(),translation=x[22].numpy())
        with torch.no_grad():
            branches,_=search.frame(item,seed,seed,model,cam,cfg)
        for branch in branches:
            xx=x.clone()
            for idx,key in [(slice(21),'body_pose'),(21,'delta'),(22,'translation')]:
                xx[idx]=torch.tensor(branch[key].reshape(xx[idx].shape))
            candidates[mode+'_refit_'+branch['name']]=xx
    repairfile=OUT/(pair+'_repair.npz')
    assert prev.read(EV/(pair+'_repair.json'))['parameter_sha256']==prev.sha(repairfile)
    for name,state in np.load(repairfile).items():
        candidates['repair_'+name]=torch.from_numpy(state)
    if pair=='pair068':
        reach_file=OUT/(pair+'_reach.npz')
        assert prev.read(EV/'reach.json')['parameter_sha256']==prev.sha(reach_file)
        for name,state in np.load(reach_file).items():
            candidates['reach_'+name]=torch.from_numpy(state)
    unique={}
    trials={}
    chosen={}
    rows={}
    with torch.no_grad():
        for tag,trial in candidates.items():
            key=trial.numpy().tobytes()
            if key in unique:continue
            unique[key]=tag
            v,_,base,positive=frame.state(trial)
            hits,active=frame.gate.check(v.numpy())
            outline=frame.observed(v)[0].item()
            observed_gap=None
            if positive and len(active)==0:
                observed_gap=frame.contact.measure(v.numpy())[frame.target][1]
            trials[tag]=dict(base=base.item(),objective=base.item()+WEIGHT*outline,
                             raw=len(hits),outside_template_zones=len(active),positive=positive,
                             near_gap_capped_mm=observed_gap)
        no_raw=[tag for tag,r in trials.items() if r['positive'] and r['raw']==0]
        for mode in ['v3','outline','hard','joint','joint_near']:
            if mode=='v3':
                best='v3_init'
            else:
                allowed=[tag for tag,r in trials.items() if r['positive'] and
                    (mode=='outline' or r['outside_template_zones']==0) and
                    (mode!='joint_near' or r['near_gap_capped_mm']<=8.)]
                if not allowed:
                    rows[mode]=dict(chosen=None)
                    continue
                best=min(allowed,key=lambda t:trials[t]['base' if mode=='hard' else 'objective'])
            chosen[mode]=candidates[best].numpy()
            rows[mode]=dict(chosen=best,result=frame.describe(candidates[best]))
    path=OUT/(pair+'_final.npz')
    np.savez_compressed(path,**chosen)
    prev.write(EV/(pair+'_final.json'),dict(rows=rows,trials=trials,raw_feasible_candidates=no_raw,final_sha256=prev.sha(path),exit_status=0))
    print(pair,{m:r['chosen'] for m,r in rows.items()},flush=True)


if __name__=='__main__':main(sys.argv[1])
