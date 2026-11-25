"""Score frozen selections and stratify direction changes by arm-local pixel error."""
import numpy as np
import torch
from refine import RUN,OUT,prev,joint,BRANCHES
from forearm_seed import angle

def main():
    torch.set_num_threads(1)
    selection=prev.read(RUN/'evidence/selection.json')
    path=OUT/'checkpoints.npz'
    assert prev.sha(path)==selection['checkpoints_sha256']
    pack=dict(np.load(path)); stored=dict(np.load(joint.OUT/'test_bank.npz'))
    choices=prev.read(joint.OUT/'test_choices.json')
    ref=prev.ref
    gtfile=ref.SOURCE/'reference.json'
    old=prev.read(joint.RUN/'evidence/test_results.json')
    assert prev.sha(gtfile)==old['truth_sha256']
    truth=prev.read(gtfile)
    pix=prev.read(RUN/'evidence/pixels.json')['cases']
    layer=ref.common.setup_model()[0].layer['neutral']
    obs=ref.common.OBS; names=[ref.common.NAMES[i] for i in obs]
    native=np.array(ref.common.NATIVE_IDS)[obs]
    records={}; arms=[]; scores=[]
    winners={}
    for n,(f,sel) in enumerate(sorted(selection['frames'].items())):
        gt=np.array(truth[f]['gt'])[obs]
        def pose(tag):
            if tag=='archive':
                q=choices[f]
                return ref.full_joints(stored,layer,q['offset']+q['best'])[native]
            return ref.full_joints(pack,layer,n*len(BRANCHES)+BRANCHES.index(tag))[native]
        baseline=min(['archive','old'],key=lambda t:sel['losses'][t])
        tags=[baseline,sel['winner']]
        winners[sel['winner']]=winners.get(sel['winner'],0)+1
        poses=[pose(t) for t in tags]
        scores.append([[1000*np.linalg.norm(x-gt,axis=1).mean(),1000*np.linalg.norm(prev.align(x,gt)-gt,axis=1).mean()] for x in poses])
        row=records[f]=dict(control=baseline,selected=sel['winner'],scores=scores[-1],arms={})
        for side,tag in [('left','L'),('right','R')]:
            e,w=[names.index(tag+'_'+j) for j in ['Elbow','Wrist']]
            direction=gt[w]-gt[e]
            angles=[angle(x[w]-x[e],direction) for x in poses]
            error=sum([pix[f]['errors'][tag+'_'+j] for j in ['Elbow','Wrist']],[])
            # Eligibility and band use only valid arm observations. Visibility for a new
            # seed requires both eyes for each of its two joint endpoints.
            diag=sel['triangulation'][side]
            if None in error or not diag['available']:
                row['arms'][side]=dict(eligible=False,angles=angles)
                continue
            peak=max(error); triang=angle(diag['direction'],direction)
            arms.append([peak,*angles,triang])
            row['arms'][side]=dict(eligible=True,peak_px=peak,angles=angles,triangulated_deg=triang)
    arms=np.array(arms)
    def summarize(rows):
        a,b=rows[:,1],rows[:,2]
        change=b-a
        return dict(n=len(a),control_deg=float(a.mean()),selected_deg=float(b.mean()),triangulated_deg=float(rows[:,3].mean()),
                    improvement_over_1deg=int((change < -1).sum()),worse_over_1deg=int((change>1).sum()),
                    within_1deg=int((abs(change)<=1).sum()),median_delta=float(np.median(change)))
    x=arms[:,0]
    groups={k:summarize(arms[mask]) for k,mask in [('all',x>=0),('lt15',x<15),('15to25',(x>=15)&(x<=25)),('gt25',x>25)]}
    assert sum(g['n'] for k,g in groups.items() if k!='all')==len(arms)
    prev.write(RUN/'evidence/refinement_scores.json',dict(groups=groups,body_mean=np.mean(scores,axis=0).tolist(),
                       winners=winners,records=records,truth_sha256=prev.sha(gtfile),exit_status=0))
if __name__=='__main__':
    main()
