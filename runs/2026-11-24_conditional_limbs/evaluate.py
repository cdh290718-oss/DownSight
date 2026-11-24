"""Score locked poses against the official pixels and 3D skeletons."""
import sys
import numpy as np
import torch
from select_pose import OUT, REF, ROOT, RUN, prev, search
from score_candidates import summarize


def degree(a,b):
    cos=(a*b).sum(-1)/(np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1))
    return np.degrees(np.arccos(np.clip(cos,-1,1)))


def evaluate(split):
    torch.set_num_threads(1)
    decisions=prev.read(OUT/(split+'_decisions.json'))
    sig=prev.read(RUN/('evidence/'+split+'_selection.json'))
    selected=OUT/(split+'_selected.npz')
    assert prev.sha(selected)==sig['selected_sha256']
    bank=dict(np.load(search.OUT/(split+'_bank.npz')))
    poses=dict(np.load(selected))
    choices=prev.read(search.OUT/(split+'_choices.json'))
    manifest, folder, target=((prev.TM,prev.base.OUT,prev.ref.SOURCE/'reference.json') if split=='test' else
                              (prev.VM,prev.VAL,prev.OUT/'validation_truth.json'))
    archive=prev.read(REF/('evidence/'+split+'_results.json'))
    assert prev.sha(target)==archive['truth_sha256'] and prev.sha(manifest)==archive['manifest_sha256']
    truth=prev.read(target); images=prev.pair_images(manifest)
    items=prev.read(folder/'prepared.json')['items']
    model,cam=prev.ref.common.setup_model()
    cameras=[cam(p) for p in prev.ref.camera_files()]
    adapters=[prev.read(p)['input_adapter'] for p in prev.ref.camera_files()]
    obs=prev.ref.common.OBS
    ids=np.array(prev.ref.common.NATIVE_IDS)[obs]
    names=[prev.ref.common.NAMES[k] for k in obs]
    subjects=set(prev.read(RUN/'evidence/training.json')['people'])
    groups=['v3','prior']
    rows={}
    for i,(key,sel) in enumerate(decisions.items()):
        r=choices[key]
        target=np.asarray(truth[key]['gt'])[obs]
        item=items[r['index']]
        im=images[int(key[-3:])-1]
        assert im[0]['subject']==truth[key]['subject'] and truth[key]['subject'] not in subjects and item['pair']==key
        row=rows[key] = dict(subject=truth[key]['subject'])
        for tag,pack,idx in [('v3',bank,r['offset']+r['best']),('prior',poses,i)]:
            x=prev.ref.full_joints(pack,model.layer['neutral'],idx)[ids]
            views=prev.ref.pixel_errors(item,x,im,prev.mapping(),cameras,adapters)
            direction=[]
            for side,j,k in [('L','Elbow','Wrist'),('R','Elbow','Wrist'),('L','Knee','Ankle'),('R','Knee','Ankle')]:
                a,b=names.index(side+'_'+j),names.index(side+'_'+k)
                direction.append(float(degree(x[b]-x[a],target[b]-target[a])))
            row[tag]=dict(mm=(1000*np.linalg.norm(x-target,axis=1)).tolist(),
                          pa_mm=(1000*np.linalg.norm(prev.align(x,target)-target,axis=1)).tolist(),
                          n_px=sum(v['n'] for v in views),px_sq=sum(v['sum_sq_px2']['fit_vs_gt'] for v in views),
                          direction_deg=direction)
        assert abs(np.mean(row['v3']['mm'])-archive['rows'][key]['scores']['joint']['mpjpe'])<.001
    def stats(keys):
        return {tag:dict(summarize([rows[k][tag] for k in keys]),
                         direction_deg=np.mean([rows[k][tag]['direction_deg'] for k in keys],axis=0).tolist()) for tag in groups}
    summary=stats(rows)
    by_subject={s:stats([k for k,v in rows.items() if v['subject']==s]) for s in sorted({v['subject'] for v in rows.values()})}
    deltas=np.array([np.mean(r['prior']['mm'])-np.mean(r['v3']['mm']) for r in rows.values()])
    compact={k:{t:dict(summarize([r[t]]),direction_deg=r[t]['direction_deg']) for t in groups} for k,r in rows.items()}
    out=dict(summary=summary,subjects=by_subject,rows=compact,
             improved=int(sum(deltas<-.01)),worse=int(sum(deltas>.01)),ties=int(sum(abs(deltas)<=.01)),exit_status=0)
    prev.write(RUN/('evidence/'+split+'_scores.json'),out)
    print(split,summary,out['improved'],out['worse'],out['ties'],flush=True)


if __name__=='__main__':
    evaluate(sys.argv[1])
