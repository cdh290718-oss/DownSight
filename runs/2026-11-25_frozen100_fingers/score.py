"""Score sealed counterfactuals against RW body truth; use the unmasked contours for comparison."""
from pathlib import Path
import numpy as np
import torch
from geometry_inputs import EV,OUT,PREV,PRIOR,prev,Frame,GridEye,fit_probe
from common_run import rotation,summary
from contact import PAIRS
from inspect_obj import self_intersections

NEW=('fixed_full','palm','no_hand','free_palm')
ALL=('v3','matched','outline')+NEW

def aggregate(rows,keys):
    result={}
    for m in ALL:
        values=[rows[k]['methods'][m] for k in keys]
        avg={name:float(np.mean([r[name] for r in values])) for name in ['pa_mm','nonpa_mm','full_outline_px','outside_triangles']}
        result[m]=dict(avg,frames=len(keys),self_frames=sum(r['left_self']+r['right_self']>0 for r in values),
                       left_self_frames=sum(r['left_self']>0 for r in values),right_self_frames=sum(r['right_self']>0 for r in values),
                       target_frames=sum(sum(r['target_counts'].values())>0 for r in values),
                       left_self_faces=sum(r['left_self'] for r in values),right_self_faces=sum(r['right_self'] for r in values))
    return result

def main():
    torch.set_num_threads(1);fit_probe.Eye=GridEye
    keys=['pair%03d'%i for i in range(1,101)]
    selections={}
    for key in keys:
        selections[key]={}
        for group in ['poses','free']:
            rec=prev.read(OUT/group/(key+'.json'))
            assert rec['exit_status']==0 and rec['sha256']==prev.sha(OUT/group/(key+'.npz'))
            selections[key][group]=rec
    previous=prev.read(PREV/'evidence/results.json')
    regions=prev.read(PREV/'evidence/regions.json')['rows']
    target_path=prev.ref.SOURCE/'reference.json'
    assert prev.sha(target_path)==previous['reference_sha256']
    truth=prev.read(target_path)
    prev.base.fit.camera_files=prev.ref.camera_files
    model,cam=prev.ref.common.setup_model()
    cfg=prev.read(prev.base.OUT/'protocol.json')
    items={i['pair']:i for i in prev.read(prev.base.OUT/'prepared.json')['items'] if i['condition']=='rw20000'}
    obs=prev.ref.common.OBS; natives=np.array(prev.ref.common.NATIVE_IDS)[obs]
    rows={};support={m:{'eyes_with_any_hand':0,'hand_channels':0,'retained_samples':0,'original_samples':0,'empty_eyes':0} for m in NEW}
    spot={}
    for i,key in enumerate(keys):
        z=dict(np.load(OUT/'poses'/(key+'.npz')))
        zz=dict(np.load(OUT/'free'/(key+'.npz')))
        assert np.array_equal(z['global0'],zz['global0'])
        z.update(zz)
        old=previous['rows'][key]
        modes={}
        for m in ('v3','matched','outline'):
            base=old['methods'][m];geom=regions[key][m]
            modes[m]=dict(pa_mm=base['pa_mm'],nonpa_mm=base['nonpa_mm'],full_outline_px=base['outline_px'],
                          outside_triangles=base['crossings_outside_template'],left_self=geom.get('L_hand/L_hand',0),
                          right_self=geom.get('R_hand/R_hand',0),target_counts=base['target_counts'])
        baseline=np.load(PRIOR/'poses'/(key+'.npz'))['v3']
        orig=dict(body_pose=baseline[:21],delta=baseline[21],translation=baseline[22],global0=z['global0'])
        frame=Frame(key,items[key],orig,model,cam,cfg)
        target=np.asarray(truth[key]['gt'])[obs]
        assert truth[key]['subject']==old['subject']
        for m in NEW:
            x=torch.from_numpy(z[m])
            if m!='free_palm':assert np.array_equal(z[m][23:],baseline[23:])
            with torch.no_grad():
                v,n,_,ok=frame.state(x);assert ok
                zeros=frame.beta[:,:3]
                layer=frame.layer(global_orient=frame.g,betas=frame.beta,body_pose=x[:21].reshape(1,63),
                                  left_hand_pose=x[23:38].reshape(1,45),right_hand_pose=x[38:].reshape(1,45),
                                  jaw_pose=zeros,leye_pose=zeros,reye_pose=zeros,expression=frame.beta)
                j=layer.joints[0]
                pred=((j-j[:1])@rotation(x[21]).T+j[:1]+x[22]).numpy()[natives]
                assert np.max(np.abs(pred-n[0,natives].numpy()))<1e-5
                hits,active=frame.gate.check(v.numpy());counts=summary(hits,frame.labels)
                hands={'/'.join(pair):counts.get('/'.join(sorted(pair)),0) for pair in PAIRS}
                modes[m]=dict(pa_mm=float(np.linalg.norm(prev.align(pred,target)-target,axis=1).mean()*1000),
                              nonpa_mm=float(np.linalg.norm(pred-target,axis=1).mean()*1000),
                              full_outline_px=frame.observed(v)[0].item()*12,
                              left_self=counts.get('L_hand/L_hand',0),right_self=counts.get('R_hand/R_hand',0),
                              target_counts=hands,outside_triangles=len(active),finger_max_delta_rad=float(abs(z[m][23:]-baseline[23:]).max()))
                if key in ['pair001','pair024','pair055','pair068','pair100']:
                    check={side:len(self_intersections(v.numpy().astype(float),frame.faces[frame.labels==side])) for side in ['L_hand','R_hand']}
                    assert all(check[side]==counts.get(side+'/'+side,0) for side in check)
                    spot.setdefault(key,{})[m]=check
            group='free' if m=='free_palm' else 'poses'
            sup=support[m]
            for eye in selections[key][group]['methods'][m]['support']:
                keep=[eye.get(n,[0,0]) for n in ['L_hand','R_hand']]
                sup['eyes_with_any_hand']+=int(any(b>=6 for a,b in keep))
                sup['hand_channels']+=sum(b>=6 for a,b in keep)
                sup['retained_samples']+=sum(b for a,b in keep if b>=6)
                sup['original_samples']+=sum(a for a,b in keep)
                sup['empty_eyes']+=int(not any(b>=6 for a,b in eye.values()))
        rows[key]=dict(subject=old['subject'],methods=modes)
        if (i+1)%20==0:print('graded',i+1,flush=True)
    subjects=sorted({r['subject'] for r in rows.values()})
    result=dict(summary=aggregate(rows,keys),subjects={s:aggregate(rows,[k for k in keys if rows[k]['subject']==s]) for s in subjects},
                support=support,rows=rows,independent_hand_checks=spot,reference_sha256=prev.sha(target_path),exit_status=0)
    prev.write(EV/'results.json',result)
    prev.write(EV/'selections.json',selections)

if __name__=='__main__':main()
