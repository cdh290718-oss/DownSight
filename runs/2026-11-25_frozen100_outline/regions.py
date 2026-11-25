"""Locate the whole-body intersections underlying the aggregate frame counts."""
from collections import Counter
import numpy as np
import torch
from shared_run import OUT,EV,prev
from common_run import rotation,groups,summary
from fit_probe import Gate
from inspect_obj import self_intersections

def main():
    torch.set_num_threads(1)
    model,_=prev.ref.common.setup_model()
    layer=model.layer['neutral'];faces=np.asarray(model.face,np.int32)
    labels=groups(layer,faces);gate=Gate(faces,labels,layer.v_template.detach().numpy())
    primary=prev.read(EV/'results.json')['rows']
    totals={k:Counter() for k in ['v3','matched','outline']}
    frames={k:Counter() for k in totals}
    rows={}
    spot={}
    for key,row in primary.items():
        states=dict(np.load(OUT/'poses'/(key+'.npz')))
        rows[key]={}
        z=torch.zeros(1,10)
        for k in totals:
            x=torch.tensor(states[k]);g=torch.from_numpy(states['global0']).reshape(1,3)
            with torch.no_grad():
                mesh=layer(global_orient=g,body_pose=x[:21].reshape(1,63),betas=z,left_hand_pose=x[23:38].reshape(1,45),
                           right_hand_pose=x[38:].reshape(1,45),jaw_pose=z[:,:3],leye_pose=z[:,:3],reye_pose=z[:,:3],expression=z)
                pivot=mesh.joints[0,:1]
                v=((mesh.vertices[0]-pivot)@rotation(x[21]).T+pivot+x[22]).numpy()
            _,h=gate.check(v)
            counts=summary(h,labels)
            assert sum(counts.values())==row['methods'][k]['crossings_outside_template']
            if key in ('pair001','pair010','pair024','pair042','pair055','pair068','pair100'):
                check={side:len(self_intersections(v.astype(float),faces[labels==side])) for side in ('L_hand','R_hand')}
                assert all(counts.get(side+'/'+side,0)==value for side,value in check.items())
                spot.setdefault(key,{})[k]=check
            totals[k].update(counts);frames[k].update(counts.keys())
            rows[key][k]=counts
    without={}
    for k in totals:
        rr=[v['methods'][k] for p,v in primary.items() if p!='pair010']
        without[k]={m:float(np.mean([r[m] for r in rr])) for m in ('pa_mm','nonpa_mm')}
    names=set().union(*totals.values())
    change={n:totals['outline'][n]-totals['v3'][n] for n in sorted(names)}
    within={k:sum(bool(r[k].get('L_hand/L_hand') or r[k].get('R_hand/R_hand')) for r in rows.values()) for k in totals}
    prev.write(EV/'regions.json',dict(any_within_hand_frames=within,independent_hand_check=spot,mesh_pair_totals=totals,frames_with_each_region_pair=frames,
               difference_outline_minus_v3=change,rows=rows,without_largest_v3_pa_change=without,exit_status=0))

if __name__=='__main__':main()
