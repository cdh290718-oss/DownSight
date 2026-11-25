"""Lock both local ankle and forefoot rotations throughout the existing full-contour fit."""
from pathlib import Path
import sys
import cv2
import numpy as np
import torch

RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[1]
OUT=ROOT/'outputs/floor24_foot_lock_20261125'
PREVIOUS=RUN.parent/'2026-11-25_fixed_full_floor24'
sys.path.insert(0,str(RUN.parent/'2026-11-25_validation200_fingers'))
from inputs import EdgeFrame, prev, search, guard, compare, from_branch, WEIGHT
FEET=[6,7,9,10]


class FixedFeet(EdgeFrame):
    def cap(self,x):
        candidate=super().cap(x)
        candidate[FEET]=self.v3[FEET]
        return candidate


def run(start,stride):
    sys.addaudithook(guard)
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    model,camera=prev.ref.common.setup_model()
    gallery=prev.read(ROOT/'runs/2026-11-23_fixed_floor24/evidence/sample.json')['rows']
    for split,split_run,folder in [('test','frozen100_outline',prev.base.OUT),('validation','validation200_fingers',prev.VAL)]:
        audit=prev.read(search.RUN/('evidence/'+split+'_search.json'))
        compare.OUT=ROOT/'outputs'/(split_run+'_20261125')
        masks=prev.read(RUN.parent/('2026-11-25_'+split_run)/'evidence/segmentation.json')['eyes']
        bankpath=search.OUT/(split+'_bank.npz')
        choicespath=search.OUT/(split+'_choices.json')
        for f,k in [(bankpath,'parameter_sha256'),(choicespath,'choices_sha256'),
                    (folder/'prepared.json','prepared_sha256'),(folder/'protocol.json','protocol_sha256')]:
            assert prev.sha(f)==audit[k]
        bank=dict(np.load(bankpath))
        choices=prev.read(choicespath)
        items=prev.read(folder/'prepared.json')['items']
        cfg=prev.read(folder/'protocol.json')
        for row in gallery[start::stride]:
            if row['split']!=split:continue
            key=row['pair']
            rec=choices[key]
            item=items[rec['index']]
            assert item['pair']==key
            for side in ('left','right'):
                eye=key+'_'+side
                assert prev.sha(compare.OUT/'seg'/(eye+'.npy'))==masks[eye]['sampled_sha256']
            base={k:bank[k][rec['offset']+rec['best']] for k in search.FIELDS}
            frame=FixedFeet(key,item,base,model,camera,cfg)
            assert not frame.v3[FEET].any() and not base['betas'].any()
            evolved,_=frame.descend(frame.v3,False,True,'v3')
            fitted=dict(base,body_pose=evolved[:21].numpy(),delta=evolved[21].numpy(),translation=evolved[22].numpy())
            with torch.no_grad():
                branches,_=search.frame(item,fitted,fitted,model,camera,cfg)
                pool={'v3':frame.v3,'adam':evolved}
                for i,name in enumerate(rec['names']):
                    pool['v3_'+name]=from_branch(frame.v3,{k:bank[k][rec['offset']+i] for k in search.FIELDS})
                pool.update({'new_'+b['name']:from_branch(evolved,b) for b in branches})
                scores={}
                for name,x in pool.items():
                    assert not x[FEET].any() and not x[23:].any()
                    v,_,raw,valid=frame.state(x)
                    if valid:scores[name]=raw.item()+WEIGHT*frame.observed(v)[0].item()
                best=min(scores,key=scores.get)
            path=OUT/'poses'/(key+'.npz')
            np.savez_compressed(path,locked=pool[best].numpy(),global0=base['global0'])
            prev.write(path.with_suffix('.json'),dict(pair=key,selected=best,sha256=prev.sha(path)))
            print(key,flush=True)


if __name__=='__main__':run(*map(int,sys.argv[1:]))
