from collections import Counter
from pathlib import Path
import sys
import numpy as np
import torch
RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[1]
OUT=ROOT/'outputs/two_stereo_contact_20261124'
EV=RUN/'evidence'
sys.path.insert(0,str(ROOT/'runs/2026-11-24_v3_contact'))
from select_pose import prev, search, vertices
from contact import groups, Contact
from geometry import rotation, project
from surface import crossings
PAIRS=['pair055','pair068']


def summary(hits,labels):
    return dict(Counter('/'.join(sorted((labels[a],labels[b]))) for a,b in hits))


def inputs():
    prev.base.fit.camera_files=prev.ref.camera_files
    model,cam=prev.ref.common.setup_model()
    folder=prev.base.OUT
    cfg=prev.read(folder/'protocol.json')
    audit=prev.read(search.RUN/'evidence/test_search.json')
    for file,key in [(folder/'protocol.json','protocol_sha256'),(folder/'prepared.json','prepared_sha256'),
                     (search.OUT/'test_bank.npz','parameter_sha256'),(search.OUT/'test_choices.json','choices_sha256')]:
        assert prev.sha(file)==audit[key]
    items=prev.read(folder/'prepared.json')['items']
    bank=dict(np.load(search.OUT/'test_bank.npz'))
    choices=prev.read(search.OUT/'test_choices.json')
    data={}
    for pair in PAIRS:
        row=choices[pair]
        orig={k:bank[k][row['offset']+row['best']] for k in search.FIELDS}
        assert not orig['betas'].any()
        data[pair]=items[row['index']],orig
    return model,cam,cfg,data


if __name__=='__main__':
    import time
    torch.set_num_threads(1)
    model,cam,cfg,data=inputs()
    layer=model.layer['neutral']
    faces=np.asarray(model.face,dtype=np.int32)
    labels=groups(layer,faces)
    for k,v in [('template',layer.v_template.detach().numpy())]+[(p,vertices(layer,rec[1])) for p,rec in data.items()]:
        t=time.monotonic()
        hits=crossings(v,faces)
        print(k,len(hits),summary(hits,labels),time.monotonic()-t,flush=True)
