"""Check the distance convention and measure triangle crossings separately from candidate scoring."""
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
import open3d as o3d
from scipy.spatial import cKDTree
from contact import Part, Surface, PAIRS, EV, OUT, RUN, groups
from inspect_obj import cuts
from select_pose import prev, search, vertices


def distance_test():
    # 100 mm box with the top removed; Part supplies a cap. Signed samples face a physical side.
    mesh = o3d.geometry.TriangleMesh.create_box(.1,.1,.1)
    v,f = np.asarray(mesh.vertices),np.asarray(mesh.triangles)
    f = f[~(v[f,2]==.1).all(1)]
    surface = Surface(Part(f),v)
    result = {}
    for name,x,want in [('inside',.005,(5.,0.)),('outside',-.007,(0.,7.)),('touching',0.,(0.,0.))]:
        value = surface.distances(np.array([[x,.05,.05]],np.float32))
        assert np.allclose(value,want,atol=1e-3)
        result[name]=value
    return result


def pair_count(v, fa, fb):
    a,b = v[fa],v[fb]
    if ((a.max((0,1))<b.min((0,1))) | (b.max((0,1))<a.min((0,1)))).any():
        return 0
    al,ah,bl,bh = a.min(1),a.max(1),b.min(1),b.max(1)
    ac,bc = (al+ah)/2,(bl+bh)/2
    ar,br = np.linalg.norm(ah-al,axis=1)/2,np.linalg.norm(bh-bl,axis=1)/2
    near = cKDTree(bc).query_ball_point(ac,ar+br.max())
    ia = np.repeat(np.arange(len(a)),[len(ids) for ids in near])
    if not len(ia):return 0
    ib = np.concatenate(near).astype(int)
    keep = np.all((al[ia]<=bh[ib]) & (bl[ib]<=ah[ia]),axis=1)
    ia,ib=ia[keep],ib[keep]
    keep=~(fa[ia,:,None]==fb[ib,None,:]).any(axis=(1,2))
    aa,bb=a[ia[keep]],b[ib[keep]]
    return int(np.sum(cuts(aa,bb)|cuts(bb,aa)))


def main():
    torch.set_num_threads(1)
    box = distance_test()
    model,_ = prev.ref.common.setup_model()
    layer=model.layer['neutral']
    faces=np.asarray(model.face)
    labels=groups(layer,faces)
    patches={name:faces[labels==name] for name in set(sum((list(x) for x in PAIRS),[]))}
    results={}
    for split in ['validation','test']:
        archive=dict(np.load(search.OUT/(split+'_bank.npz')))
        selected=dict(np.load(OUT/(split+'_selected.npz')))
        assert np.count_nonzero(selected['betas'])==np.count_nonzero(archive['betas'])==0
        choices=prev.read(search.OUT/(split+'_choices.json'))
        evaluation=prev.read(EV/(split+'_evaluation.json'))
        rows={}
        for i,pair in enumerate(selected['pairs'].tolist()):
            r=choices[pair]
            row={}
            for mode,pack,index in [('before',archive,r['offset']+r['best']),('after',selected,i)]:
                if mode=='after' and evaluation['rows'][pair]['selected']=='v3':
                    assert all(np.array_equal(pack[k][i],archive[k][r['offset']+r['best']]) for k in search.FIELDS)
                    row[mode]=row['before']
                else:
                    v=vertices(layer,{k:pack[k][index] for k in search.FIELDS})
                    row[mode]={'/'.join((a,b)):pair_count(v,patches[a],patches[b]) for a,b in PAIRS}
                diag=evaluation['rows'][pair][mode].get('crossings')
                if diag is not None:
                    for key,n in row[mode].items():
                        assert n==diag.get(' / '.join(sorted(key.split('/'))),0)
            rows[pair]=row
        summary={k:sum(any(r[k].values()) for r in rows.values()) for k in ['before','after']}
        summary['new_crossing_frames']=sum(not any(r['before'].values()) and any(r['after'].values()) for r in rows.values())
        results[split]=dict(rows=rows,strict_crossing_frames=summary)
        print(split,summary,flush=True)
    sources={p.name:prev.sha(p) for p in sorted(RUN.glob('*.py'))}
    prev.write(EV/'verification.json',dict(source_sha256=sources,run_sh_sha256=prev.sha(RUN/'run.sh'),
        execution_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        dirty_worktree_during_execution=True,device='CPU',open3d_version=o3d.__version__,
        beta_components=10,all_zero_checked=True,depth_gap_test_mm=box,splits=results,exit_status=0,
        commands=['bash '+str(RUN.relative_to(RUN.parents[1]))+'/run.sh '+cmd for cmd in
                  ['select_pose test','select_pose validation','evaluate test','evaluate validation','verify']]))


if __name__=='__main__':
    main()
