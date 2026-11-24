"""Bidirectional sampled intrusion into closed limb segments, in millimetres."""
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import open3d as o3d

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
OUT = ROOT/'outputs/v3_contact_20261124'
EV = RUN/'evidence'
sys.path.insert(0,str(ROOT/'runs/2026-11-23_floor24_intersections'))
from inspect_obj import groups

PAIRS = [('L_hand','R_hand'),('L_hand','L_thigh'),('L_hand','R_thigh'),
         ('R_hand','L_thigh'),('R_hand','R_thigh')]
TOL_MM = 2.
GAP_CAP = 12.


class Part:
    def __init__(self, faces):
        self.ids, indices = np.unique(faces, return_inverse=True)
        self.original = indices.reshape(-1,3)
        links = np.vstack([self.original[:,[i,(i+1)%3]] for i in range(3)])
        edges, cnt = np.unique(np.sort(links,axis=1),axis=0,return_counts=True)
        assert cnt.max()==2
        border = edges[cnt==1]
        graph = defaultdict(set)
        for a,b in border:
            graph[a].add(b)
            graph[b].add(a)
        todo=set(graph)
        self.rings=[]
        extra=[]
        while todo:
            component=set()
            stack=[next(iter(todo))]
            while stack:
                node=stack.pop()
                if node not in component:
                    component.add(node)
                    stack.extend(graph[node]-component)
            todo-=component
            ring=sorted(component)
            center=len(self.ids)+len(self.rings)
            self.rings.append(ring)
            extra.extend([int(a),int(b),center] for a,b in border if a in component)
        self.closed=np.vstack((self.original,extra)).astype(np.uint32)
        edges=np.vstack([self.closed[:,[i,(i+1)%3]] for i in range(3)])
        assert np.all(np.unique(np.sort(edges,axis=1),axis=0,return_counts=True)[1]==2)


class Surface:
    def __init__(self, part, vertices):
        v=vertices[part.ids]
        self.lo,self.hi=v.min(0),v.max(0)
        self.samples=np.vstack([v,v[part.original].mean(1)]).astype(np.float32)
        self.vertices=np.vstack([v]+[v[ring].mean(0) for ring in part.rings]).astype(np.float32)
        self.part=part
        self.scene=None

    def distances(self, samples):
        # Points farther than 12 mm outside the bounding box cannot be close or inside.
        margin=GAP_CAP/1000
        inside_box=((samples>=self.lo-margin)&(samples<=self.hi+margin)).all(1)
        if not inside_box.any():
            return 0.,GAP_CAP
        if self.scene is None:
            self.scene=o3d.t.geometry.RaycastingScene(nthreads=1)
            self.scene.add_triangles(o3d.core.Tensor(self.vertices),o3d.core.Tensor(self.part.closed))
        x=samples[inside_box]
        query=o3d.core.Tensor(x)
        nearest=self.scene.compute_closest_points(query,nthreads=1)
        tissue=nearest['primitive_ids'].numpy()<len(self.part.original)
        if not tissue.any():
            return 0.,GAP_CAP
        x=x[tissue]
        d=np.linalg.norm(nearest['points'].numpy()[tissue]-x,axis=1)*1000
        interior=self.scene.compute_occupancy(o3d.core.Tensor(x),nthreads=1,nsamples=3).numpy()>.5
        depth=float(d[interior].max()) if interior.any() else 0.
        return depth, (0. if depth>0 else min(GAP_CAP,float(d.min())))


class Contact:
    def __init__(self,layer,faces):
        labels=groups(layer,faces)
        self.parts={p:Part(faces[labels==p]) for pair in PAIRS for p in pair}

    def measure(self,vertices):
        surf={k:Surface(p,vertices) for k,p in self.parts.items()}
        result={}
        for a,b in PAIRS:
            x,y=surf[a],surf[b]
            gap=np.maximum(0,np.maximum(x.lo-y.hi,y.lo-x.hi))
            d,g=0.,GAP_CAP
            if np.linalg.norm(gap)*1000<=GAP_CAP:
                da,ga=x.distances(y.samples)
                db,gb=y.distances(x.samples)
                d,g=max(da,db),min(ga,gb)
            result[a+'/'+b]=[d,g]
        return result


def penalty(measures):
    return .01*sum((max(0.,depth-TOL_MM)/10.)**2 for depth,_ in measures.values())


def preserves_near(before,after):
    return all(gap>5. or after[k][1]<=8. for k,(_,gap) in before.items())
