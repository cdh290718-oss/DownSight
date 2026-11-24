"""All strict triangle crossings, excluding triangles sharing vertices (archived convention)."""
import numpy as np
from numba import njit


@njit
def crosses(a,b):
    e1=b[1]-b[0]
    e2=b[2]-b[0]
    for k in range(3):
        direction=a[(k+1)%3]-a[k]
        offset=a[k]-b[0]
        p=np.cross(direction,e2)
        det=np.dot(e1,p)
        if abs(det)<=1e-16:
            continue
        u=np.dot(offset,p)/det
        if u<=1e-7 or u>=1-1e-7:
            continue
        q=np.cross(offset,e1)
        v=np.dot(direction,q)/det
        t=np.dot(e2,q)/det
        if v>1e-7 and u+v<1-1e-7 and 1e-7<t<1-1e-7:
            return True
    return False


@njit
def sweep(f,tri,lo,hi,order):
    hits=[]
    for pos in range(len(order)):
        a=order[pos]
        for n in range(pos+1,len(order)):
            b=order[n]
            if lo[b,0]>hi[a,0]: break
            if (lo[b,1]>hi[a,1] or lo[a,1]>hi[b,1] or lo[b,2]>hi[a,2] or lo[a,2]>hi[b,2]):continue
            shares=False
            for i in range(3):
                for j in range(3):
                    shares |= f[a,i]==f[b,j]
            if not shares and (crosses(tri[a],tri[b]) or crosses(tri[b],tri[a])):
                hits.append((min(a,b),max(a,b)))
    return hits


def crossings(v,faces):
    tri=np.ascontiguousarray(v[faces],dtype=np.float64)
    lo,hi=tri.min(1),tri.max(1)
    return np.asarray(sweep(faces,tri,lo,hi,np.argsort(lo[:,0])),dtype=int).reshape(-1,2)
