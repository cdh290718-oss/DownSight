"""Check order-independent triangle visibility on overlapping depth planes."""
import numpy as np
from rasterizer import rays,render
K=np.array([60.,60.,40.,40.,0.,.5]);v=np.array([[-.3,-.3,1.],[.3,-.3,1.],[0,.3,1.],[-.6,-.6,2.],[.6,-.6,2.],[0,.6,2.]],np.float32);f=np.array([[0,1,2],[3,4,5]],np.int32);head=np.array([1,1,1,0,0,0],np.float32);r=rays(K,80,80);c,z,h,_=render(v,f,head,K,r);c2,z2,h2,_=render(v,f[::-1].copy(),head,K,r);assert np.isfinite(z[40,40]) and abs(float(z[40,40])-1)<1e-5;assert h[40,40]>.99;assert np.array_equal(np.isfinite(z),np.isfinite(z2)) and np.allclose(z,z2) and np.array_equal(c,c2);print('PASS: overlapping near triangle wins regardless of face order; head color belongs to visible face')
