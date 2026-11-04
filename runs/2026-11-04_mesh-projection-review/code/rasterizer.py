"""Fisheye triangle rasterizer with per-pixel camera-ray depth test; no point-cloud renderer."""
import numpy as np
from numba import njit

def project(x,ds):
 fx,fy,cx,cy,xi,a=ds;X,Y,Z=np.moveaxis(np.asarray(x),-1,0);d=np.sqrt(X*X+Y*Y+Z*Z);z1=xi*d+Z;den=a*np.sqrt(X*X+Y*Y+z1*z1)+(1-a)*z1;den=np.maximum(den,1e-12);return np.stack([fx*X/den+cx,fy*Y/den+cy],axis=-1)

def rays(ds,w,h):
 fx,fy,cx,cy,xi,a=ds;Y,X=np.indices((h,w),dtype=np.float64);x=(X-cx)/fx;y=(Y-cy)/fy;r=x*x+y*y;disc=1-(2*a-1)*r;z=(1-a*a*r)/(a*np.sqrt(np.maximum(0,disc))+1-a);disc2=z*z+(1-xi*xi)*r;k=(z*xi+np.sqrt(np.maximum(0,disc2)))/(z*z+r);v=np.stack([k*x,k*y,k*z-xi],axis=-1);v/=np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),1e-12);v[(disc<=0)|(disc2<=0)]=np.nan;return v.astype(np.float32)

@njit(cache=False)
def raster(vertices,faces,head,uv,ray,depth,colors,head_pixel,max_edge):
 h,w=depth.shape;drawn=0;rejected=0
 for i in range(faces.shape[0]):
  ia,ib,ic=faces[i];a=vertices[ia];b=vertices[ib];c=vertices[ic]
  if a[2]<=1e-5 or b[2]<=1e-5 or c[2]<=1e-5:rejected+=1;continue
  p=uv[ia];q=uv[ib];r=uv[ic]
  if not (np.isfinite(p).all() and np.isfinite(q).all() and np.isfinite(r).all()):rejected+=1;continue
  d0=(p[0]-q[0])**2+(p[1]-q[1])**2;d1=(p[0]-r[0])**2+(p[1]-r[1])**2;d2=(q[0]-r[0])**2+(q[1]-r[1])**2
  if max(d0,d1,d2)>max_edge*max_edge:rejected+=1;continue
  xmin=max(0,int(np.floor(min(p[0],q[0],r[0])))-2);xmax=min(w-1,int(np.ceil(max(p[0],q[0],r[0])))+2);ymin=max(0,int(np.floor(min(p[1],q[1],r[1])))-2);ymax=min(h-1,int(np.ceil(max(p[1],q[1],r[1])))+2)
  if xmin>xmax or ymin>ymax:continue
  e0=b-a;e1=c-a;n=np.cross(e0,e1);ndot=n[0]*a[0]+n[1]*a[1]+n[2]*a[2];dot00=np.dot(e0,e0);dot01=np.dot(e0,e1);dot11=np.dot(e1,e1);den=dot00*dot11-dot01*dot01
  if abs(den)<1e-16:continue
  nn=np.sqrt(np.dot(n,n));cx=(a[0]+b[0]+c[0])/3.;cy=(a[1]+b[1]+c[1])/3.;cz=(a[2]+b[2]+c[2])/3.;shade=.63+.37*abs(n[0]*cx+n[1]*cy+n[2]*cz)/(nn*np.sqrt(cx*cx+cy*cy+cz*cz)+1e-12);drawn+=1
  for y in range(ymin,ymax+1):
   for x in range(xmin,xmax+1):
    v=ray[y,x];nd= n[0]*v[0]+n[1]*v[1]+n[2]*v[2]
    if not np.isfinite(nd) or abs(nd)<1e-12:continue
    t=ndot/nd
    if t<=0 or t>=depth[y,x]:continue
    dx=t*v[0]-a[0];dy=t*v[1]-a[1];dz=t*v[2]-a[2];d20=dx*e0[0]+dy*e0[1]+dz*e0[2];d21=dx*e1[0]+dy*e1[1]+dz*e1[2];u=(dot11*d20-dot01*d21)/den;v1=(dot00*d21-dot01*d20)/den;v0=1-u-v1
    if u< -1e-5 or v1< -1e-5 or v0< -1e-5:continue
    depth[y,x]=t;hw=v0*head[ia]+u*head[ib]+v1*head[ic];head_pixel[y,x]=hw;edge=min(v0,u,v1)<.035;sh=shade*(.48 if edge else 1.0)
    # BGR: body warm cyan, head magenta. Vertex labels are interpolated.
    colors[y,x,0]=np.uint8((180*(1-hw)+215*hw)*sh);colors[y,x,1]=np.uint8((210*(1-hw)+65*hw)*sh);colors[y,x,2]=np.uint8((50*(1-hw)+225*hw)*sh)
 return drawn,rejected

def render(vertices,faces,head,ds,ray):
 h,w=ray.shape[:2];uv=project(vertices,ds).astype(np.float32);depth=np.full((h,w),np.inf,np.float32);color=np.zeros((h,w,3),np.uint8);hp=np.zeros((h,w),np.float32);drawn,rejected=raster(vertices.astype(np.float32),faces.astype(np.int32),head.astype(np.float32),uv,ray,depth,color,hp,250.);return color,depth,hp,{'projected_faces':int(drawn),'rejected_near_or_long':int(rejected),'visible_pixels':int(np.isfinite(depth).sum()),'visible_head_pixels':int((hp>.5).sum())}
