"""Palm-aligned local views with strict intersecting surface triangles highlighted."""
from mesh_audit import ROOT,RUN,OUT,ref,groups,self_intersections,read
import torch,cv2
import numpy as np

def paint(v,f,hit,basis):
    q=v@basis
    bounds=q[np.unique(f),:2]
    scale=205/np.ptp(bounds,axis=0).max()
    xy=np.rint((q[:,:2]-(bounds.max(0)+bounds.min(0))/2)*scale*[1,-1]+[130,138]).astype('i4')
    tri=q[f]
    n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    light=abs(n@np.array([-.3,.4,.85]))/np.maximum(np.linalg.norm(n,axis=1),1e-12)
    hot=set(np.asarray(hit,dtype=int).ravel().tolist())
    im=np.full((260,260,3),249,np.uint8)
    for i in np.argsort(tri[:,:,2].mean(1)):
        shade=round(99+128*light[i])
        cv2.fillConvexPoly(im,xy[f[i]],(45,60,220) if i in hot else (shade,shade,shade),cv2.LINE_AA)
    return im

def main():
    model,_=ref.common.setup_model();layer=model.layer['neutral'];faces=np.asarray(model.face)
    labels=groups(layer,faces);counts=read(RUN/'evidence/native.json')
    dest=RUN/'assets';dest.mkdir(exist_ok=True)
    for key in ('val001_left','val184_left','val072_right','val155_right'):
        k,side=key.split('_');i=int(side=='right')
        src=np.load(OUT/'poses'/(key+'.npz'))
        x=np.load(ROOT/'outputs/validation200_v5_20261127/rows'/(k+'.npz'))['v5']
        z=np.load(ROOT/'outputs/validation200_fingers_20261125/poses'/(k+'.npz'))
        t=lambda data:torch.as_tensor(data,dtype=torch.float32).reshape(1,-1)
        zeros=torch.zeros(1,3)
        mesh=layer(global_orient=t(z['global0']),betas=torch.zeros(1,10),body_pose=t(x[:21]),left_hand_pose=t(x[23:38]),right_hand_pose=t(x[38:]),jaw_pose=zeros,leye_pose=zeros,reye_pose=zeros,expression=torch.zeros(1,10))
        v,j=mesh.vertices[0].numpy(),mesh.joints[0].numpy()
        hand=25+15*i
        cols=[]
        for tag,vv,ff,p in [('MANO',src['vertices'],src['faces'],src['joints'][[0,5,9,17]]),
                            ('SMPL-X',v,faces[labels==('L_hand','R_hand')[i]],j[[20+i,hand,hand+3,hand+6]])]:
            vv=vv.astype(float);hits=self_intersections(vv,ff)
            assert len(hits)==counts[key]['native' if tag=='MANO' else 'final']
            unit=lambda t:t/np.linalg.norm(t)
            up=unit(p[2]-p[0]);across=unit(np.cross(up,p[1]-p[3]));h=unit(np.cross(up,across))
            panels=[paint(vv,ff,hits,np.stack([axis,up,np.cross(axis,up)],axis=1)) for axis in (h,across)]
            cv2.putText(panels[0],tag+' | '+str(len(hits))+' pairs',(8,18),0,.49,(30,30,30),1,cv2.LINE_AA)
            cols.append(np.vstack(panels))
        assert cv2.imwrite(str(dest/(key+'.jpg')),np.hstack(cols))
if __name__=='__main__':
    torch.set_num_threads(1)
    with torch.no_grad():main()
