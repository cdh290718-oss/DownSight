from pathlib import Path
from collections import Counter
import sys,json,hashlib
import numpy as np
import cv2
import torch
RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[1]
sys.path.insert(0,str(RUN.parent/'2026-11-23_floor24_intersections'))
from inspect_obj import ref,groups,self_intersections
from scipy.spatial.transform import Rotation
ARCH=RUN.parent/'2026-11-27_validation200_v5/evidence'
read=lambda p:json.loads(p.read_text())
cases=read(ARCH/'results.json')['rows']
choices=read(ARCH/'selection.json')
sides=('left','right')
def unit(x):return x/np.linalg.norm(x)
def panel(positions,faces,hits,basis,side):
    projected=[v@basis for v in positions]
    common=np.vstack([q[np.unique(faces),:2] for q in projected])
    scale=245/max(np.ptp(common,axis=0)); center=(common.min(0)+common.max(0))/2
    panels=[]
    for label,q,contact in zip(('v4','v5'),projected,hits):
        img=np.full((308,300,3),245,np.uint8)
        xy=np.rint((q[:,:2]-center)*scale*[1,-1]+[150,170]).astype('i4')
        triangles=q[faces]
        normal=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
        light=np.abs(normal@unit(np.array([-.3,.4,.85])))/np.maximum(np.linalg.norm(normal,axis=1),1e-10)
        marked=set(n for pair in contact for n in pair)
        for n in np.argsort(triangles[:,:,2].mean(1)):
            shade=int(105+124*light[n])
            color=(46,65,214) if n in marked else (shade,min(255,shade+6),shade)
            cv2.fillConvexPoly(img,xy[faces[n]],color,cv2.LINE_AA)
        cv2.putText(img,label+'  '+side+'  |  '+str(len(contact))+' pairs',(12,25),0,.57,(42,44,46),1,cv2.LINE_AA)
        panels.append(img)
    return np.hstack(panels)
def main():
    torch.set_num_threads(1)
    model,_=ref.common.setup_model();layer=model.layer['neutral'];faces=np.asarray(model.face)
    labels=groups(layer,faces); weights=layer.lbs_weights.numpy()
    outputs={};samples={'val003','val001','val044','val184'}
    for k in sorted(cases):
        prev=ROOT/'outputs/validation200_fingers_20261125/poses'/(k+'.npz')
        new=ROOT/'outputs/validation200_v5_20261127/rows'/(k+'.npz')
        assert all(hashlib.sha256(f.read_bytes()).hexdigest()==choices[k][h] for f,h in [(prev,'seed_sha256'),(new,'sha256')])
        z=np.load(prev);a=z['fixed_full'];b=np.load(new)['v5']
        wrist=a.copy();wrist[19:21]=b[19:21]
        finger=a.copy();finger[23:]=b[23:]
        def geometry(x):
            t=lambda y:torch.as_tensor(y,dtype=torch.float32).reshape(1,-1)
            with torch.no_grad():
                g=layer(global_orient=t(z['global0']),betas=t(np.zeros(10)),body_pose=t(x[:21]),left_hand_pose=t(x[23:38]),right_hand_pose=t(x[38:]),jaw_pose=t(np.zeros(3)),leye_pose=t(np.zeros(3)),reye_pose=t(np.zeros(3)),expression=t(np.zeros(10)))
            rot=Rotation.from_rotvec(x[21]).as_matrix()
            return g.vertices[0].numpy()@rot.T,g.joints[0].numpy()@rot.T
        values=[geometry(x) for x in [a,wrist,finger,b]]
        row=outputs[k]={};tiles=[]
        for i,side in enumerate(sides):
            f=faces[labels==('L_hand','R_hand')[i]]
            hits=[self_intersections(v,f) for v,j in values]
            counts=list(map(len,hits))
            fingerweights=weights[f].sum(1)[:,25+i*15:40+i*15].reshape(-1,5,3).sum(2)
            names=np.array(['palm','index','middle','pinky','ring','thumb'])
            parts=names[np.column_stack([weights[f].sum(1)[:,20+i],fingerweights]).argmax(1)]
            pairs=Counter('/'.join(sorted([parts[l],parts[r]])) for l,r in hits[-1])
            archived=cases[k]['methods']['v5'][side+'_self']
            row[side]=dict(v4=counts[0],wrist_only=counts[1],finger_only=counts[2],v5=counts[3],archived=archived,source=choices[k]['hands'][side]['source'],regions=pairs)
            if k in samples and archived:
                j=values[0][1];ix=25+i*15
                up=unit(j[ix+3]-j[20+i]);across=unit(np.cross(up,j[ix]-j[ix+6]))
                hor=unit(np.cross(up,across))
                for angle in (0,np.pi/2):
                    h=np.cos(angle)*hor+np.sin(angle)*across
                    tiles.append(panel([values[0][0],values[-1][0]],f,[hits[0],hits[-1]],np.stack([h,up,np.cross(h,up)],axis=1),side))
        if tiles:assert cv2.imwrite(str(RUN/'assets'/(k+'.jpg')),np.vstack(tiles))
        if int(k[-3:])%25==0:print(k,flush=True)
    (RUN/'evidence/hands.json').write_text(json.dumps(outputs,indent=1)+'\n')
if __name__=='__main__':main()
