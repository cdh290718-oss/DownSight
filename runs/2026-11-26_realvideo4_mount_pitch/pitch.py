"""Profile an effective pitch correction with translation as a nuisance variable."""
import sys
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

RUN=Path(__file__).resolve().parent
sys.path.insert(0,str(RUN.parent/'2026-11-26_realvideo4_head_mount'))
from check import Case, ROIS, sha, write
from measure import T, cams, projection, OUT, image, read
from fileset import SESSION

center=-T[:3,:3].T@T[:3,3]
axis=center/np.linalg.norm(center)
pivot=center/2
def rot(deg):
    return Rotation.from_rotvec(axis*np.deg2rad(deg)).as_matrix()

def inverse_beam(beam,R,d):
    c,v=beam
    return ((c-pivot-d)@R+pivot,v@R)

class Profile:
    def __init__(self,key):
        self.key=key
        self.case=c=Case(key)
        self.rows=np.arange(350,720,4)
        self.columns=[]
        self.regions=[]
        for eye in range(2):
            if c.edges is not None:
                xx,yy=np.meshgrid(c.edges[eye][:,0],self.rows)
                self.columns.append(c.beams(np.column_stack([xx.ravel(),yy.ravel()]),eye))
            a,b,x,y=ROIS[eye]
            yy,xx=np.mgrid[b:y:4,a:x:4]
            self.regions.append(c.beams(np.column_stack([xx.ravel(),yy.ravel()]),eye))

    def mouth(self,R,d):
        pts=(self.case.landmarks-pivot)@R.T+pivot+d
        return np.vstack((projection(pts[:1],cams[0]),projection(pts[1:]@T[:3,:3].T+T[:3,3],cams[1])))

    def score(self,R,d):
        c=self.case
        mouth=self.mouth(R,d)-c.corners
        edges=[]
        for eye,beam in enumerate(self.columns):
            mask=c.hit(inverse_beam(beam,R,d),np.zeros(3)).reshape(len(self.rows),-1)
            # Refine the first foreground hit in its four-pixel band.
            end=np.where(mask.any(0),self.rows[mask.argmax(0)],720)
            yy=np.maximum(end[None,:]-np.arange(3,-1,-1)[:,None],350)
            xx=np.broadcast_to(c.edges[eye][:,0],yy.shape)
            exact=c.beams(np.column_stack([xx.ravel(),yy.ravel()]),eye)
            hit=c.hit(inverse_beam(exact,R,d),np.zeros(3)).reshape(yy.shape)
            first=np.where(hit.any(0),np.min(np.where(hit,yy,720),axis=0),720)
            edges.append(first-c.edges[eye][:,1])
        return dict(mouth_px=float(np.sqrt((mouth**2).sum(1).mean())),
                    edge_px=float(np.abs(edges).mean()))

    def optimize(self,deg):
        R=rot(deg)
        d=least_squares(lambda t:(self.mouth(R,t)-self.case.corners).ravel(),np.zeros(3),bounds=(-.2,.2)).x
        def calc(t):
            s=self.score(R,t)
            return sum(s.values()),s
        best,s=calc(d)
        for step in [.002,.001,.0005,.00025]:
            for _ in range(12):
                moves=[(calc(d+v*step),d+v*step) for v in np.r_[np.eye(3),-np.eye(3)]]
                (val,nxt),v=min(moves,key=lambda item:item[0][0])
                if val>=best:break
                best,s,d=val,nxt,v
        return dict(deg=deg,translation_m=d.tolist(),train=s)

def samples():
    rows={}
    for k in range(1,5):
        key=f'clip{k:03d}'
        rows[key]={}
        for eye in ['left','right']:
            path,=list((SESSION/'videos'/(eye+'_lower')).glob(f'*{k:03d}.mp4'))
            cap=cv2.VideoCapture(str(path))
            n=cap.get(cv2.CAP_PROP_FRAME_COUNT)
            ids=[]
            for q in [.1,.3,.5,.7,.9]:
                idx=round(n*q)
                cap.set(cv2.CAP_PROP_POS_FRAMES,idx)
                ok,im=cap.read()
                assert ok
                ids.append(idx)
            cap.release()
            rows[key][eye]=dict(video_sha256=sha(path),viewed_frames_zero_based=ids)
    return rows

def main():
    A,B=[Profile(k) for k in ['clip003','clip004']]
    grid=list(range(-40,41,5))
    groups={}
    for train,test in [(A,B),(B,A)]:
        profile=[]
        for deg in grid:
            row=train.optimize(deg)
            row['heldout']=test.score(rot(deg),np.array(row['translation_m']))
            profile.append(row)
        best=min(profile,key=lambda r:sum(r['train'].values()))
        groups[train.key]=dict(heldout_key=test.key,profile=profile,best=best)
        print(train.key,best,flush=True)
    # Vary the rig-relative transform from the visible-group candidate; use each
    # distinct archived body mesh for self-occlusion. Evaluate both negative ROIs.
    base=groups[A.key]['best']
    tests=[Profile(f'clip00{i}') for i in (1,2)]
    negatives=[]
    for deg in range(-80,81,5):
        Q=rot(deg)
        R=Q@rot(base['deg'])
        d=Q@np.array(base['translation_m'])
        row=dict(extra_deg=deg,observations={})
        for p in tests:
            by_shift={}
            for shift_mm in [-10,0,10]:
                # Test just one translation nuisance, the cam3 optical axis.
                t=d+np.array([0.,0.,shift_mm/1000])
                hits=[p.case.hit(inverse_beam(b,R,t),np.zeros(3)) for b in p.regions]
                by_shift[str(shift_mm)]=[dict(head_hits=int(h.sum()),n=len(h)) for h in hits]
            row['observations'][p.key]=by_shift
        negatives.append(row)
    prior=read(RUN.parent.joinpath('2026-11-26_realvideo4_head_mount/evidence/fit003.json'))
    receipts={}
    receipts['mesh_sha256']={p.key:sha(OUT/(p.key+'.npz')) for p in [A,B]+tests}
    receipts['previous_fit_sha256']=sha(RUN.parent/'2026-11-26_realvideo4_head_mount/evidence/fit003.json')
    baseline={p.key:p.score(np.eye(3),np.array(prior['delta_cam3_m'])) for p in [A,B]}
    write(RUN/'evidence/angles.json',dict(visible=groups,negative=negatives,baseline_prior=baseline,
          axis_cam3=axis.tolist(),pivot_cam3_m=pivot.tolist(),inputs=receipts,
          video_inspection=samples(),exit_status=0))
    def overlay(p,transform):
        tiles=[]
        for eye in range(2):
            a=680 if eye==0 else 0
            yy,xx=np.mgrid[535:720,a:a+600]
            beam=p.case.beams(np.column_stack([xx.ravel(),yy.ravel()]),eye)
            im=p.case.images[eye][535:720,a:a+600].copy()
            for R,t,color in transform:
                mask=p.case.hit(inverse_beam(beam,R,t),np.zeros(3)).reshape(xx.shape).astype('uint8')
                edge=cv2.morphologyEx(mask,cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))
                im[edge.astype(bool)]=color
            for x,y in p.case.edges[eye]:
                cv2.circle(im,(int(x)-a,int(y)-535),2,(0,245,255),-1)
            x,y=p.case.corners[eye]
            cv2.drawMarker(im,(int(x)-a,int(y)-535),(0,245,255),cv2.MARKER_CROSS,12)
            for R,t,color in transform:
                x,y=p.mouth(R,t)[eye]
                cv2.circle(im,(round(x)-a,round(y)-535),5,color,2)
            tiles.append(im)
        return np.hstack(tiles)
    out=[]
    for p,train in [(A,B),(B,A)]:
        g=groups[train.key]
        zero=next(z for z in g['profile'] if z['deg']==0)
        params=[(rot(z['deg']),np.array(z['translation_m']),color) for z,color in [(zero,(200,100,255)),(g['best'],(90,230,90))]]
        label=np.full((30,1200,3),35,np.uint8)
        cv2.putText(label,f'{p.key} / fitted on {train.key}      cam3                                             cam4',(8,22),0,.55,(255,255,255),1,cv2.LINE_AA)
        out.extend([label,overlay(p,params)])
    assert cv2.imwrite(str(RUN/'assets/heldout.jpg'),np.vstack(out))
if __name__=='__main__':main()
