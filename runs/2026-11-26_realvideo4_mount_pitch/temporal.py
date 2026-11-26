"""Joint fit on the two middle frames; score separately marked early/late frames."""
from pitch import RUN, Profile, rot, inverse_beam, SESSION, cv2, np, write, least_squares
from frames import timestamps
from measure import observed_edge, windows

def frames(key):
    files=[next((SESSION/'videos'/(eye+'_lower')).glob('*'+key[-3:]+'.mp4')) for eye in ['left','right']]
    clocks=[timestamps(p.with_name(p.stem+'_时间戳.csv')) for p in files]
    start=max(c[0] for c in clocks)
    stop=min(c[-1] for c in clocks)
    for frac in [.1,.9]:
        li=int(np.argmin(abs(clocks[0]-(start+(stop-start)*frac))))
        ri=int(np.argmin(abs(clocks[1]-clocks[0][li])))
        imgs=[]
        for i,path in zip([li,ri],files):
            cap=cv2.VideoCapture(str(path))
            assert cap.set(cv2.CAP_PROP_POS_FRAMES,i)
            ok,im=cap.read()
            assert ok and int(cap.get(cv2.CAP_PROP_POS_FRAMES))==i+1
            imgs.append(im)
            cap.release()
        yield f'{key}_{int(100*frac)}',imgs,dict(indices_zero_based=[li,ri],delta_ms=abs(int(clocks[0][li]-clocks[1][ri]))/10000)

# Near anatomical mouth corner in each camera, read on magnified raw frames.
markers={'clip003_10':[[849,665],[484,655]],'clip003_90':[[850,664],[488,657]],
         'clip004_10':[[865,685],[468,678]],'clip004_90':[[866,694],[460,683]]}

def joint_score(profiles,R,d):
    values=[p.score(R,d) for p in profiles]
    return {k:float(np.mean([s[k] for s in values])) for k in values[0]}

def joint_fit(profiles,deg):
    R=rot(deg)
    d=least_squares(lambda t:np.concatenate([(p.mouth(R,t)-p.case.corners).ravel() for p in profiles]),
                    np.zeros(3),bounds=(-.2,.2)).x
    def value(t):
        return sum(joint_score(profiles,R,t).values())
    best=value(d)
    for step in [.002,.001,.0005,.00025]:
        for _ in range(12):
            moves=[(value(d+v*step),d+v*step) for v in np.r_[np.eye(3),-np.eye(3)]]
            v,t=min(moves,key=lambda m:m[0])
            if v>=best: break
            best,d=v,t
    return dict(deg=deg,translation_m=d.tolist(),train=joint_score(profiles,R,d))

def main():
    train=[Profile(k) for k in ['clip003','clip004']]
    candidates=[joint_fit(train,deg) for deg in range(-40,41,5)]
    best=min(candidates,key=lambda z:sum(z['train'].values()))
    control=next(z for z in candidates if z['deg']==0)
    held={}
    panels=[]
    for p in train:
        for tag,imgs,info in frames(p.key):
            p.case.edges=[observed_edge(im,w) for im,w in zip(imgs,windows[p.key])]
            p.case.corners=np.array(markers[tag])
            info.update(mouth_label_px=markers[tag],edges=[uv.tolist() for uv in p.case.edges])
            tiles=[]
            for eye,im in enumerate(imgs):
                start=710 if eye==0 else 0
                tile=im[550:,start:start+570].copy()
                for x,y in p.case.edges[eye]:
                    cv2.circle(tile,(x-start,y-550),2,(0,245,255),-1)
                x,y=markers[tag][eye]
                cv2.drawMarker(tile,(x-start,y-550),(0,245,255),cv2.MARKER_CROSS,14)
                tiles.append(tile)
            for name,c,color in [('zero',control,(203,102,255)),('fit',best,(94,230,84))]:
                R=rot(c['deg'])
                t=np.array(c['translation_m'])
                info[name]=p.score(R,t)
                for eye,tile in enumerate(tiles):
                    start=710 if eye==0 else 0
                    yy,xx=np.mgrid[550:720,start:start+570]
                    beam=p.case.beams(np.column_stack([xx.ravel(),yy.ravel()]),eye)
                    mask=p.case.hit(inverse_beam(beam,R,t),np.zeros(3)).reshape(xx.shape).astype('uint8')
                    edge=cv2.morphologyEx(mask,cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))
                    tile[edge.astype(bool)]=color
                    x,y=p.mouth(R,t)[eye]
                    cv2.circle(tile,(round(x)-start,round(y)-550),5,color,2)
            line=np.full((27,1140,3),35,np.uint8)
            cv2.putText(line,tag+'%       cam3                                             cam4',(5,21),0,.56,(255,255,255),1)
            panels.extend([line,np.hstack(tiles)])
            held[tag]=info
    assert cv2.imwrite(str(RUN/'assets/temporal.jpg'),np.vstack(panels))
    scores={name:{metric:float(np.mean([row[name][metric] for row in held.values()])) for metric in ['mouth_px','edge_px']} for name in ['zero','fit']}
    print(best,control,scores,flush=True)
    write(RUN/'evidence/temporal.json',dict(training=candidates,selected=best,zero=control,
          heldout=held,heldout_frame_means=scores,exit_status=0))
if __name__=='__main__':main()
