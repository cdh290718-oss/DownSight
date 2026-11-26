"""Full-resolution check of selected examples from the negative-region sweep."""
from pitch import RUN, Profile, ROIS, rot, inverse_beam, np, cv2, read, write
def main():
    base=read(RUN/'evidence/angles.json')['visible']['clip003']['best']
    output={}
    panels=[]
    for key in ['clip001','clip002']:
        p=Profile(key)
        results={}
        tiles=[]
        for eye,(a,b,c,d) in enumerate(ROIS):
            sx=680 if eye==0 else 0
            yy,xx=np.mgrid[500:720,sx:sx+600]
            beam=p.case.beams(np.column_stack([xx.ravel(),yy.ravel()]),eye)
            tile=p.case.images[eye][500:720,sx:sx+600].copy()
            cv2.rectangle(tile,(a-sx,b-500),(c-1-sx,d-501),(0,245,255),1)
            for deg in [0,-15,-20,-25,-30,-40]:
                Q=rot(deg)
                R=Q@rot(base['deg'])
                t=Q@np.array(base['translation_m'])
                mask=p.case.hit(inverse_beam(beam,R,t),np.zeros(3)).reshape(xx.shape)
                roi=mask[b-500:d-500,a-sx:c-sx]
                assert roi.size==(c-a)*(d-b)
                results.setdefault(str(deg),[]).append(dict(head_hits=int(roi.sum()),n=int(roi.size)))
                if deg in [0,-15,-25]:
                    line=cv2.morphologyEx(mask.astype('uint8'),cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))
                    tile[line.astype(bool)]={0:(203,102,255),-15:(230,160,80),-25:(94,230,84)}[deg]
            tiles.append(tile)
        name=np.full((27,1200,3),35,np.uint8)
        cv2.putText(name,key+'        cam3                                           cam4',(5,20),0,.55,(255,255,255),1)
        panels.extend([name,np.hstack(tiles)])
        output[key]=results
    assert cv2.imwrite(str(RUN/'assets/negative.jpg'),np.vstack(panels))
    write(RUN/'evidence/negative_native.json',dict(base_from='angles.json: visible.clip003.best',rows=output,exit_status=0))
if __name__=='__main__':main()
