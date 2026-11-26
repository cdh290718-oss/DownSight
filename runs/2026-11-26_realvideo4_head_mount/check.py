"""Hold the fit fixed across clips; compare the visible head masks at native scale."""
from measure import Case,RUN,OUT,ROOT,read,write,sha,points,windows,cv2,np,eyes

FIT_SHA='2972b61a227024ec90e1d881e1f2b87ecaf3c64dc87d65007ca0dda95c058f3a'
# Evaluation landmarks were labelled after 003 was sealed. Trim the very edge
# of both sensors in the definite non-face check to avoid ambiguous border pixels.
points['clip004']=[[864,692],[460,679]]
windows['clip004']=[(860,1010,550),(340,490,550)]
ROIS=[(800,550,1250,706),(30,550,480,706)]
COLORS=[(203,102,255),(94,230,84)]

def main(refitted=False):
    src=RUN/'evidence/fit003.json'
    assert sha(src)==FIT_SHA
    fit=read(src)
    delta=np.array(fit['delta_cam3_m'])
    rows={}
    (RUN/'assets').mkdir(exist_ok=True)
    for key in ['clip001','clip002','clip003','clip004']:
        case=Case(key)
        other=Case(key,OUT/'refitted') if refitted else case
        shift=np.zeros(3) if refitted else delta
        row=dict(head_original_m=case.head.tolist(),head_candidate_m=(other.head+shift).tolist(),views={})
        if case.edges is not None:
            row.update(before=case.errors(np.zeros(3)),after=other.errors(shift))
        panels=[]
        for e in range(2):
            im=case.images[e].copy()
            # The bottom 220 rows cover all the annotated control patches and mouth/edge points.
            ys,xs=np.mgrid[500:720,0:1280]
            beam=case.beams(np.column_stack((xs.ravel(),ys.ravel())),e)
            a,b,c,d=ROIS[e]
            info={}
            for tag,geometry,offset,color in zip(('before','after'),(case,other),(np.zeros(3),shift),COLORS):
                mask=geometry.hit(beam,offset).reshape(xs.shape).astype('uint8')
                # Exclude artificial image and crop boundaries when displaying contours.
                edge=cv2.morphologyEx(mask,cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8)).astype(bool)
                im[500:][edge]=color
                info[tag+'_head_px_bottom220']=int(mask.sum())
                if case.corners is None:
                    info[tag+'_wrong_head_px']=int(mask[b-500:d-500,a:c].sum())
                else:
                    xy=geometry.mouth(offset)[e]
                    cv2.circle(im,tuple(xy.round().astype(int)),7,color,2)
            if case.corners is None:
                info['negative_rectangle_xyxy']=list(ROIS[e])
                info['area_px']=(c-a)*(d-b)
                cv2.rectangle(im,(a,b),(c-1,d-1),(0,225,255),2)
            else:
                for xy in case.edges[e]:
                    cv2.circle(im,tuple(xy.astype(int)),3,(0,225,255),-1)
                cv2.drawMarker(im,tuple(case.corners[e]),(0,225,255),cv2.MARKER_CROSS,17,2)
            row['views'][eyes[e]]=info
            # Keep the frame bottom and lateral sensor boundary. Separate views share a scale.
            roi=im[500:,600:] if e==0 else im[500:,:680]
            panels.append(roi)
        both=np.hstack(panels)
        header=np.full((44,both.shape[1],3),35,np.uint8)
        cv2.putText(header,key+'     cam3 (left)                                                       cam4 (right)',(14,29),0,.78,(240,240,240),1,cv2.LINE_AA)
        path=RUN/'assets'/(('refit_' if refitted else '')+key+'.jpg')
        assert cv2.imwrite(str(path),np.vstack((header,both)),[cv2.IMWRITE_JPEG_QUALITY,88])
        row['overlay_sha256']=sha(path)
        rows[key]=row
        print(key,row.get('after',{}).get('mouth_rmse_px'),[(v.get('before_wrong_head_px'),v.get('after_wrong_head_px')) for v in row['views'].values()])
    write(RUN/('evidence/refitted.json' if refitted else 'evidence/check.json'),dict(frozen_fit_sha256=FIT_SHA,rows=rows,exit_status=0))
if __name__=='__main__':
    import sys
    main('refitted' in sys.argv[1:])
