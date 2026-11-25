"""Attribute the observed direction error to stereo elbow and wrist point substitutions."""
from compare import RUN,SOURCE,PREV,seed,p,np,direction_error

def main():
    f='val055'
    items={r['pair']:r for r in p.read(SOURCE/'fit/prepared.json')['items']}
    _,cls=p.ref.common.setup_model()
    cams=[cls(path) for path in p.ref.camera_files()]
    ads=[p.read(path)['input_adapter'] for path in p.ref.camera_files()]
    names=['L_Elbow','L_Wrist']
    tri={}; xy={}
    for mode in ['detector','annotated_all']:
        row=items[f+'_'+mode]
        pts={v['name']:v for v in row['points']}
        t=np.array(row['inputs']['views']['right']['camera']['left_to_view'])
        tri[mode]=np.array([seed.triangulate(pts[k],t,cams,ads) for k in names])
        xy[mode]=np.array([[((np.array(pts[k]['xy'][i])-a['offset'])/a['scale']) for k in names] for i,a in enumerate(ads)])
    observed=tri['detector']; gt=tri['annotated_all']
    mixed={}
    for label,e,w in [('both_detected',observed[0],observed[1]),('official_wrist',observed[0],gt[1]),
                      ('official_elbow',gt[0],observed[1]),('both_official',gt[0],gt[1])]:
        mixed[label]=float(direction_error(w-e,gt[1]-gt[0]))
    offset=observed-gt
    proj={mode:v[:,1]-v[:,0] for mode,v in xy.items()}
    result=dict(frame=f,names=names,mixed_forearm_angle_deg=mixed,triangulated_minus_official_mm=(1000*offset).tolist(),
        detector_minus_official_px=(xy['detector']-xy['annotated_all']).tolist(),
        image_forearm_lengths_px={m:np.linalg.norm(v,axis=1).tolist() for m,v in proj.items()},
        image_forearm_direction_difference_deg=direction_error(
            np.pad(proj['detector'],((0,0),(0,1))),np.pad(proj['annotated_all'],((0,0),(0,1)))).tolist())
    p.write(RUN/'evidence/val055_geometry.json',result)
if __name__=='__main__':
    main()
