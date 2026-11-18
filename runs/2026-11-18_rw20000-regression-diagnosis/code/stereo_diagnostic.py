"""Read-only ray geometry diagnostic; not a replacement 3D loss or GT metric."""
import hashlib
import json
from pathlib import Path
import numpy as np
from egosmplx_pipeline.camera import FishEyeCameraCalibrated

RUN=Path(__file__).resolve().parents[1];ROOT=RUN.parents[1]
def read(p):return json.loads(p.read_text())

def main():
    archive=ROOT.parent/'EgoFishPose/experiments/benchmark_body_hand10_head01_20261014_v1'
    files=[archive/'body_calibration_fisheye'/('camera_'+side+'.json') for side in ['left','right']]
    cams=[FishEyeCameraCalibrated(p) for p in files]
    rigfile=ROOT/'runs/2026-11-15_unrealego-gt2d/evidence/preparation_ideal.json'
    rig=read(rigfile);rotation=np.array(rig['rig_rotation']);translation=np.array(rig['rig_translation_m'])
    origin=-rotation.T@translation
    def geometry(xy):
        rays=[c.camera2world_ray(np.array([u],float))[0] for c,u in zip(cams,xy)]
        a=rays[0];b=rotation.T@rays[1]
        matrix=np.stack([a,-b],axis=1);dist=np.linalg.lstsq(matrix,origin,rcond=None)[0]
        pa=a*dist[0];pb=origin+b*dist[1]
        return (pa+pb)/2,dict(ray_gap_mm=float(np.linalg.norm(pa-pb)*1000),
            parallax_deg=float(np.rad2deg(np.arccos(np.clip(a@b,-1,1)))),positive_depths=bool((dist>0).all()))
    refs=read(ROOT/'outputs/2d_intervention_20261117_retry1/prepared.json')['items']
    gt={r['pair']:{p['name']:geometry(p['xy']) for p in r['points']} for r in refs if r['condition']=='all_full'}
    rows=read(ROOT/'outputs/rw_a20000_evaluation_20261118/d_fit/prepared.json')['items'];records=[]
    for row in rows:
        if row['condition'] not in ['rw5000','rw20000']:continue
        for p in row['points']:
            point,m=geometry(p['xy']);ref,refmetrics=gt[row['pair']][p['name']]
            records.append(dict(pair=row['pair'],version=row['condition'],joint=p['name'],
                reference_ray_gap_mm=refmetrics['ray_gap_mm'],midpoint_distance_to_2d_reference_triangulation_mm=float(np.linalg.norm(point-ref)*1000),**m))
    aggregate={}
    for joint in sorted({r['joint'] for r in records}):
        aggregate[joint]={}
        for version in ['rw5000','rw20000']:
            rr=[r for r in records if r['joint']==joint and r['version']==version]
            aggregate[joint][version]={k:float(np.mean([r[k] for r in rr])) for k in ['ray_gap_mm','midpoint_distance_to_2d_reference_triangulation_mm']}
    result=dict(command='PYTHONPATH=<REPO> <BODY_PYTHON> code/stereo_diagnostic.py',exit_status=0,records=records,by_joint=aggregate,
        calibration_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},rig_sha256=hashlib.sha256(rigfile.read_bytes()).hexdigest(),
        scope='Closest-ray midpoint compared to triangulated projected-2D reference, not body-3D GT. Unfiltered points may be ill conditioned. Descriptive geometry only; no change to golden-D fitting.')
    (RUN/'evidence/stereo_diagnostic.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(right_wrist=aggregate['R_Wrist'],body07=[r for r in records if r['pair']=='body07' and r['joint'] in ['R_Knee','R_Ankle']]),indent=2))

if __name__=='__main__':main()
