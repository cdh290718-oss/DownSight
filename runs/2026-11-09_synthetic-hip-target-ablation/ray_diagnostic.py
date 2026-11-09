"""Post-fit analysis only: retain automatic target ray, evaluate at held-out true depth."""
import hashlib,json
import numpy as np
from common import RUN,OUT,OLD,read
source=OLD/'dataset/samples/000001/left_egosmplx65.npz'
with np.load(source) as z:truth=z['joints_3d_camera_m'].copy()
automatic=read(OUT/'triangulated.json')['points'];stages=read(RUN/'evidence/metrics.json')['views'][0]['stages'];rows=[]
for p in automatic:
 if p['name'] not in ['left_knee','right_knee','left_ankle','right_ankle']:continue
 idx=p['index'];target=np.array(p['xyz']);gt=truth[idx];scaled=target*gt[2]/target[2]
 rows.append(dict(name=p['name'],index=idx,automatic_target_xyz_m=target.tolist(),true_xyz_m=gt.tolist(),automatic_ray_at_true_Z_m=scaled.tolist(),ray_at_true_Z_error_xyz_mm=((scaled-gt)*1000).tolist(),control_error_xyz_mm=stages['control_left']['per_joint'][idx]['error_xyz_mm'],gt_hips_error_xyz_mm=stages['gt_hips_left']['per_joint'][idx]['error_xyz_mm']))
q=dict(exit_status=0,scope='Evaluation-only cam3 diagnostic; no fitting or target changes. Scale automatic triangulated 3D point along ray from cam3 origin to held-out true Z, then inspect residual Y. Triangulated point ray need not exactly equal either original detection ray because of reprojection residual.',equation='p_at_true_Z = p_auto * Z_GT / Z_auto',gt_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),rows=rows,interpretation='Residual Y after correcting Z indicates a directional mismatch in the automatic target. Agreement with fitted Y supports an observation-direction contribution; it does not separate definition mismatch, detection errors, calibration or optimization interactions.')
(RUN/'evidence/ray_diagnostic.json').write_text(json.dumps(q,indent=2)+'\n')
for r in rows:print(r['name'],'ray_Y_mm',r['ray_at_true_Z_error_xyz_mm'][1],'fit_Y_mm',r['gt_hips_error_xyz_mm'][1])
