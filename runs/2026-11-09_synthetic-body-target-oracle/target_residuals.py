"""Evaluate automatic-target bias, fit residual, and ankle-relative feet separately."""
import hashlib,json
import numpy as np
from common import ROOT,RUN,OUT,OLD,read
from body_targets import replace_targets
source=OLD/'dataset/samples/000001/left_egosmplx65.npz'
with np.load(source) as data:truth=data['joints_3d_camera_m'].copy()
points=read(OUT/'triangulated.json')['points'];rows=[]
for c in read(RUN/'cases.json'):
 label=c['case']+'_'+c['seed'];effective,intervention=replace_targets(points,read(OUT/'target_overrides'/(c['target_override']+'.json')) if c['target_override'] else None)
 with np.load(OUT/label/'final.npz') as p:pred=p['smplx_joint_cam'][:65].copy()
 targets=[]
 for t in effective:
  if not t['accepted']:continue
  idx=t['index'];observed=np.array(t['xyz']);error=(observed-truth[idx])*1000;fit=(pred[idx]-observed)*1000;absolute=(pred[idx]-truth[idx])*1000;white=np.linalg.solve(np.linalg.cholesky(t['covariance']),fit/1000)
  targets.append(dict(name=t['name'],source=t['source'],group=t['group'],index=idx,target_to_GT_xyz_mm=error.tolist(),target_to_GT_mm=float(np.linalg.norm(error)),fit_to_target_xyz_mm=fit.tolist(),fit_to_target_mm=float(np.linalg.norm(fit)),fit_to_GT_mm=float(np.linalg.norm(absolute)),whitened_norm=float(np.linalg.norm(white))))
 feet=[]
 for i,ankle in [(14,5),(15,5),(16,5),(17,6),(18,6),(19,6)]:
  relative=((pred[i]-pred[ankle])-(truth[i]-truth[ankle]))*1000;feet.append(dict(index=i,ankle=ankle,ankle_relative_error_xyz_mm=relative.tolist(),ankle_aligned_mm=float(np.linalg.norm(relative))))
 opt=read(OUT/label/'optimization.json');rows.append(dict(case=label,targets=targets,feet=feet,pose_increments_deg=opt['pose_increments_deg'],hard_caps_deg=opt['hard_caps_deg'],last_loss_steps=[dict(step=q['step'],loss=q['loss']) for q in opt['trace'][-5:]]))
q=dict(exit_status=0,scope='Post-fit diagnosis only; scalar distances do not add, error vectors do. Automatic hand targets retain their biases in the GT-body intervention. Foot targets were not added.',truth_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),cases=rows)
(RUN/'evidence/target_residuals.json').write_text(json.dumps(q,indent=2)+'\n');print('target diagnostics',len(rows))
