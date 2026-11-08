"""Export explicitly requested A/B/C/D/E meshes, all in cam3 CV meters."""
import hashlib,json,os
from pathlib import Path
import numpy as np
from egosmplx_pipeline.bootstrap import setup
from egosmplx_pipeline.io import load,read
ROOT=Path(__file__).resolve().parents[2];RUN=Path(__file__).resolve().parent;DEST=ROOT/'data/meshes/synthetic_ABCDE'
config=read(ROOT/'outputs/synthetic_original_20261108/configs/left.json');paths=config['paths'];os.environ.update(EGOSMPLX_CONFIG=paths['egosmplx_config'],EGOSMPLX_REPO_ROOT=paths['egosmplx_repo']);model,_=setup(paths['egosmplx_repo'])
from egosmplx_pipeline import fusion
fusion.initialize(model);faces=np.asarray(model.face,dtype=np.int64)
ABCD=ROOT/'outputs/synthetic_stereo_abcd_20261108';E=ROOT/'outputs/synthetic_shape_prior_20261108'
specs=[('A_cam3',ABCD/'A_left_left','A independent fit using cam3 observations'),('A_cam4',ABCD/'A_right_left','A independent fit using cam4 observations; stored in cam3 frame'),('B',ABCD/'B_left','Shared stereo; fixed predicted shape'),('C',ABCD/'C_left','Shared stereo; doubled pose hard bounds'),('D',ABCD/'D_left','Shared stereo; fixed ground-truth shape'),('E',E/'E_left','Shared stereo; optimized shape with prior weight 0.001')]
rows=[]
for label,folder,description in specs:
 source=folder/'final.npz';params=load(source);opt=read(folder/'optimization.json');digest=hashlib.sha256(source.read_bytes()).hexdigest();assert digest==opt['final']['sha256']
 rebuilt,_=fusion.geometry(params);vertices=params['smplx_mesh_cam'];error=float(np.max(abs(vertices-rebuilt)));assert error<1e-5
 assert vertices.shape==(10475,3) and faces.shape==(20908,3);assert np.isfinite(vertices).all();assert faces.min()==0 and faces.max()<len(vertices)
 target=DEST/(label+'.obj')
 with target.open('x') as f:
  f.write('# DownSight synthetic A-B-C-D-E comparison\n# '+description+'\n# Coordinate frame: cam3 CV; meters; X right, Y down, Z forward\n# Network initialization: cam3 (left); no MANO hand replacement\n')
  for v in vertices:f.write('v %.9f %.9f %.9f\n'%tuple(v))
  for face in faces:f.write('f %d %d %d\n'%tuple(face+1))
 vv=[];ff=[]
 for line in target.read_text().splitlines():
  if line.startswith('v '):vv.append([float(x) for x in line.split()[1:]])
  elif line.startswith('f '):ff.append([int(x)-1 for x in line.split()[1:]])
 vv=np.array(vv);ff=np.array(ff);roundtrip=float(np.max(abs(vv-vertices)));assert roundtrip<=5.01e-10;assert np.array_equal(ff,faces)
 rows.append(dict(file=str(target.relative_to(ROOT)),description=description,coordinate_frame='cam3 CV',unit='meter',seed='left/cam3',source_parameters=str(source.relative_to(ROOT)),source_sha256=digest,best_step=opt['best_step'],case=opt['case'],shape_prior_weight=opt.get('shape_prior_weight'),vertex_count=len(vertices),face_count=len(faces),bytes=target.stat().st_size,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),mesh_reconstruction_max_m=error,obj_roundtrip_max_m=roundtrip,bounds_min_m=vertices.min(0).tolist(),bounds_max_m=vertices.max(0).tolist(),behind_camera_vertex_count=int((vertices[:,2]<0).sum()),mano_replacement=False))
result=dict(exit_status=0,command='<BODY_PYTHON> runs/2026-11-08_publish-synthetic-abcde/export.py',scope='Six explicitly requested meshes; A has two independent fits. B/C/D/E use primary cam3 network initialization. Every OBJ uses cam3 CV meters. No transforms, recentering or near-plane vertex removal.',records=rows)
(RUN/'evidence/meshes.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS exported six OBJ files; rebuilt parameters and OBJ roundtrip verified')
