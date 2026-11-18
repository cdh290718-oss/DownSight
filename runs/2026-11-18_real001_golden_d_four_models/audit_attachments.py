"""Validate user-authorized coefficient-only Release attachments before publication."""
import json,zipfile,io,re
from pathlib import Path
import numpy as np
from common import RUN,OUT,read,write,sha
export=read(RUN/'evidence/exports.json');root=OUT/'download_bundle';archive=OUT/export['archive']['file'];expected={x['file']:x for x in export['files']};allowed={'betas','body_pose','global_orient','left_hand_pose','right_hand_pose','jaw_pose','leye_pose','reye_pose','expression','transl','coordinate_frame','gender','units','pose_representation','flat_hand_mean','use_pca'}
def check_npz(raw,name):
 with np.load(io.BytesIO(raw),allow_pickle=False) as z:
  assert set(z.files)==allowed,(name,z.files)
  for k in z.files:
   a=z[k];assert a.dtype.kind!='O' and a.size<1000,(name,k)
   if a.dtype.kind in 'fiu':assert np.isfinite(a).all()
   if a.dtype.kind in 'US':assert not re.search(r'/(?:vepfs|root|home|peter)',str(a))
files={archive.name:sha(archive)}
for entry in expected.values():
 p=root/entry['relative_path'];assert p.name==entry['file'] and sha(p)==entry['sha256'];check_npz(p.read_bytes(),p.name);files[p.name]=sha(p)
with zipfile.ZipFile(archive) as z:
 names=z.namelist();assert len(names)==len(set(names));assert sum(n.endswith('.npz') for n in names)==11
 for name in names:
  assert not name.startswith('/') and '..' not in Path(name).parts
  raw=z.read(name);assert len(raw)<50*1024*1024
  if name.endswith('.npz'):check_npz(raw,name)
  else:
   assert name.endswith(('.json','.md'));s=raw.decode();assert not re.search(r'/(?:vepfs[^/\s]*|peter|root|home)/|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]+|-----BEGIN .*PRIVATE KEY',s)
   if name.endswith('.json'):json.loads(s)
write(RUN/'evidence/attachment_audit.json',dict(command='<EGOSMPLX_ENV>/bin/python audit_attachments.py',exit_status=0,files_sha256=files,npz_files=11,archive_members=len(names),coefficient_keys=sorted(allowed),no_object_arrays=True,no_model_weights_or_mesh_arrays=True,no_images_or_dataset=True,no_machine_paths_credentials=True,authorization='User explicitly requested uploading current and historical fitted SMPL-X parameters; release attachments only, not Git-tracked NPZ'))
print('PASS: 11 compact parameter NPZ files plus ZIP; no model weights or raw output')
