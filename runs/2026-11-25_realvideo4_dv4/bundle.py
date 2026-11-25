"""Package only the four verified meshes in cam3 coordinates."""
import hashlib
import json
from pathlib import Path
import zipfile
RUN=Path(__file__).resolve().parent
OUT=RUN.parents[1]/'outputs/realvideo4_dv4_20261125/export'
cases=json.loads((RUN/'evidence/results.json').read_text())['cases']
assert len(cases)==4
files={}
archive=OUT/'realvideo4_Dv4_OBJ_cam3_m.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for key,rec in sorted(cases.items()):
        meta=rec['obj']
        path=OUT/meta['name']
        assert path.name==key+'_dv4_cam3_m.obj'
        data=path.read_bytes()
        assert len(data)<50_000_000 and hashlib.sha256(data).hexdigest()==meta['sha256']
        lines=data.decode('ascii').splitlines()
        assert lines[0]=='# '+key+' Golden D v4, cam3 CV, meters'
        assert len(lines)==1+meta['vertices']+meta['faces']
        assert all(line.startswith('v ') for line in lines[1:1+meta['vertices']])
        assert all(line.startswith('f ') for line in lines[1+meta['vertices']:])
        z.writestr(path.name,data)
        files[path.name]=meta['sha256']
with zipfile.ZipFile(archive) as z:
    assert set(z.namelist())==set(files) and z.testzip() is None
    assert all(hashlib.sha256(z.read(name)).hexdigest()==digest for name,digest in files.items())
files[archive.name]=hashlib.sha256(archive.read_bytes()).hexdigest()
(RUN/'evidence/attachment_audit.json').write_text(json.dumps(dict(files_sha256=files,units='meters',frame='cam3 OpenCV',exit_status=0),indent=2)+'\n')
