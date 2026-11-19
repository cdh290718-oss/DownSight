"""Publish the pre-audited OBJ ZIP on a new GitHub release after checking remote HEAD."""
import hashlib
import json
import os
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

run=Path(__file__).resolve().parent
root=run.parents[1]
out=root/'outputs/rw_landmarks_head_views_20261119'
audit=json.loads((run/'evidence/attachment_audit.json').read_text())
tag=audit['tag']
repo='cdh290718-oss/DownSight'
api='https://api.github.com/repos/'+repo
archive=out/audit['archive']
digest=audit['files_sha256'][archive.name]
assert hashlib.sha256(archive.read_bytes()).hexdigest()==digest
head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
assert subprocess.check_output(['git','ls-remote','origin','refs/heads/main'],text=True).split()[0]==head
proc=subprocess.run(['git','credential','fill'],input='protocol=https\nhost=github.com\n\n',
                    text=True,capture_output=True,check=True,env={**os.environ,'GIT_TERMINAL_PROMPT':'0'})
credential=dict(s.split('=',1) for s in proc.stdout.splitlines() if '=' in s)


def request(url,method='GET',payload=None):
    import urllib.parse
    assert urllib.parse.urlparse(url).hostname in ['api.github.com','uploads.github.com']
    binary=isinstance(payload,bytes)
    data=payload if binary else (json.dumps(payload).encode() if payload is not None else None)
    headers={'Authorization':'Bearer '+credential['password'],'Accept':'application/vnd.github+json',
             'Content-Type':'application/octet-stream' if binary else 'application/json',
             'User-Agent':'DownSight-release'}
    req=urllib.request.Request(url,data=data,headers=headers,method=method)
    with urllib.request.urlopen(req,timeout=90) as result:
        return json.load(result)


state=out/'release_state.json'
if state.exists():
    release=request(api+'/releases/'+str(json.loads(state.read_text())['id']))
else:
    try:
        request(api+'/releases/tags/'+tag)
    except urllib.error.HTTPError as exc:
        if exc.code!=404: raise
    else:
        raise RuntimeError('Existing published tag; refusing overwrite')
    release=request(api+'/releases','POST',dict(tag_name=tag,target_commitish=head,
          name='RW 四姿态：身体点与头朝向 OBJ 对照',
          body='四对 RW 双目样本，左相机 CV 米制 OBJ，条件和图片见[报告](https://github.com/'+repo+'/blob/'+head+'/runs/'+run.name+'/report.md)。',
          draft=True,prerelease=False))
    state.write_text(json.dumps({'id':release['id'],'commit':head}))
assert release['tag_name']==tag and release['target_commitish']==head
assets={a['name']:a for a in release['assets']}
if archive.name not in assets:
    assert release['draft']
    asset=request(release['upload_url'].split('{')[0]+'?name='+archive.name,'POST',archive.read_bytes())
else:
    asset=assets[archive.name]
assert asset['size']==archive.stat().st_size and asset['state']=='uploaded'
if asset.get('digest'):
    assert asset['digest']=='sha256:'+digest
if release['draft']:
    release=request(api+'/releases/'+str(release['id']),'PATCH',dict(draft=False))
assert not release['draft']
public=asset['browser_download_url']
# Verify the unauthenticated public attachment, independent of the authenticated upload.
wrong='Unexpected public archive SHA-256'
h=hashlib.sha256()
with urllib.request.urlopen(urllib.request.Request(public,headers={'User-Agent':'DownSight-verify'}),timeout=60) as res:
    for chunk in iter(lambda:res.read(1048576),b''): h.update(chunk)
assert h.hexdigest()==digest,wrong
receipt=dict(tag=tag,url=release['html_url'],archive_url=public,sha256=digest,commit=head)
(out/'release_verified.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
