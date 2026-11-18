"""Publish only audited fitted OBJ archive using existing Git authentication.

The GitHub connector has no Release mutation/upload tool. Credentials remain in
process memory and are never printed or persisted. No external packages needed.
"""
import json,subprocess,os,urllib.request,urllib.error,urllib.parse,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];RUN=Path(__file__).resolve().parent;OUT=ROOT/'outputs/real001_obj_rmse_20261118';REPO='cdh290718-oss/DownSight';TAG='real001-obj-rmse-20261118';API='https://api.github.com/repos/'+REPO
read=lambda p:json.loads(p.read_text())
r=subprocess.run(['git','credential','fill'],input='protocol=https\nhost=github.com\n\n',capture_output=True,text=True,env={**os.environ,'GIT_TERMINAL_PROMPT':'0'});cred=dict(line.split('=',1) for line in r.stdout.splitlines() if '=' in line);assert r.returncode==0 and cred.get('password'),'Existing Git authentication unavailable'
token=cred['password']
def request(url,method='GET',body=None,binary=False):
 assert urllib.parse.urlparse(url).hostname in ['api.github.com','uploads.github.com']
 data=body if binary else json.dumps(body,ensure_ascii=False).encode() if body is not None else None
 req=urllib.request.Request(url,data=data,method=method,headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json','Content-Type':'application/octet-stream' if binary else 'application/json','X-GitHub-Api-Version':'2022-11-28'})
 with urllib.request.urlopen(req,timeout=60) as response:return json.load(response)
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/main'],cwd=ROOT,text=True).split()[0];assert head==remote,'Push audited run before publishing attachments'
# Audit validates exactly the archive hash to upload.
audit=read(RUN/'evidence/attachment_audit.json');exports=read(RUN/'evidence/exports.json');files=[OUT/exports['archive']['file']]
assert {p.name for p in files}==set(audit['files_sha256'])
for p in files:assert hashlib.sha256(p.read_bytes()).hexdigest()==audit['files_sha256'][p.name]
statepath=OUT/'release_state.json'
if statepath.exists():release=request(API+'/releases/'+str(read(statepath)['id']));assert release['tag_name']==TAG and release['draft']
else:
 try:request(API+'/releases/tags/'+TAG)
 except urllib.error.HTTPError as e:
  if e.code!=404:raise
 else:raise RuntimeError('Tag already published; refusing to overwrite')
 body='真实样本001、29秒：四模型已拟合 SMPL-X 的 OBJ 网格。ZIP 内含四份 cam3 主坐标网格和相同四个人体的 cam4 坐标版本，以及相机信息与说明。无 MANO 手部拼接。\n\n[自动点二维 RMSE 报告](https://github.com/'+REPO+'/blob/'+head+'/runs/'+RUN.name+'/report.md)\n\nRMSE 对照各模型自身自动点，不是真值误差。'
 release=request(API+'/releases',method='POST',body={'tag_name':TAG,'target_commitish':head,'name':'真实样本001：四模型 OBJ 与自动点 RMSE','body':body,'draft':True,'prerelease':False});statepath.write_text(json.dumps({'id':release['id'],'tag_name':TAG,'commit':head}))
existing={a['name']:a for a in release['assets']};uploaded=[]
for p in files:
 if p.name in existing:
  a=existing[p.name];assert a['size']==p.stat().st_size
 else:a=request(release['upload_url'].split('{')[0]+'?name='+urllib.parse.quote(p.name),method='POST',body=p.read_bytes(),binary=True)
 uploaded.append({k:a[k] for k in ['name','size','browser_download_url']});print('Uploaded',p.name,flush=True)
release=request(API+'/releases/'+str(release['id']),method='PATCH',body={'draft':False});assert not release['draft'];(OUT/'release_published.json').write_text(json.dumps({'id':release['id'],'url':release['html_url'],'tag':TAG,'target_commit':head,'assets':uploaded},indent=2));print('Published',release['html_url'],flush=True)
