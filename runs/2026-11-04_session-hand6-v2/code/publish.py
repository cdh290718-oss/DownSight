"""Publish a new, audited static directory without replacing any old site file."""
import argparse,base64,hashlib,json,subprocess
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True);p.add_argument('--gh',required=True);p.add_argument('--evidence',type=Path,required=True);p.add_argument('--slug',default='session-hand6-original-v2-20261104');a=p.parse_args();repo='cdh290718-oss/egofishpose-demo';base='repos/'+repo
 def api(path,method='GET',data=None):
  cmd=[a.gh,'api',path,'--method',method]
  if data is not None:cmd+=['--input','-']
  return json.loads(subprocess.check_output(cmd,input=json.dumps(data).encode() if data is not None else None))
 audit=json.loads((a.evidence/'prepush-check.json').read_text())
 if not audit['passed']:raise RuntimeError('Public increment audit has not passed')
 audited={x['path']:x['sha256'] for x in audit['site']}
 current={f.relative_to(a.site).as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in a.site.rglob('*') if f.is_file()}
 if audited!=current:raise RuntimeError('Site changed after public audit')
 branch=api(base+'/git/ref/heads/main');parent=branch['object']['sha'];commit=api(base+'/git/commits/'+parent);tree=commit['tree']['sha']
 root=api(base+'/git/trees/'+tree)
 if any(x['path']==a.slug for x in root['tree']):raise RuntimeError('Slug already exists; will not overwrite')
 entries=[];files={}
 for f in sorted(a.site.rglob('*')):
  if not f.is_file():continue
  raw=f.read_bytes();name=f.relative_to(a.site).as_posix()
  if len(raw)>=50_000_000 or f.suffix not in ('.html','.json','.jpg','.css','.js'):raise ValueError(name)
  if f.suffix!='.jpg':
   text=raw.decode()
   if any(s in text for s in ('/vepfs-','/root/','ghp_','github_pat_','BEGIN PRIVATE KEY')):raise ValueError('Unsafe public text '+name)
  files[name]=hashlib.sha256(raw).hexdigest();blob=api(base+'/git/blobs','POST',dict(content=base64.b64encode(raw).decode(),encoding='base64'))
  entries.append(dict(path=a.slug+'/'+name,mode='100644',type='blob',sha=blob['sha']))
 newtree=api(base+'/git/trees','POST',dict(base_tree=tree,tree=entries))
 c=api(base+'/git/commits','POST',dict(message='Publish fixed five-pair original versus v2 comparison',tree=newtree['sha'],parents=[parent]))
 if api(base+'/git/ref/heads/main')['object']['sha']!=parent:raise RuntimeError('Remote changed before publication; stop without force')
 api(base+'/git/refs/heads/main','PATCH',dict(sha=c['sha'],force=False))
 result=dict(repository=repo,previous_commit=parent,commit=c['sha'],slug=a.slug,url='https://cdh290718-oss.github.io/egofishpose-demo/'+a.slug+'/',files=files,changed_paths=[x['path'] for x in entries],published=True,online_verified=False)
 (a.evidence/'publication.json').write_text(json.dumps(result,indent=2)+'\n');print(result['url'])
if __name__=='__main__':main()
