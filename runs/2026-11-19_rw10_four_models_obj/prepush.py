"""Audit only the intended public increment; never stage unrelated work."""
import ast,json,re,subprocess,sys
from pathlib import Path
import hashlib
RUN=Path(__file__).resolve().parent;ROOT=RUN.parents[1];EVIDENCE=RUN/'evidence'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
BASELINE='34ff79ba1ec199db8a48803935a7f15e15810aa5'

def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()

def main():
    paths=sorted(p for p in RUN.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='prepush.json')
    paths.append(ROOT/'runs/2026-11-18_rw20000_frozen200/report.md')
    files=[]
    # Construct patterns to avoid the audit source matching its own literals.
    banned_paths=['/'+s+'/' for s in ['root','peter','home','vepfs-mlp2','tmp']]
    secrets=[r'gh'+'[pousr]_[A-Za-z0-9]{20,}',r'github'+'_pat_[A-Za-z0-9_]{20,}',r'-----BEGIN '+'(?:RSA |OPENSSH )?PRIVATE KEY-----']
    for p in paths:
        assert p.suffix in ['.py','.md','.json','.txt'],p
        assert p.stat().st_size<50*1024*1024,p
        assert p.name!='.env' and 'provenance' not in p.parts,p
        text=p.read_text();assert not any(s in text for s in banned_paths),p
        assert not any(re.search(s,text) for s in secrets),p
        if p.suffix=='.py':ast.parse(text)
        if p.suffix=='.json':json.loads(text)
        files.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)))
    expected={str(p.relative_to(ROOT)) for p in paths}|{str((EVIDENCE/'prepush.json').relative_to(ROOT))}
    staged=set(git('diff','--cached','--name-only').splitlines());assert staged<=expected,(staged-expected)
    remote=git('ls-remote','origin','refs/heads/main').split()[0]
    assert remote==BASELINE,'Remote changed: inspect before pushing'
    head=git('rev-parse','HEAD')
    if head!=BASELINE:
        assert git('rev-list','--count',BASELINE+'..HEAD')=='1'
        pending=set(git('diff','--name-only',BASELINE,'HEAD').splitlines());assert pending<=expected,pending-expected
        # Inspect the actual pending committed blobs as well as working files.
        for path in pending:
            content=git('show',head+':'+path)
            assert not any(s in content for s in banned_paths),path
            assert not any(re.search(s,content) for s in secrets),path
    else:pending=set()
    status=git('status','--porcelain').splitlines()
    audit=dict(scope='Current run and original report update only; unrelated working tree and historical files are not published or certified',baseline=BASELINE,remote_before_push=remote,worktree_reviewed=True,staged_reviewed=True,pending_commits_reviewed=True,unrelated_changes_preserved=True,checks=dict(text_only=True,under_50MB_per_file=True,no_weights_datasets_video_npy_npz_full_outputs=True,no_credentials_env_machine_paths=True,no_full_provenance_patch=True,syntax_and_json_valid=True),files=files,staged_paths=sorted(staged),pending_commit_paths=sorted(pending),worktree_entries=len(status),verification_exit_status=read(EVIDENCE/'attachment_audit.json')['exit_status'])
    if '--check-only' not in sys.argv:write(EVIDENCE/'prepush.json',audit)
    print('Public increment audit passed:',len(files),'files')

if __name__=='__main__':main()
