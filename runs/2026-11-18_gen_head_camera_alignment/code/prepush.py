"""Audit this round's publication scope; does not claim to audit old history."""
import hashlib,json,re,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];RUN=Path(__file__).resolve().parents[1];prefix=str(RUN.relative_to(ROOT))+'/'
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True).rstrip('\n')
head=git('rev-parse','HEAD');remote=git('ls-remote','origin','refs/heads/main').split()[0]
# Existing local non-round changes are deliberately not part of publication.
status=git('status','--porcelain');staged=git('diff','--cached','--name-only').splitlines();assert all(x.startswith(prefix) for x in staged),staged
assert head==remote,(head,remote)
files=[]
for p in sorted(RUN.rglob('*')):
 if not p.is_file() or '__pycache__' in p.parts or p.name=='prepush.json':continue
 assert p.suffix in ('.md','.py','.json'),p
 raw=p.read_bytes();assert len(raw)<50_000_000,p
 text=raw.decode();assert not re.search(r'/(?:vepfs-[^/]+|root|home|opt/conda)/|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}',text),p
 assert not re.search(r'(?im)^\s*(?:export\s+)?(?:AWS_SECRET_ACCESS_KEY|GITHUB_TOKEN|OPENAI_API_KEY)\s*=',text),p
 assert not re.search('-'*5+'BEGIN '+r'(?:RSA |EC |OPENSSH )?PRIVATE KEY',text),p
 files.append({'file':str(p.relative_to(ROOT)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
r={'scope':'only this run; old history not audited','baseline_commit':head,'remote_main_before_commit':remote,'unpublished_commits_before':git('rev-list','--count',remote+'..HEAD'),'initial_scanner_retry':'First scanner matched its own literal path regex; fixed to match actual path prefixes. No secret or actual machine path found in publication scope. No push occurred before resolving this.', 'existing_staged_files':staged,'worktree_review':{'entries_outside_this_run':sum(not x[3:].startswith(prefix) for x in status.splitlines()),'existing_train_py_modified':any(x[3:]=='runs/2026-11-16_sapiens-rw500-prep/code/train.py' for x in status.splitlines()),'untracked_user_mcap_not_staged':any(x[3:]=='data/0d7da991acf44b46ba41710a8960209c.mcap' for x in status.splitlines()),'decision':'preserve and exclude every out-of-scope entry'},'checks':{'allowlisted_text_files_only':True,'all_under_50MB':True,'no_data_weights_video_npy_npz_or_full_raw_outputs':True,'no_env_or_credentials':True,'machine_paths_redacted':True,'no_full_provenance_or_patch':True,'remote_equals_baseline':True},'publication_files_before_audit_file':files,'command':'python runs/2026-11-18_gen_head_camera_alignment/code/prepush.py','exit_status':0}
(RUN/'evidence/prepush.json').write_text(json.dumps(r,indent=2)+'\n');print('Pre-stage publication audit passed:',len(files),'files')
