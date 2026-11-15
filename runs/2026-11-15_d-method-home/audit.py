"""Audit only the explicitly selected public increment."""
import json
import re
import subprocess
from pathlib import Path
root = Path(__file__).resolve().parents[2]
run = Path(__file__).resolve().parent
base = '8a245314d975c7ec1ec16b5ca08c35503254680a'
def git(*args):
    return subprocess.check_output(['git', *args], cwd=root, text=True).strip()
mode = 'committed' if git('rev-parse', 'HEAD') != base else 'staged'
files = git('diff', '--name-only', base, 'HEAD').splitlines() if mode == 'committed' else git('diff', '--cached', '--name-only').splitlines()
errors = []
rows = []
patterns = [r'gh[pousr]_[A-Za-z0-9]{20,}', r'github_pat_[A-Za-z0-9_]{20,}', '-----' + r'BEGIN .*PRIVATE KEY' + '-----', '/' + 'vepfs-', '/' + 'home/', '/' + 'root/']
for name in files:
    assert name == 'README.md' or name.startswith('runs/2026-11-15_d-method-home/'), name
    content = subprocess.check_output(['git', 'show', ('HEAD:' if mode == 'committed' else ':')+name], cwd=root)
    p = root / name
    if content != p.read_bytes(): errors.append(name+': working tree differs')
    if len(content) >= 50_000_000: errors.append(name+': size')
    if p.suffix not in {'.md', '.py', '.json'} or p.name == '.env': errors.append(name+': type')
    if any(re.search(pattern, content.decode()) for pattern in patterns): errors.append(name+': sensitive content')
    rows.append({'path': name, 'bytes': len(content)})
remote = git('ls-remote', 'origin', 'refs/heads/main').split()[0]
if remote != base: errors.append('remote changed')
check = subprocess.run(['git', 'diff', '--cached', '--check'], cwd=root, capture_output=True, text=True)
if check.returncode: errors.append('staged diff check failed')
result = {'mode': mode, 'base': base, 'remote_head': remote, 'files': rows, 'workspace_status': git('status','--short'), 'errors': errors, 'manual_review': 'Only documentation, small check scripts and redacted evidence; no model/data/video/full provenance or patch. Existing unrelated untracked files excluded.', 'exit_status': 1 if errors else 0}
if mode == 'staged':
    (run/'evidence/prepush.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(result, ensure_ascii=False, indent=2))
assert files and not errors
