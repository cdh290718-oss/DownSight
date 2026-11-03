"""Record a run attempt inside its output directory, without changing fit behavior."""
import contextlib
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import uuid


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def git_state(root, patch_path=None):
    def git(*args):
        p = subprocess.run(['git', '-C', str(root), *args], stdout=subprocess.PIPE,
                           stderr=subprocess.DEVNULL, check=False)
        return p.stdout if p.returncode == 0 else None
    repo_root = git('rev-parse', '--show-toplevel')
    if repo_root is None:
        return {'available': False, 'reason': 'not a git repository'}
    repo_root = Path(os.fsdecode(repo_root.rstrip(b'\n')))
    patch = git('diff', '--binary', '--no-ext-diff', 'HEAD', '--')
    raw = git('ls-files', '--others', '--exclude-standard', '-z', '--full-name')
    if patch is None or raw is None:
        raise RuntimeError('Cannot capture git changes')
    if patch_path is not None:
        Path(patch_path).write_bytes(patch)
    untracked = {}
    for item in raw.split(b'\0'):
        if not item:
            continue
        rel = os.fsdecode(item)
        file = repo_root / rel
        if file.is_symlink():
            untracked[rel] = {'type': 'symlink', 'sha256': hashlib.sha256(os.fsencode(os.readlink(file))).hexdigest()}
        elif file.is_file():
            untracked[rel] = {'type': 'file', 'sha256': digest(file)}
    commit = git('rev-parse', 'HEAD')
    payload = json.dumps(untracked, sort_keys=True).encode()
    return {'available': True, 'commit': commit.decode().strip() if commit else None,
            'dirty': bool(patch or untracked), 'diff_sha256': hashlib.sha256(patch).hexdigest(),
            'untracked': untracked, 'untracked_sha256': hashlib.sha256(payload).hexdigest()}


def snapshot(config, rows, cached, resume, integrity, folder, run_id):
    # Full config and patches remain in the local output; sanitize before publishing.
    encoded = json.dumps(config, sort_keys=True, default=str).encode()
    root = Path(__file__).resolve().parent.parent
    code = {str(p.relative_to(root)): digest(p) for p in sorted((root / 'egosmplx_pipeline').rglob('*.py'))}
    backend_value = config.get('paths', {}).get('egosmplx_repo')
    backend = Path(backend_value) if backend_value else None
    inputs = {}
    for row in rows:
        inputs[row['id']] = {key: digest(row[key]) for key in ('image', 'calibration')}
    return {'schema_version': 1, 'run_id': run_id,
            'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'state': 'running', 'cached_predictions': cached, 'resume': resume,
            'config_sha256': hashlib.sha256(encoded).hexdigest(), 'expanded_config': config,
            'pipeline_git': git_state(root, folder / (run_id + '.pipeline.patch')), 'pipeline_files_sha256': code,
            'backend_git': git_state(backend, folder / (run_id + '.backend.patch')) if backend is not None and backend.is_dir() else None,
            'integrity': integrity, 'input_sha256': inputs,
            'configured_checkpoint_sha256': config.get('checkpoint_sha256', {}),
            'python': sys.version, 'platform': platform.platform(),
            'note': 'Full expanded configuration and patches are local only; sanitize before sharing.'}


def save(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    os.replace(str(temp), str(path))


@contextlib.contextmanager
def record_run(output, config, rows, cached, resume, integrity=None):
    folder = Path(output) / 'provenance'
    folder.mkdir(exist_ok=True)
    run_id = uuid.uuid4().hex
    data = snapshot(config, rows, cached, resume, integrity, folder, run_id)
    path = folder / (run_id + '.json')
    save(path, data)
    try:
        yield data
    except BaseException as exc:
        data['state'] = 'failed'
        data['error_type'] = type(exc).__name__
        raise
    else:
        data['state'] = 'complete'
    finally:
        data['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save(path, data)
