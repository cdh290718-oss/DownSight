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


def git_state(root):
    def git(*args):
        p = subprocess.run(['git', '-C', str(root), *args], stdout=subprocess.PIPE,
                           stderr=subprocess.DEVNULL, text=True, check=False)
        return p.stdout.strip() if p.returncode == 0 else None
    status = git('status', '--porcelain=v1', '--untracked-files=normal')
    return {'commit': git('rev-parse', 'HEAD'), 'dirty': bool(status) if status is not None else None,
            'status_sha256': hashlib.sha256(status.encode()).hexdigest() if status is not None else None}


def snapshot(config, rows, cached, resume):
    # Store a digest of the complete expanded config instead of publishing local paths or secrets.
    encoded = json.dumps(config, sort_keys=True, default=str).encode()
    root = Path(__file__).resolve().parent.parent
    code = {str(p.relative_to(root)): digest(p) for p in sorted((root / 'egosmplx_pipeline').rglob('*.py'))}
    backend_value = config.get('paths', {}).get('egosmplx_repo')
    backend = Path(backend_value) if backend_value else None
    backend_files = {}
    for name in ['main/base.py', 'common/utils/human_models.py']:
        p = backend / name if backend else None
        if p is not None and p.is_file():
            backend_files[name] = digest(p)
    inputs = {}
    for row in rows:
        inputs[row['id']] = {key: digest(row[key]) for key in ('image', 'calibration')}
    return {'schema_version': 1, 'run_id': uuid.uuid4().hex,
            'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'state': 'running', 'cached_predictions': cached, 'resume': resume,
            'config_sha256': hashlib.sha256(encoded).hexdigest(),
            'pipeline_git': git_state(root), 'pipeline_files_sha256': code,
            'backend_git': git_state(backend) if backend is not None and backend.is_dir() else None,
            'backend_selected_files_sha256': backend_files, 'input_sha256': inputs,
            'configured_checkpoint_sha256': config.get('checkpoint_sha256', {}),
            'python': sys.version, 'platform': platform.platform(),
            'note': 'Checkpoint digests are configured values; backend and environment coverage is partial.'}


def save(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    os.replace(str(temp), str(path))


@contextlib.contextmanager
def record_run(output, config, rows, cached, resume):
    data = snapshot(config, rows, cached, resume)
    folder = Path(output) / 'provenance'
    folder.mkdir(exist_ok=True)
    path = folder / (data['run_id'] + '.json')
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
