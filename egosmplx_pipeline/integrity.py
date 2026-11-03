"""Verify the declared backend and actual runtime resource contents."""
import json
from pathlib import Path
import re
from egosmplx_pipeline.io import sha

ROOT = Path(__file__).resolve().parent.parent
CONTRACT = ROOT / 'docs/backend_contract.json'
RESOURCE_KEYS = {'egosmplx': 'egosmplx_checkpoint', 'pose': 'pose_checkpoint',
                 'seg': 'seg_checkpoint', 'wilor': 'wilor_checkpoint',
                 'detector': 'wilor_detector', 'egosmplx_config': 'egosmplx_config'}


def cached_sha(path, cache_path=None):
    # Always hash content: some filesystems do not reliably advance metadata.
    path = Path(path).resolve(strict=True)
    before = path.stat()
    result = sha(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns, after.st_ino):
        raise ValueError('Resource changed while hashing: ' + str(path))
    return result


def verify_resources(config):
    result = {}
    for name, key in RESOURCE_KEYS.items():
        path = config.get('paths', {}).get(key)
        if not path or not Path(path).is_file():
            raise FileNotFoundError('Required resource: ' + key)
        actual = cached_sha(path)
        expected = config.get('checkpoint_sha256', {}).get(name)
        if expected is not None and (not re.fullmatch('[a-f0-9]{64}', expected) or expected != actual):
            raise ValueError('SHA-256 mismatch for ' + name)
        result[name] = actual
    return result


def verify_backend(repo, contract_path=CONTRACT):
    contract_path = Path(contract_path)
    contract = json.loads(contract_path.read_text())
    result = {}
    for entry in contract['verified_files']:
        rel = Path(entry['path'])
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError('Invalid backend contract path')
        file = Path(repo) / rel
        if not file.is_file():
            raise FileNotFoundError('Backend contract missing: ' + str(rel))
        actual = cached_sha(file)
        if actual != entry['sha256']:
            raise ValueError('Backend contract mismatch: ' + str(rel))
        result[str(rel)] = actual
    return {'contract_sha256': sha(contract_path), 'files_sha256': result}
