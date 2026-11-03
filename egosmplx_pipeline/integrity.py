"""Verify the declared backend and actual runtime resource contents."""
import json
import os
from pathlib import Path
import re
from egosmplx_pipeline.io import sha, write

ROOT = Path(__file__).resolve().parent.parent
CONTRACT = ROOT / 'docs/backend_contract.json'
CACHE = ROOT / '.local/sha256-cache.json'
RESOURCE_KEYS = {'egosmplx': 'egosmplx_checkpoint', 'pose': 'pose_checkpoint',
                 'seg': 'seg_checkpoint', 'wilor': 'wilor_checkpoint',
                 'detector': 'wilor_detector', 'egosmplx_config': 'egosmplx_config'}


def cached_sha(path, cache_path=CACHE):
    path = Path(path).resolve(strict=True)
    stat = path.stat()
    key = json.dumps([str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns,
                      stat.st_dev, stat.st_ino])
    try:
        cache = json.loads(Path(cache_path).read_text())
    except (FileNotFoundError, ValueError, OSError):
        cache = {}
    if key in cache and re.fullmatch('[a-f0-9]{64}', cache[key]):
        return cache[key]
    result = sha(path)
    after = path.stat()
    if (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns, after.st_ino):
        raise ValueError('Resource changed while hashing: ' + str(path))
    cache[key] = result
    write(cache_path, cache)
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
