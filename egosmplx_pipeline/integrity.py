"""Verify the declared backend and actual runtime resource contents."""
import hashlib
import json
import os
from pathlib import Path
import re
from egosmplx_pipeline.io import sha

ROOT = Path(__file__).resolve().parent.parent
CONTRACT = ROOT / 'docs/backend_contract.json'
RESOURCE_KEYS = {'egosmplx': 'egosmplx_checkpoint', 'pose': 'pose_checkpoint',
                 'seg': 'seg_checkpoint', 'wilor': 'wilor_checkpoint',
                 'detector': 'wilor_detector', 'egosmplx_config': 'egosmplx_config'}


def file_sha256(path):
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
        expected = config.get('checkpoint_sha256', {}).get(name)
        if not isinstance(expected, str) or not re.fullmatch('[a-f0-9]{64}', expected):
            raise ValueError('Missing or invalid expected SHA-256 for ' + name)
        actual = file_sha256(path)
        if expected != actual:
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
        actual = file_sha256(file)
        if actual != entry['sha256']:
            raise ValueError('Backend contract mismatch: ' + str(rel))
        result[str(rel)] = actual
    return {'contract_sha256': sha(contract_path), 'files_sha256': result}


SOURCE_EXTENSIONS = {'.py', '.pyi', '.sh', '.bash', '.c', '.cc', '.cpp', '.cu', '.h', '.hpp', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.json', '.txt', '.xml', '.cmake', '.js', '.ts', '.jsx', '.tsx', '.css', '.html'}
# Exclude generated outputs, model assets, input data and separately deployed third-party environments.
EXCLUDED_TOP = {'output', 'outputs', 'experiments', 'pretrained_models', '.runtime', 'official_baseline', 'generated_meshes', 'inference_outputs', 'inference_inputs', 'logs', 'assets', 'gt', '0000', 'EgoPW_sample', 'EgoPW_test', 'Xperience-10M', 'manual_annotations', 'my_pictures', 'session_hand_6'}
EXCLUDED_DIRS = {'.git', '__pycache__', '.pytest_cache', 'node_modules', 'site-packages', 'human_model_files', 'prepared', 'raw', 'images', 'annotations', 'pseudo-GT', 'egodex_smplx_params'}


def backend_source_manifest(repo):
    root = Path(repo).resolve(strict=True)
    files = {}
    for current, dirs, names in os.walk(root, followlinks=False):
        base = Path(current)
        rel_base = base.relative_to(root)
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDED_DIRS and not (rel_base == Path('.') and (d in EXCLUDED_TOP or d.startswith('.tmp') or d.startswith('DownwardStereo') or d.startswith('beta_'))))
        for name in sorted(names):
            p = base / name
            if p.suffix.lower() not in SOURCE_EXTENSIONS and p.name not in {'Makefile', 'CMakeLists.txt', 'Dockerfile'}:
                continue
            if p.suffix.lower() in {'.json', '.txt', '.xml'} and rel_base.parts and rel_base.parts[0] == 'data':
                continue
            rel = p.relative_to(root).as_posix()
            if p.is_symlink():
                files[rel] = {'symlink_target_sha256': hashlib.sha256(os.fsencode(os.readlink(p))).hexdigest(), 'sha256': file_sha256(p) if p.is_file() else None}
            elif p.is_file():
                files[rel] = {'sha256': file_sha256(p)}
        for name in list(dirs):
            p = base / name
            if p.is_symlink():
                files[p.relative_to(root).as_posix() + '/'] = {'symlink_target_sha256': hashlib.sha256(os.fsencode(os.readlink(p))).hexdigest(), 'contents_scanned': False}
                dirs.remove(name)
    encoded = json.dumps(files, sort_keys=True, separators=(',', ':')).encode()
    return {'algorithm': 'sha256', 'scope': 'source extensions and non-data config/text; generated, assets, external environments excluded; directory symlinks not followed',
            'files': files, 'file_count': len(files), 'tree_sha256': hashlib.sha256(encoded).hexdigest(),
            'excluded_top': sorted(EXCLUDED_TOP), 'excluded_dirs': sorted(EXCLUDED_DIRS), 'extensions': sorted(SOURCE_EXTENSIONS)}
