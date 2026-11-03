import hashlib
import json
import os
from pathlib import Path
import pytest
from egosmplx_pipeline import integrity


def test_resources_verify_actual_hashes_and_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(integrity, 'cached_sha', lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest())
    paths = {}; expected = {}
    for name, key in integrity.RESOURCE_KEYS.items():
        p = tmp_path / name; p.write_bytes(name.encode()); paths[key] = str(p)
        expected[name] = hashlib.sha256(name.encode()).hexdigest()
    config = {'paths': paths, 'checkpoint_sha256': expected}
    assert integrity.verify_resources(config) == expected
    (tmp_path / 'egosmplx').write_bytes(b'different')
    with pytest.raises(ValueError, match='egosmplx'):
        integrity.verify_resources(config)


def test_backend_contract_is_single_source_of_truth(tmp_path, monkeypatch):
    monkeypatch.setattr(integrity, 'cached_sha', lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest())
    backend = tmp_path / 'backend'; backend.mkdir()
    (backend / 'one.py').write_bytes(b'one')
    contract = tmp_path / 'contract.json'
    contract.write_text(json.dumps({'verified_files': [{'path': 'one.py', 'sha256': hashlib.sha256(b'one').hexdigest()}]}))
    assert 'one.py' in integrity.verify_backend(backend, contract)['files_sha256']
    (backend / 'one.py').write_bytes(b'two')
    with pytest.raises(ValueError, match='contract mismatch'):
        integrity.verify_backend(backend, contract)


def test_cache_recomputes_after_change(tmp_path):
    p = tmp_path / 'resource'; cache = tmp_path / 'cache.json'
    p.write_bytes(b'old'); first = integrity.cached_sha(p, cache)
    p.write_bytes(b'new'); os.utime(p, ns=(p.stat().st_atime_ns, p.stat().st_mtime_ns + 1000000000)); second = integrity.cached_sha(p, cache)
    assert first != second
