import hashlib
import json
from pathlib import Path
import pytest
from egosmplx_pipeline import integrity


def test_resources_verify_actual_hashes_and_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(integrity, 'file_sha256', lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest())
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
    monkeypatch.setattr(integrity, 'file_sha256', lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest())
    backend = tmp_path / 'backend'; backend.mkdir()
    (backend / 'one.py').write_bytes(b'one')
    contract = tmp_path / 'contract.json'
    contract.write_text(json.dumps({'verified_files': [{'path': 'one.py', 'sha256': hashlib.sha256(b'one').hexdigest()}]}))
    assert 'one.py' in integrity.verify_backend(backend, contract)['files_sha256']
    (backend / 'one.py').write_bytes(b'two')
    with pytest.raises(ValueError, match='contract mismatch'):
        integrity.verify_backend(backend, contract)


def test_file_sha256_recomputes_after_change(tmp_path):
    p = tmp_path / 'resource'
    p.write_bytes(b'old'); first = integrity.file_sha256(p)
    p.write_bytes(b'new'); second = integrity.file_sha256(p)
    assert first != second


def test_preflight_rejects_weight_mismatch(tmp_path, monkeypatch):
    from egosmplx_pipeline.configuration import preflight
    config = {'paths': {'body_python': str(tmp_path), 'egosmplx_repo': str(tmp_path),
                        'egosmplx_config': str(tmp_path)}}
    def fail(_):
        raise ValueError('SHA-256 mismatch for egosmplx')
    monkeypatch.setattr(integrity, 'verify_resources', fail)
    with pytest.raises(ValueError, match='SHA-256 mismatch'):
        preflight(config, [], True)


def test_resume_fingerprint_changes_with_resource_or_contract(tmp_path, monkeypatch):
    from egosmplx_pipeline import pipeline
    (tmp_path / 'egosmplx_pipeline').mkdir()
    monkeypatch.setattr(pipeline, 'PACKAGE_ROOT', tmp_path)
    first = pipeline.fingerprint({}, [], {'resources': {'egosmplx': 'first'}, 'backend': {'contract_sha256': 'a'}})
    weight = pipeline.fingerprint({}, [], {'resources': {'egosmplx': 'second'}, 'backend': {'contract_sha256': 'a'}})
    contract = pipeline.fingerprint({}, [], {'resources': {'egosmplx': 'first'}, 'backend': {'contract_sha256': 'b'}})
    assert first != weight and first != contract


def test_missing_expected_hash_rejected(tmp_path, monkeypatch):
    paths = {}
    for name, key in integrity.RESOURCE_KEYS.items():
        p = tmp_path / name; p.write_text(name); paths[key] = str(p)
    with pytest.raises(ValueError, match='Missing or invalid expected SHA-256'):
        integrity.verify_resources({'paths': paths, 'checkpoint_sha256': {}})


def test_backend_source_manifest_changes_for_unlisted_code(tmp_path):
    root = tmp_path / 'backend'; (root / 'main').mkdir(parents=True)
    p = root / 'main' / 'arbitrary.py'; p.write_text('one')
    first = integrity.backend_source_manifest(root)
    p.write_text('two')
    second = integrity.backend_source_manifest(root)
    assert first['tree_sha256'] != second['tree_sha256']
    assert 'main/arbitrary.py' in second['files']


def test_resume_fingerprint_changes_when_backend_source_tree_changes(tmp_path, monkeypatch):
    from egosmplx_pipeline import pipeline
    package = tmp_path / 'pipeline'; (package / 'egosmplx_pipeline').mkdir(parents=True)
    monkeypatch.setattr(pipeline, 'PACKAGE_ROOT', package)
    root = tmp_path / 'backend'; (root / 'main').mkdir(parents=True)
    source = root / 'main' / 'network.py'; source.write_text('first')
    first = pipeline.fingerprint({}, [], {'backend': {'source_tree': integrity.backend_source_manifest(root)}})
    source.write_text('second')
    second = pipeline.fingerprint({}, [], {'backend': {'source_tree': integrity.backend_source_manifest(root)}})
    assert first != second
