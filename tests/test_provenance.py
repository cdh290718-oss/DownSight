import json

import pytest

from egosmplx_pipeline.provenance import record_run


def test_attempt_success_and_failure_are_separate(tmp_path):
    image = tmp_path / 'image'; image.write_bytes(b'image')
    camera = tmp_path / 'camera'; camera.write_bytes(b'camera')
    rows = [{'id': 'frame', 'image': str(image), 'calibration': str(camera)}]
    config = {'paths': {}, 'secret': 'not-for-publication'}
    with record_run(tmp_path, config, rows, False, False):
        pass
    with pytest.raises(ValueError):
        with record_run(tmp_path, config, rows, True, True):
            raise ValueError('sensitive error details')
    files = list((tmp_path / 'provenance').glob('*.json'))
    assert len(files) == 2
    records = [json.loads(p.read_text()) for p in files]
    assert {r['state'] for r in records} == {'complete', 'failed'}
    failed = next(r for r in records if r['state'] == 'failed')
    assert failed['error_type'] == 'ValueError'
    assert all(r['expanded_config']['secret'] == 'not-for-publication' for r in records)
    assert all('sensitive error details' not in p.read_text() for p in files)
    assert all(len(r['input_sha256']['frame']['image']) == 64 for r in records)


def test_git_state_hashes_contents_and_saves_patch(tmp_path):
    import subprocess
    from egosmplx_pipeline.provenance import git_state
    def git(*args):
        subprocess.run(['git', '-C', str(tmp_path), *args], check=True, stdout=subprocess.DEVNULL)
    git('init', '-q')
    git('config', 'user.email', 'test@example.invalid')
    git('config', 'user.name', 'test')
    tracked = tmp_path / 'tracked.txt'; tracked.write_text('base\n')
    git('add', 'tracked.txt'); git('commit', '-qm', 'initial')
    patch = tmp_path.parent / 'changes.patch'
    tracked.write_text('first\n'); first = git_state(tmp_path, patch)
    assert b'first' in patch.read_bytes()
    tracked.write_text('second\n'); second = git_state(tmp_path, patch)
    assert first['diff_sha256'] != second['diff_sha256']
    new = tmp_path / 'new.txt'; new.write_text('one')
    third = git_state(tmp_path)
    new.write_text('two'); fourth = git_state(tmp_path)
    assert third['untracked_sha256'] != fourth['untracked_sha256']
