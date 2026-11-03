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
    assert all('not-for-publication' not in p.read_text() for p in files)
    assert all('sensitive error details' not in p.read_text() for p in files)
    assert all(len(r['input_sha256']['frame']['image']) == 64 for r in records)
