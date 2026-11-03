import json
from pathlib import Path
import pytest
pytest.importorskip('torch')
from tools.compare_fisheye_cameras import compare


def test_comparison_checks_backprojection_projection_and_runtime_patch(tmp_path):
    cal = tmp_path / 'camera.json'
    cal.write_text(json.dumps({'size': [1280,720], 'image_center':[630,350],
                               'polynomialC2W':[-100.], 'polynomialW2C':[150.,80.,2.]}))
    source = Path(__file__).resolve().parents[1] / 'egosmplx_pipeline/camera.py'
    result = compare(source, cal)
    assert result['pixel_samples'] == 187
    assert result['projection_numpy_max_px'] == 0
    assert result['backprojection_numpy_max_m'] == 0
    assert result['projection_patch_vs_numpy_max_px'] < 0.001
