"""Verify the integrity checker rejects a corrupted resource without changing files."""
import importlib.util
from pathlib import Path
from unittest.mock import patch

root = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('golden_d_checker', root / 'tools/check_golden_d.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
original = Path.read_bytes
victim = root / 'runs/2026-11-09_initialization-ablation/fit.py'


def changed(path):
    data = original(path)
    return data + b'altered' if path == victim else data

with patch.object(Path, 'read_bytes', changed):
    try:
        module.main()
    except ValueError as exc:
        assert 'resource mismatch' in str(exc)
        print('PASS: modified source bytes rejected without editing workspace files')
    else:
        raise AssertionError('Checker accepted modified source')
