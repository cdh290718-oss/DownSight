"""Bind the frozen contour implementation to the validation split and its own output directory."""
import importlib.util
from pathlib import Path
import sys

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
OUT = ROOT / 'outputs/validation200_fingers_20261125'
EV = RUN / 'evidence'
FINGERS = RUN.parent / '2026-11-25_frozen100_fingers'
sys.path.insert(0, str(FINGERS))
from geometry_inputs import GridEye, from_branch, fit_probe, prev, search, guard, WEIGHT
from hand_edges import MaskedEye, HANDS
import compare

spec = importlib.util.spec_from_file_location('locked_fingers', FINGERS / 'fit.py')
locked = importlib.util.module_from_spec(spec)
spec.loader.exec_module(locked)
EdgeFrame = locked.EdgeFrame
compare.OUT = OUT
fit_probe.Eye = GridEye
prev.base.fit.camera_files = prev.ref.camera_files
