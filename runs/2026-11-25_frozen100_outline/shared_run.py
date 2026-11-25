from pathlib import Path
import sys
RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[1]
OUT=ROOT/'outputs/frozen100_outline_20261125'
EV=RUN/'evidence'
sys.path.insert(0,str(ROOT/'runs/2026-11-24_two_stereo_contact'))
from common_run import prev,search
from fit_probe import Frame,WEIGHT
from repair_local import guard
from outline import Eye,NAMES,STRIDE,boundary
from inspect_obj import groups
from surface import crossings
import candidates as wrist
