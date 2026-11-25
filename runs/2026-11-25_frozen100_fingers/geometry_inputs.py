from pathlib import Path
import sys
RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[1]
OUT=ROOT/'outputs/frozen100_fingers_20261125'
EV=RUN/'evidence'
PREV=ROOT/'runs/2026-11-25_frozen100_outline'
sys.path.insert(0,str(PREV))
from compare import GridEye,from_branch,fit_probe
from shared_run import prev,search,Frame,WEIGHT,guard,OUT as PRIOR
from outline import NAMES,STRIDE,boundary,MAX_PX,SIGMA_PX
