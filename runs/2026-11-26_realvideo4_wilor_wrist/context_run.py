from pathlib import Path
import sys

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
PREVIOUS = RUN.parent/'2026-11-26_realvideo4_wilor_outline'
OUT = ROOT/'outputs/realvideo4_wilor_wrist_20261126'
sys.path.insert(0, str(PREVIOUS))
import paired
from paired import read, write, sha, ref, palm, SIDES
