"""Use the baseline renderer and keep only the hand crops for public review."""
from pathlib import Path
import sys
RUN = Path(__file__).resolve().parent
PREVIOUS = RUN.parent/'2026-11-26_realvideo34_purple_hand_a'
sys.path.insert(0,str(PREVIOUS))
import gallery_a
gallery_a.RUN = RUN
gallery_a.OUT = RUN.parents[1]/'outputs/purple_arm_hand_a_20261126'
gallery_a.main()
