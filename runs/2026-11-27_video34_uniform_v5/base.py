from pathlib import Path
import sys,json,hashlib
RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[1]
OUT=ROOT/'outputs/video34_uniform_v5_20261127'
D3=RUN.parent/'2026-11-25_realvideo4_dv3'
D4=RUN.parent/'2026-11-25_realvideo4_dv4'
sys.path.insert(0,str(D3))
from fileset import read,write,sha,SIDES,SESSION,cameras
