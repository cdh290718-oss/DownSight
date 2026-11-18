"""Wait for detector output, then prepare and run the fixed 20k-step fit."""
import time,subprocess,sys
from common import ROOT,RUN,OUT
start=time.time()
while not (OUT/'detections.json').exists():
 if time.time()-start>600:raise RuntimeError('Detector output was not produced within 600 seconds; inspect detect.log')
 time.sleep(2)
subprocess.run([sys.executable,'-u',str(RUN/'prepare.py')],cwd=ROOT,check=True)
subprocess.run([sys.executable,'-u',str(RUN/'fit.py'),'--threads','1'],cwd=ROOT,check=True)
