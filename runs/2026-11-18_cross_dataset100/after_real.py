"""Queue UnrealEgo2 after the real-image job to share a single GPU sequentially."""
import json,time,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[2];status=root/'outputs/cross_dataset100_20261118/inference_real_retry_status.json'
while True:
 try:d=json.loads(status.read_text())
 except (FileNotFoundError,json.JSONDecodeError):time.sleep(2);continue
 if 'exit_status' in d:
  if d['exit_status']!=0:raise RuntimeError('Real-image inference failed; queued inference not started')
  break
 time.sleep(2)
sys.exit(subprocess.run([sys.executable,'-u',str(Path(__file__).with_name('evaluate.py')),'--datasets','unrealego2'],cwd=root).returncode)
