"""Replay one originally launched fixed-finger shard item using the final dual-mode entry point."""
import importlib.util
import numpy as np
from geometry_inputs import RUN,OUT,EV,prev

def main():
    spec=importlib.util.spec_from_file_location('finger_experiment_fit',RUN/'fit.py')
    fitter=importlib.util.module_from_spec(spec);spec.loader.exec_module(fitter)
    fitter.OUT=OUT/'repeat'
    fitter.OUT.mkdir(exist_ok=True)
    fitter.main(0,100,'fixed')
    base=OUT/'poses/pair001.npz';fresh=fitter.OUT/'poses/pair001.npz'
    with np.load(base) as a,np.load(fresh) as b:
        assert set(a.files)==set(b.files)
        diffs={k:float(np.max(np.abs(a[k]-b[k]))) for k in a.files}
        assert not any(diffs.values())
    prev.write(EV/'replay.json',dict(pair='pair001',max_abs_difference=diffs,archived_hash=prev.sha(base),
               replay_hash=prev.sha(fresh),exit_status=0))
if __name__=='__main__':main()
