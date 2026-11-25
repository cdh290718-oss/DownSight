"""Repeat the historical 20k-step run with exactly the archived batch shape and ordering."""
import sys
from shared_run import OUT,prev,guard
def main():
    fit=prev.base.fit
    folder=prev.base.OUT
    audit=prev.read(prev.base.RUN/'evidence/inputs.json')
    assert prev.sha(folder/'prepared.json')==audit['solver_input_sha256']
    assert prev.sha(folder/'protocol.json')==audit['protocol_sha256']
    sys.addaudithook(guard)
    fit.RUN=folder
    fit.OUT=OUT/'cold'
    fit.OUT.mkdir(exist_ok=True)
    (fit.OUT/"prepared.json").write_bytes((folder/"prepared.json").read_bytes())
    fit.Problem=prev.base.trial.TorsoOnly
    fit.camera_files=prev.ref.camera_files
    sys.argv=[__file__,'--threads','1']
    fit.main()
if __name__=='__main__':main()
