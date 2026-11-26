"""Bind the archived paired optimizer to the new silhouettes and the original initialization."""
import sys
from context_run import RUN, OUT, PREVIOUS, paired, read, sha

def main(index):
    key = f'clip{index+1:03}'
    old = read(PREVIOUS/'evidence'/(key+'.json'))
    old_output = paired.OUT
    for mode in ('control','wilor'):
        path = old_output/'poses'/(key+'_'+mode+'.npz')
        assert sha(path) == old['results'][mode]['pose_sha256']
    paired.RUN = RUN
    paired.OUT = OUT
    paired.MODES = ['wilor']
    paired.run(index)
    new = read(RUN/'evidence'/(key+'.json'))
    assert new['start_sha256'] == old['start_sha256']
    for side in paired.SIDES:
        assert new['results']['start']['hands'][side] == old['results']['start']['hands'][side]

if __name__=='__main__':
    main(int(sys.argv[1]))
