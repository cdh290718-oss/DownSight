from base import *

def refine(index):
    sys.path.insert(0,str(D4))
    import refit
    refit.OLD=refit.OUT=OUT
    refit.PRIOR=refit.RUN=RUN
    refit.cameras=cameras
    refit.run(index)
