# TODO reorganize
from importlib import reload

from optimize_config import *

Optimization.update(OptBest, qqVHH_LO, qqWHH_LO)
import optimize_coupling_basis

Optimization.reset()
Optimization.update(OptBest, ggZHH_NNLO)
reload(optimize_coupling_basis)