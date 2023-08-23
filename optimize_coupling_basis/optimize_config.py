from __future__ import annotations

from functools import cached_property
from typing import Any, Callable

import numpy as np
import scipy.optimize as opt
from heptools.config import Config, Undefined, config_property
from heptools.physics import Coupling, FormulaXS


def lme(array):
    '''https://en.wikipedia.org/wiki/LogSumExp'''
    return np.log(np.mean(np.exp(array)))

class Optimization(Config):
    model: type[FormulaXS] = Undefined
    basis: np.ndarray = Undefined
    kappa: list[str] = Undefined
    bounds: dict[str, tuple[float, float]] = Undefined
    dimensions: list[int] = Undefined
    rounding: int = Undefined

    default_basis: np.ndarray = Undefined
    reserved: list[dict[str, float]] = [{}]

    algorithm_global: dict[str, Callable[[Any], Any]] = Undefined
    algorithm_local: list[str] = Undefined
    objective_functions: dict[str, Callable[[Any], float]] = Undefined

    @config_property
    def sample(cls) -> FormulaXS:
        return cls.model(cls.basis)

    @config_property
    def reserved_basis(cls) -> np.ndarray:
        coupling = Coupling(cls.kappa)
        for i in cls.reserved:
            coupling.meshgrid(**i)
        return coupling.couplings

class qqVHH_LO(Optimization):
    kappa = ['CV', 'C2V', 'C3']
    bounds = {'CV': (-5,5), 'C2V': (-15,15), 'C3': (-30,30)}
    dimensions = [6, 7, 8, 9]
    rounding = 2
    class model(FormulaXS):
        @cached_property
        def diagrams(self):
            return ['CV', 'C2V', 'C3'], [[1, 0, 1], [2, 0, 0], [0, 1, 0]] 
        @cached_property
        def search_pattern(self):
            ...

    default_basis = np.array([[1,1,1], [1.5,1,1], [0.5,1,1], [1,2,1], [1,0,1], [1,1,0], [1,1,2], [1,1,20]])

class qqZHH_LO(qqVHH_LO):
    basis = np.asarray([
        [1.0, 1.0,  1.0, 2.642e-04],
        [1.5, 1.0,  1.0, 5.738e-04],
        [0.5, 1.0,  1.0, 1.663e-04],
        [1.0, 2.0,  1.0, 6.770e-04],
        [1.0, 0.0,  1.0, 9.037e-05],
        [1.0, 1.0,  0.0, 1.544e-04],
        [1.0, 1.0,  2.0, 4.255e-04],
        [1.0, 1.0, 20.0, 1.229e-02]
    ])

class qqWHH_LO(qqVHH_LO):
    basis = np.asarray([
        [1.0, 1.0,  1.0, 4.152e-04],
        [1.5, 1.0,  1.0, 8.902e-04],
        [0.5, 1.0,  1.0, 2.870e-04],
        [1.0, 2.0,  1.0, 1.115e-03],
        [1.0, 0.0,  1.0, 1.491e-04],
        [1.0, 1.0,  0.0, 2.371e-04],
        [1.0, 1.0,  2.0, 6.880e-04],
        [1.0, 1.0, 20.0, 2.158e-02]
    ])

class ggZHH_NNLO(Optimization):
    kappa = ['CV', 'C2V', 'C3', 'CF']
    bounds = {'CV': (-5,5), 'C2V': (-15,15), 'C3': (-30,30), 'CF': (-5,5)}
    dimensions = [18, 19]
    rounding = 4
    class model(FormulaXS):
        @cached_property
        def diagrams(self):
            return ['CV', 'C2V', 'C3', 'CF'], [
                [0, 0, 0, 2], [1, 0, 0, 1], [0, 0, 1, 1],
                [2, 0, 0, 0], [1, 0, 1, 0], [0, 1, 0, 0]]
        @cached_property
        def search_pattern(self):
            ...
    basis = np.concatenate((
        np.array([[1,1,1,1], [0,10,0,0], [0,10,1,0], [0,10,20,0], [0,10,2,0], [0,10,5,0], [0,1,0,0], [0,1,20,0], [0,1,2,0], [0,1,5,0], [0,2,0,0], [0,2,1,0], [0,2,20,0], [0,2,2,0], [0,2,5,0], [0,5,0,0], [0,5,1,0], [0,5,20,0], [0,5,2,0], [0,5,5,0], [0.5,0,0,0], [0.5,0,1,0], [0.5,0,20,0], [0.5,0,2,0], [0.5,0,5,0], [0.5,10,0,0], [0.5,10,1,0], [0.5,10,20,0], [0.5,10,2,0], [0.5,10,5,0], [0.5,1,0,0], [0.5,1,1,0], [0.5,1,20,0], [0.5,1,2,0], [0.5,1,5,0], [0.5,2,0,0], [0.5,2,1,0], [0.5,2,20,0], [0.5,2,2,0], [0.5,2,5,0], [0.5,5,0,0], [0.5,5,1,0], [0.5,5,20,0], [0.5,5,2,0], [0.5,5,5,0], [1,0,0,0], [1,0,20,0], [1,0,2,0], [1,0,5,0], [1,10,0,0], [1,10,20,0], [1,10,2,0], [1,10,5,0], [1,1,2,0], [1,1,5,0], [1,2,0,0], [1,2,1,0], [1,2,20,0], [1,2,2,0], [1,2,5,0], [1,5,0,0], [1,5,1,0], [1,5,20,0], [1,5,2,0], [1,5,5,0], [2,0,0,0], [2,0,1,0], [2,0,20,0], [2,0,2,0], [2,0,5,0], [2,10,0,0], [2,10,1,0], [2,10,20,0], [2,10,2,0], [2,10,5,0], [2,1,0,0], [2,1,20,0], [2,1,2,0], [2,1,5,0], [2,2,0,0], [2,2,1,0], [2,2,20,0], [2,2,2,0], [2,2,5,0], [2,5,0,0], [2,5,1,0], [2,5,20,0], [2,5,2,0], [2,5,5,0], [0,0,1,1], [0,0,20,1], [0,0,2,1], [0,0,5,1], [0,10,0,1], [0,10,1,1], [0,10,20,1], [0,10,2,1], [0,10,5,1], [0,1,0,1], [0,1,20,1], [0,1,2,1], [0,1,5,1], [0,2,0,1], [0,2,1,1], [0,2,20,1], [0,2,2,1], [0,2,5,1], [0,5,0,1], [0,5,1,1], [0,5,20,1], [0,5,2,1], [0,5,5,1], [0.5,0,0,1], [0.5,0,1,1], [0.5,0,20,1], [0.5,0,2,1], [0.5,0,5,1], [0.5,10,0,1], [0.5,10,1,1], [0.5,10,20,1], [0.5,10,2,1], [0.5,10,5,1], [0.5,1,0,1], [0.5,1,1,1], [0.5,1,20,1], [0.5,1,2,1], [0.5,1,5,1], [0.5,2,0,1], [0.5,2,1,1], [0.5,2,20,1], [0.5,2,2,1], [0.5,2,5,1], [0.5,5,0,1], [0.5,5,1,1], [0.5,5,20,1], [0.5,5,2,1], [0.5,5,5,1], [1,0,0,1], [1,0,20,1], [1,0,2,1], [1,0,5,1], [1,10,0,1], [1,10,20,1], [1,10,2,1], [1,10,5,1], [1,1,2,1], [1,1,5,1], [1,2,0,1], [1,2,1,1], [1,2,20,1], [1,2,2,1], [1,2,5,1], [1,5,0,1], [1,5,1,1], [1,5,20,1], [1,5,2,1], [1,5,5,1], [2,0,0,1], [2,0,1,1], [2,0,20,1], [2,0,2,1], [2,0,5,1], [2,10,0,1], [2,10,1,1], [2,10,20,1], [2,10,2,1], [2,10,5,1], [2,1,0,1], [2,1,20,1], [2,1,2,1], [2,1,5,1], [2,2,0,1], [2,2,1,1], [2,2,20,1], [2,2,2,1], [2,2,5,1], [2,5,0,1], [2,5,1,1], [2,5,20,1], [2,5,2,1], [2,5,5,1], [0,0,0,3], [0,0,1,3], [0,0,2,3], [0,0,5,3], [0,10,0,3], [0,10,1,3], [0,10,20,3], [0,10,2,3], [0,10,5,3], [0,1,0,3], [0,1,20,3], [0,1,2,3], [0,1,5,3], [0,2,0,3], [0,2,1,3], [0,2,20,3], [0,2,2,3], [0,2,5,3], [0,5,0,3], [0,5,1,3], [0,5,20,3], [0,5,2,3], [0,5,5,3], [0.5,0,0,3], [0.5,0,1,3], [0.5,0,20,3], [0.5,0,2,3], [0.5,0,5,3], [0.5,10,0,3], [0.5,10,1,3], [0.5,10,20,3], [0.5,10,2,3], [0.5,10,5,3], [0.5,1,0,3], [0.5,1,1,3], [0.5,1,20,3], [0.5,1,2,3], [0.5,1,5,3], [0.5,2,0,3], [0.5,2,1,3], [0.5,2,20,3], [0.5,2,2,3], [0.5,2,5,3], [0.5,5,0,3], [0.5,5,1,3], [0.5,5,20,3], [0.5,5,2,3], [0.5,5,5,3], [1,0,0,3], [1,0,20,3], [1,0,2,3], [1,0,5,3], [1,10,20,3], [1,10,2,3], [1,10,5,3], [1,1,2,3], [1,1,5,3], [1,2,1,3], [1,2,2,3], [1,2,5,3], [1,5,1,3], [1,5,2,3], [1,5,5,3], [2,0,1,3], [2,0,2,3], [2,0,5,3], [2,10,1,3], [2,10,2,3], [2,10,5,3], [2,1,2,3], [2,1,5,3], [2,2,1,3], [2,2,2,3], [2,2,5,3], [2,5,1,3], [2,5,2,3], [2,5,5,3]]),
        np.array([4.76e-05, 0.01239, 0.01239, 0.01239, 0.01239, 0.01239, 0.0001239, 0.0001239, 0.0001239, 0.0001239, 0.0004955, 0.0004955, 0.0004955, 0.0004955, 0.0004955, 0.003097, 0.003097, 0.003097, 0.003097, 0.003097, 9.133e-05, 6.405e-05, 0.001513, 4.644e-05, 5.435e-05, 0.009945, 0.01052, 0.01983, 0.01088, 0.01207, 4.885e-06, 1.776e-05, 0.002262, 4.118e-05, 0.0001746, 0.0001549, 0.0002115, 0.003248, 0.0002771, 0.0005378, 0.002018, 0.002255, 0.007656, 0.002429, 0.003065, 0.001461, 0.004441, 0.001025, 0.0006526, 0.005064, 0.02487, 0.00637, 0.00852, 0.0004812, 0.00036, 0.0002977, 0.0002201, 0.006584, 0.000178, 0.000301, 0.0003215, 0.000489, 0.01165, 0.0006948, 0.001572, 0.02338, 0.02168, 0.01391, 0.01983, 0.015, 0.002465, 0.002103, 0.02585, 0.001843, 0.00207, 0.02017, 0.01411, 0.01699, 0.01263, 0.01724, 0.01581, 0.01441, 0.01434, 0.01049, 0.01005, 0.008863, 0.01684, 0.007844, 0.005596, 0.0001257, 0.004065, 9.815e-05, 0.0001718, 0.0138, 0.01308, 0.004187, 0.01289, 0.01083, 0.0004871, 0.002831, 0.0002592, 0.0001249, 0.001001, 0.0008348, 0.002023, 0.0006538, 0.0003256, 0.004179, 0.003635, 0.001034, 0.003256, 0.002334, 9.06e-05, 7.082e-05, 0.0004429, 5.334e-05, 2.503e-05, 0.01383, 0.01377, 0.009031, 0.01345, 0.01259, 0.0003748, 0.0003495, 0.0002155, 0.0003065, 0.0001981, 0.0009337, 0.0008703, 0.0002292, 0.0008021, 0.0006125, 0.00395, 0.003877, 0.001721, 0.003731, 0.003302, 0.0001481, 0.0005806, 0.0001279, 0.0001194, 0.009842, 0.0134, 0.01026, 0.01071, 5.289e-05, 9.067e-05, 0.00018, 0.000199, 0.00122, 0.0002194, 0.0003028, 0.002005, 0.002097, 0.003989, 0.002177, 0.002382, 0.01141, 0.01073, 0.00657, 0.009918, 0.007914, 0.0007022, 0.0008155, 0.01535, 0.0009857, 0.001879, 0.009288, 0.00634, 0.007938, 0.006239, 0.007378, 0.006815, 0.006377, 0.006214, 0.004779, 0.003104, 0.002773, 0.007907, 0.002454, 0.001893, 0.01445, 0.01279, 0.01137, 0.008457, 0.04336, 0.03969, 0.01184, 0.0347, 0.02572, 0.01625, 0.02192, 0.01237, 0.008917, 0.01827, 0.01599, 0.01986, 0.01406, 0.009882, 0.02584, 0.02313, 0.01504, 0.02044, 0.01402, 0.01396, 0.01258, 0.01057, 0.01117, 0.007818, 0.04597, 0.0436, 0.01035, 0.04038, 0.03221, 0.0161, 0.01456, 0.009474, 0.01302, 0.009181, 0.01845, 0.01684, 0.008629, 0.01511, 0.01077, 0.0269, 0.02496, 0.00749, 0.02287, 0.017, 0.01172, 0.004323, 0.00974, 0.007036, 0.01275, 0.03992, 0.03374, 0.01172, 0.00863, 0.01518, 0.01394, 0.01046, 0.02361, 0.02194, 0.01735, 0.002627, 0.002447, 0.002002, 0.02275, 0.02219, 0.02076, 0.00334, 0.002792, 0.004732, 0.004485, 0.003824, 0.009743, 0.009372, 0.008389]).reshape((-1, 1))), 
        axis = -1)

    default_basis = np.array([[0.5,1,0,0], [2,10,5,0], [0,0,5,1], [0,10,20,1], [0,5,20,1], [0.5,0,20,1], [0.5,0,5,1], [0.5,1,20,1], [0.5,2,20,1], [2,10,5,1], [0,5,20,3], [2,10,0,1], [2,10,1,1], [1,0,20,1], [1,2,0,1], [2,5,5,1], [1,0,0,1], [1,1,1,1], [2,0,1,3], [2,0,2,3], [0,1,5,3], [0,2,2,3], [0,5,5,3], [1,0,20,3]])

class OptStudy(Optimization):
    algorithm_global = {
        'dual_annealing': opt.dual_annealing,
        'differential_evolution': opt.differential_evolution,
    }
    algorithm_local = ['Nelder-Mead', 'Powell', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP', 'BFGS', 'trust-constr']
    objective_functions = {
        'lme': lme,
        'max': np.nanmax,
        'mean': np.mean,
    }

class OptBest(Optimization):
    algorithm_global = {
        'dual_annealing': opt.dual_annealing,
    }
    algorithm_local = ['BFGS', 'trust-constr']
    objective_functions = {
        'lme': lme,
    }