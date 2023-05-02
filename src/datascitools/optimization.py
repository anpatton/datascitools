from typing import List
from dataclasses import dataclass
from scipy.optimize import differential_evolution

@dataclass
class EvoVar:
    varname : str
    init : float
    bounds : tuple

def diff_evo(loss_fn : callable, params : List):
    varnames = [p.varname for p in params]
    x0 = tuple([p.init for p in params])
    bounds = [p.bounds for p in params]
    opt_val = differential_evolution(loss_fn, x0=x0, bounds=bounds)
    return {k:v for k,v in zip(varnames, opt_val.x)}

# b = EvoVar(varname='b', init=-0.4, bounds=(-4,4))
# c = EvoVar(varname='c', init=23_000, bounds=(5_000,50_000))
# w = EvoVar(varname='w', init=0.2, bounds=(0.001,0.999))
# s = EvoVar(varname='s', init=0.5, bounds=(0.1,2))
# best_params = diff_evo(get_weighted_loss_wrapper, [b,c,w,s])