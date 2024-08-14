import sys
import numpy as np
import scipy as sp
from tqdm import tqdm
from qunfold import QUnfolder

try:
    import gurobipy
except ImportError:
    pass


def normalize_response(response, truth_mc):
    iszero = truth_mc == 0.0
    response /= np.where(iszero, 1.0, truth_mc)
    diag = np.diag(response)
    response[np.diag_indices_from(response)] = np.where(iszero, 1.0, diag)
    return response


def compute_chi2(observed, expected):
    iszero = expected == 0
    observed = np.where(iszero, observed + 1, observed)
    expected = np.where(iszero, expected + 1, expected)
    chi2 = np.sum((observed - expected) ** 2 / expected)
    chi2_red = chi2 / (len(expected) - 1)
    return chi2_red


def lambda_optimizer(response, measured, truth, binning, num_reps=30, verbose=False, seed=None):
    if "gurobipy" not in sys.modules:
        raise ModuleNotFoundError("Function 'lambda_optimizer' requires Gurobi solver")
    np.random.seed(seed)

    def objective_fun(lam):
        unfolder = QUnfolder(response, measured, binning=binning, lam=lam)
        unfolder.initialize_qubo_model()
        sol, _ = unfolder.solve_gurobi_integer()
        obs = sol[1:-1]
        exp = truth[1:-1]
        chi2 = compute_chi2(observed=obs, expected=exp)
        return chi2

    best_lam = 0
    min_fun = objective_fun(best_lam)
    options = {"xatol": 0, "maxiter": 100, "disp": 3 if verbose else 0}
    for _ in tqdm(range(num_reps), desc="Optimizing lambda"):
        bounds = (0, np.random.rand())
        minimizer = sp.optimize.minimize_scalar(fun=objective_fun, method="bounded", bounds=bounds, options=options)
        lam = minimizer.x
        fun = minimizer.fun
        if fun < min_fun:
            best_lam = lam
            min_fun = fun
    return best_lam
