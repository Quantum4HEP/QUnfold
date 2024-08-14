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


def compute_chi2(observed, expected, covariance):
    diag = np.diag(covariance)
    iszero = diag == 0
    covariance[np.diag_indices_from(covariance)] = np.where(iszero, 1.0, diag)
    residuals = observed - expected
    try:
        inv_cov = np.linalg.inv(covariance)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(covariance)
    chi2 = residuals.T @ inv_cov @ residuals
    chi2_red = chi2 / len(expected)
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
        cov = np.diag(exp)
        chi2 = compute_chi2(observed=obs, expected=exp, covariance=cov)
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
