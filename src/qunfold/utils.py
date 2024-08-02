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
    """
    Normalize the response matrix using the Monte Carlo generated truth histogram.

    Args:
        response (numpy.ndarray): response matrix to normalize.
        truth_mc (numpy.ndarray): Monte Carlo truth histogram.

    Returns:
        numpy.ndarray: normalized response matrix.
    """
    return response / (truth_mc + 1e-12)


def compute_chi2(observed, expected, covariance, toys=False):
    nonzero = expected != 0
    observed = observed[nonzero]
    expected = expected[nonzero]
    if toys:
        cov = covariance[nonzero, :][:, nonzero]
    else:
        cov = np.diag(expected)
    residuals = observed - expected
    chi2 = residuals.T @ np.linalg.pinv(cov) @ residuals
    chi2_red = chi2 / len(expected)
    return chi2_red


def lambda_optimizer(
    response, measured, truth, binning, num_reps=30, verbose=False, seed=None
):
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
        minimizer = sp.optimize.minimize_scalar(
            fun=objective_fun,
            method="bounded",
            bounds=bounds,
            options=options,
        )
        lam = minimizer.x
        fun = minimizer.fun
        if fun < min_fun:
            best_lam = lam
            min_fun = fun
    return best_lam
