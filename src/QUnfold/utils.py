import sys
import numpy as np
import scipy as sp
from tqdm import tqdm
from QUnfold import QUnfoldQUBO

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


def compute_chi2(observed, expected, covariance=None):
    nonzero = expected != 0
    observed = observed[nonzero]
    expected = expected[nonzero]
    residuals = observed - expected
    if covariance is None:
        chi2 = np.sum(np.square(residuals) / expected)
    else:
        covariance = covariance[nonzero, :][:, nonzero]
        chi2 = residuals.T @ np.linalg.pinv(covariance) @ residuals
    chi2_red = chi2 / len(expected)
    return chi2_red


def compute_chi2_toys(observed, expected, covariance):
    nonzero = expected != 0
    observed = observed[nonzero]
    expected = expected[nonzero]
    covariance = covariance[nonzero, :][:, nonzero]
    residuals = observed - expected
    chi2 = residuals.T @ np.linalg.pinv(covariance) @ residuals
    chi2_red = chi2 / len(expected)
    return chi2_red


def lambda_optimizer(response, measured, truth, num_reps=30, verbose=False, seed=None):
    if "gurobipy" not in sys.modules:
        raise ModuleNotFoundError("Function 'lambda_optimizer' requires Gurobi solver")
    np.random.seed(seed)

    def objective_fun(lam):
        unfolder = QUnfoldQUBO(response, measured, lam=lam)
        unfolder.initialize_qubo_model()
        sol, _, _ = unfolder.solve_gurobi_integer()
        pk = truth[1:-1] / np.sum(truth[1:-1])
        qk = sol[1:-1] / np.sum(sol[1:-1])
        mae = np.mean(np.abs(pk - qk))
        return mae

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
