import numpy as np
import scipy as sp
from QUnfold import QUnfoldQUBO


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


def compute_chi2(observed, expected, covariance):
    """
    Compute the reduced chi-square between the observed and the expected histogram.

    Args:
        observed (numpy.ndarray): observed histogram.
        expected (numpy.ndarray): expected histogram.
        covariance (numpy.ndarray): covariance matrix.

    Returns:
        float: reduced chi-square.
    """
    residuals = observed - expected
    chi2 = residuals.T @ np.linalg.pinv(covariance) @ residuals
    chi2_red = chi2 / len(expected)
    return chi2_red


def lambda_optimizer(
    response, measured, truth, min_lam=0.0, max_lam=1.0, verbose=False
):
    def lambda_func(lam):
        unfolder = QUnfoldQUBO(response, measured, lam=lam)
        unfolder.initialize_qubo_model()
        sol, _, cov = unfolder.solve_gurobi_integer()
        chi2 = compute_chi2(observed=sol, expected=truth, covariance=cov)
        return chi2

    try:
        options = {"disp": 3 if verbose else 0}
        minimizer = sp.optimize.minimize_scalar(
            lambda_func, bracket=(min_lam, max_lam), method="brent", options=options
        )
    except AttributeError:
        raise ModuleNotFoundError(
            "Function 'lambda_optimizer' requires Gurobi solver: 'pip install gurobipy'"
        )
    return minimizer.x
