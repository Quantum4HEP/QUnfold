import numpy as np


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
    

from QUnfold import QUnfoldQUBO
from QUnfold.utils import compute_chi2
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize_scalar

def lambda_optimizer(response, measured, truth, maxlam = 1, minlam = 0):
    def lamba_func(lam = .1):
        unfolder = QUnfoldQUBO(response, measured, lam=lam)
        unfolder.initialize_qubo_model()
        sol, err, cov = unfolder.solve_gurobi_integer()
        chi2 = compute_chi2(observed=sol, expected=truth, covariance=cov)
        return chi2

    minimizer = minimize_scalar(lamba_func,bracket=(minlam,maxlam),method='brent' ,options={'disp':3})
    return minimizer.x
