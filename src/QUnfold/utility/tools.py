import numpy as np
import scipy as sp


def normalize_response(response, truth_mc):
    """
    Normalize the response matrix using the Monte Carlo generated truth histogram.

    Args:
        response (numpy.ndarray): response matrix to be normalized.
        truth_mc (numpy.ndarray): Monte Carlo truth histogram for the normalization.

    Returns:
        numpy.ndarray: normalized response matrix.
    """
    return response / (truth_mc + 1e-6)


def compute_chi2(unfolded, truth, covariance=None):
    """
    Compute the chi-square statistic for the unfolded histogram.

    Args:
        unfolded (numpy.ndarray): unfolded histogram to be compared with the truth.
        truth (numpy.ndarray): target truth histogram.
        covariance (numpy.ndarray, optional): covariance matrix to compute the chi2 (default is None).
        If None, chi2 is computed by using scipy.

    Returns:
        float: chi-square statistic.
    """
    null_indices = truth == 0
    truth[null_indices] += 1
    unfolded[null_indices] += 1
    if covariance is None:
        chi2, _ = sp.stats.chisquare(unfolded, truth)
    else:
        residuals = unfolded - truth
        inv_covariance = np.linalg.inv(covariance)
        chi2 = residuals.T @ inv_covariance @ residuals
    dof = len(unfolded)
    return chi2 / dof
