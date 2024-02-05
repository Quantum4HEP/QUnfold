import numpy as np
from scipy.stats import chisquare


def normalize_response(response, truth_mc):
    """
    Function used to normalize the response matrix using the Monte Carlo generated truth distribution.

    Args:
        response (numpy.ndarray): the response matrix to be normalized.
        truth_mc (numpy.ndarray): the Monte Carlo truth histogram used to normalize the response matrix.

    Returns:
        numpy.ndarray: the normalized response matrix.
    """
    return response / (truth_mc + 1e-6)


def compute_chi2(unfolded, truth, covariance=None):
    """
    Compute the chi-square statistic for the unfolded distribution.

    Args:
        unfolded (numpy.ndarray): the unfolded distribution to be compare with truth.
        truth (numpy.ndarray): the true distribution against which to compare.
        covariance (numpy.ndarray, optional): the covariance matrix to be used to compute chi2 (default None). If None, chi2 is computed using scipy.

    Returns:
        float: The computed chi-square statistic.
    """
    chi2 = None
    null_indices = truth == 0
    truth[null_indices] += 1
    unfolded[null_indices] += 1
    if covariance is None:
        chi2, _ = chisquare(
            unfolded,
            np.sum(unfolded) / np.sum(truth) * truth,
        )
    else:
        residuals = unfolded - truth
        inv_covariance_matrix = np.linalg.inv(covariance)
        chi2 = residuals.T @ inv_covariance_matrix @ residuals
    dof = len(unfolded)
    return chi2 / dof
