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
