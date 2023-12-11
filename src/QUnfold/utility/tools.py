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
