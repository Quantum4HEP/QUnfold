import numpy as np
import numpy.testing as nptest
from QUnfold.utility import normalize_response


def test_normalize_response():
    response = np.array([[2, 0], [5, 1]])
    truth = np.array([4, 2])
    response = normalize_response(response, truth)
    np.testing.assert_allclose(
        response, np.array([[0.5, 0.0], [1.25, 0.5]]), rtol=1e-6, atol=1e-6
    )
