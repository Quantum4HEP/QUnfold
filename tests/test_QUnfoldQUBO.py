import numpy as np
from QUnfold import QUnfoldQUBO


def test_qunfoldqubo_constructor_and_setters():
    """
    Test the constructor and setters of the QUnfoldQUBO class.
    """

    response_matrix = np.array([[1, 2], [3, 4]])
    measured_distribution = np.array([0.5, 0.5])
    regularization_parameter = 0.1

    # Test constructors
    qunfold_qubo = QUnfoldQUBO(
        response_matrix, measured_distribution, lam=regularization_parameter
    )

    assert np.array_equal(qunfold_qubo.R, response_matrix)
    assert np.array_equal(qunfold_qubo.d, measured_distribution)
    assert qunfold_qubo.lam == regularization_parameter

    # Test setters
    qunfold_qubo.set_lam_parameter(0.2)
    qunfold_qubo.set_measured(np.array([0.8, 0.5]))
    qunfold_qubo.set_response(np.array([[1, 3], [3, 4]]))

    assert qunfold_qubo.lam == 0.2
    assert np.array_equal(qunfold_qubo.d, np.array([0.8, 0.5]))
    assert np.array_equal(qunfold_qubo.R, np.array([[1, 3], [3, 4]]))


def test_get_laplacian():
    """
    Test the _get_laplacian method of the QUnfoldQUBO class.
    """

    qunfold_qubo = QUnfoldQUBO(
        None, None
    )  # You can instantiate with dummy values since the method is static

    # Test for a 3x3 Laplacian matrix
    laplacian_matrix_3x3 = qunfold_qubo._get_laplacian(3)
    expected_matrix_3x3 = np.array([[-1, 1, 0], [1, -2, 1], [0, 1, -1]])
    assert np.array_equal(laplacian_matrix_3x3, expected_matrix_3x3)

    # Test for a 4x4 Laplacian matrix
    laplacian_matrix_4x4 = qunfold_qubo._get_laplacian(4)
    expected_matrix_4x4 = np.array(
        [[-1, 1, 0, 0], [1, -2, 1, 0], [0, 1, -2, 1], [0, 0, 1, -1]]
    )
    assert np.array_equal(laplacian_matrix_4x4, expected_matrix_4x4)


def test_statistics():
    """
    Test the statistics of the QUnfoldQUBO class (covariance, chi2, etc...).
    """

    R = np.array([[1, 2], [3, 4]])
    meas = np.array([0.5, 0.5])
    unfolder = QUnfoldQUBO(R, meas, lam=0.1)
    unfolder.initialize_qubo_model()
    unfolder.solve_simulated_annealing(num_reads=10, n_toys=1000)

    # Test covariance and correlation
    cov = unfolder.cov_matrix
    corr = unfolder.corr_matrix
    assert cov.shape == (2, 2)
    assert corr.shape == (2, 2)

    # Test chi2
    truth = np.array([0.5, 0.5])
    chi2_std = unfolder.compute_chi2(truth, method="std")
    chi2_cov = unfolder.compute_chi2(truth, method="cov")
    assert chi2_std != None
    assert chi2_cov != None
