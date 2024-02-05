from QUnfold import QUnfoldQUBO


def QUnfold_unfolder(unf_type, response, measured, distr, n_toys):
    unfolded = None
    error = None
    cov_matrix = None
    n_reads = 100  # 1000
    lam = 0.05

    if distr == "normal":
        lam = 0.03
    elif distr == "gamma":
        lam = 0.055
    elif distr == "exponential":
        lam = 0.001
    elif distr == "breit-wigner":
        lam = 0.0001
    unfolder = QUnfoldQUBO(response, measured, lam=lam)
    unfolder.initialize_qubo_model()

    if unf_type == "SA":
        unfolded, error, cov_matrix, _ = unfolder.solve_simulated_annealing(
            num_reads=n_reads, num_toys=n_toys
        )
    elif unf_type == "HYB":
        unfolded, error, cov_matrix, _ = unfolder.solve_hybrid_sampler(num_toys=n_toys)
    elif unf_type == "QA":
        unfolded, error, cov_matrix, _ = unfolder.solve_quantum_annealing(
            num_reads=n_reads, num_toys=n_toys
        )

    return unfolded, error, cov_matrix
