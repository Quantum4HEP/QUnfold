from QUnfold import QUnfoldQUBO


def run_QUnfold(method, response, measured, lam, num_reads=None, num_toys=None):
    unfolder = QUnfoldQUBO(response=response, measured=measured, lam=lam)
    unfolder.initialize_qubo_model()

    if method == "GRB":
        sol, err, cov = unfolder.solve_gurobi_integer()
    elif method == "SA":
        sol, err, cov = unfolder.solve_simulated_annealing(
            num_reads=num_reads,
            num_toys=num_toys,
        )
    elif method == "HYB":
        sol, err, cov = unfolder.solve_hybrid_sampler(
            num_toys=num_toys,
        )
    elif method == "QA":
        unfolder.set_quantum_device()
        unfolder.set_graph_embedding()
        sol, err, cov, _ = unfolder.solve_quantum_annealing(
            num_reads=num_reads,
            num_toys=num_toys,
        )
    return sol, err, cov
