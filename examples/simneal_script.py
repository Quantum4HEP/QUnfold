import numpy as np
from QUnfold import QUnfoldQUBO, QUnfoldPlotter
from QUnfold.utils import normalize_response, lambda_optimizer


if __name__ == "__main__":
    # Set parameters for data generation
    num_entries = 10000
    num_bins = 15
    min_bin = -7.0
    max_bin = 7.0
    bins = np.linspace(min_bin, max_bin, num_bins + 1)
    binning = np.array([-np.inf] + bins.tolist() + [np.inf])
    mean = 0.0
    std = 2.6
    mean_smear = -0.3
    std_smear = 0.5

    seed = 42
    np.random.seed(seed)

    # Generate and normalize response matrix
    mc_data = np.random.normal(loc=mean, scale=std, size=num_entries)
    reco_data = mc_data + np.random.normal(
        loc=mean_smear, scale=std_smear, size=num_entries
    )
    response = np.histogram2d(reco_data, mc_data, bins=binning)[0]
    mc_truth = np.histogram(mc_data, bins=binning)[0]
    response = normalize_response(response, mc_truth)

    # Generate random true data
    true_data = np.random.normal(loc=mean, scale=std, size=num_entries)

    # Apply gaussian smearing to get measured data
    meas_data = true_data + np.random.normal(
        loc=mean_smear, scale=std_smear, size=num_entries
    )

    # Generate truth and measured histograms
    truth = np.histogram(true_data, bins=binning)[0]
    measured = np.histogram(meas_data, bins=binning)[0]

    # Find optimal value for regularization parameter
    lam = lambda_optimizer(response, measured, truth, num_reps=20, seed=seed)

    # Run simulated annealing to solve QUBO problem
    unfolder = QUnfoldQUBO(response, measured, lam=lam)
    unfolder.initialize_qubo_model()
    sol, err, cov = unfolder.solve_simulated_annealing(
        num_reads=10, num_toys=100, seed=seed
    )

    ################## Quantum Annealing solver ##################
    """
    unfolder.set_quantum_device(
        device_name="Advantage_system6.4",
        dwave_token="<your_dwave_token>",
    )
    unfolder.set_graph_embedding()
    print("DWave_device =", unfolder._sampler.solver)
    print("num_logical_qubits =", unfolder.num_logical_qubits)
    print("num_physical_qubits =", unfolder.num_physical_qubits)
    sol, err, cov = unfolder.solve_quantum_annealing(num_reads=4000)
    """

    plotter = QUnfoldPlotter(
        response=response,
        measured=measured,
        truth=truth,
        unfolded=sol,
        error=err,
        covariance=cov,
        binning=binning,
        chi2=True,
    )
    plotter.saveResponse("examples/simneal_response.png")
    plotter.savePlot("examples/simneal_result.png", method="SA")
