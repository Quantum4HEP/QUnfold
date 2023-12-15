import numpy as np
from QUnfold import QUnfoldQUBO, QUnfoldPlotter
from QUnfold.utility import normalize_response


if __name__ == "__main__":
    # Set parameters for data generation
    num_entries = 20000
    num_bins = 15
    min_bin = -7.0
    max_bin = 7.0
    bins = np.linspace(min_bin, max_bin, num_bins + 1)

    seed = 42
    np.random.seed(seed)
    mean = 0.0
    std = 2.6
    mean_smear = -0.3
    std_smear = 0.5

    # Generate and normalize response matrix
    mc_data = np.random.normal(loc=mean, scale=std, size=num_entries)
    reco_data = mc_data + np.random.normal(
        loc=mean_smear, scale=std_smear, size=num_entries
    )
    response, _, _ = np.histogram2d(reco_data, mc_data, bins=bins)
    mc_truth, _ = np.histogram(mc_data, bins=bins)
    response = normalize_response(response, mc_truth)

    # Generate random true data
    true_data = np.random.normal(loc=mean, scale=std, size=num_entries)

    # Apply gaussian smearing to get measured data
    meas_data = true_data + np.random.normal(
        loc=mean_smear, scale=std_smear, size=num_entries
    )

    # Generate truth and measured histograms
    truth, _ = np.histogram(true_data, bins=bins)
    measured, _ = np.histogram(meas_data, bins=bins)

    # Run simulated annealing to solve QUBO problem
    unfolder = QUnfoldQUBO(response, measured, lam=0.1)
    unfolder.initialize_qubo_model()
    unfolded, error = unfolder.solve_simulated_annealing(num_reads=100, seed=seed)

    # Plot unfolding result
    plotter = QUnfoldPlotter(
        response=response,
        measured=measured,
        truth=truth,
        unfolded=unfolded,
        error=error,
        binning=bins,
    )
    plotter.saveResponse("examples/simneal_response.png")
    plotter.savePlot("examples/simneal_result.png", method="SA")
