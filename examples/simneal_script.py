import numpy as np
from QUnfold import QUnfoldQUBO, QUnfoldPlotter
from QUnfold.utility import normalize_response


if __name__ == "__main__":
    # Set parameters for data generation
    num_entries = 10000
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
    binning = np.array([-np.inf] + bins.tolist() + [np.inf])
    response, _, _ = np.histogram2d(reco_data, mc_data, bins=binning)
    mc_truth, _ = np.histogram(mc_data, bins=binning)
    response = normalize_response(response, mc_truth)

    # Generate random true data
    true_data = np.random.normal(loc=mean, scale=std, size=num_entries)

    # Apply gaussian smearing to get measured data
    meas_data = true_data + np.random.normal(
        loc=mean_smear, scale=std_smear, size=num_entries
    )

    # Generate truth and measured histograms
    truth, _ = np.histogram(true_data, bins=binning)
    measured, _ = np.histogram(meas_data, bins=binning)

    # Run simulated annealing to solve QUBO problem
    unfolder = QUnfoldQUBO(response, measured, lam=0.1)
    unfolder.initialize_qubo_model()
    unfolded, error = unfolder.solve_simulated_annealing(
        num_reads=10, num_toys=100, seed=seed
    )

    # Plot unfolding result
    plotter = QUnfoldPlotter(
        response=response[1:-1, 1:-1],
        measured=measured[1:-1],
        truth=truth[1:-1],
        unfolded=unfolded[1:-1],
        error=error[1:-1],
        binning=bins,
        chi2=unfolder.compute_chi2(truth, "std"),
    )
    plotter.saveResponse("examples/simneal_response.png")
    plotter.savePlot("examples/simneal_result.png", method="SA")
