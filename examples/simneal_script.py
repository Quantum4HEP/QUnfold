import numpy as np
from QUnfold import QUnfoldQUBO, QUnfoldPlotter
from QUnfold.utils import normalize_response, lambda_optimizer


# Set parameters for synthetic data generation
entries = 20000
bins = 16
xrange = np.linspace(start=-8.0, stop=8.0, num=bins + 1).tolist()
binning = np.array([-np.inf] + xrange + [np.inf])  # under/over-flow bins
mean, std = (0.23, 2.64)  # normal distribution
mean_smear, std_smear = (-0.35, 0.69)  # gaussian smearing

# Set random seed for replicability
seed = 42
np.random.seed(seed)

# Generate normalized response matrix from Monte Carlo samples
truth_mc_data = np.random.normal(loc=mean, scale=std, size=entries)
smearing_mc = np.random.normal(loc=mean_smear, scale=std_smear, size=entries)
reco_mc_data = truth_mc_data + smearing_mc
response = np.histogram2d(reco_mc_data, truth_mc_data, bins=binning)[0]
truth_mc = np.histogram(truth_mc_data, bins=binning)[0]
response = normalize_response(response, truth_mc=truth_mc)

# Generate measured histogram applying gaussian smearing
truth_data = np.random.normal(loc=mean, scale=std, size=entries)
smearing = np.random.normal(loc=mean_smear, scale=std_smear, size=entries)
measured_data = truth_data + smearing
measured = np.histogram(measured_data, bins=binning)[0]

# Find optimal value for regularization parameter
lam = lambda_optimizer(
    response=response,
    measured=measured,
    truth=truth_mc,
    binning=binning,
    num_reps=10,
    seed=seed,
)

# Run simulated annealing to solve QUBO problem
unfolder = QUnfoldQUBO(response=response, measured=measured, binning=binning, lam=lam)
unfolder.initialize_qubo_model()
sol, err, cov = unfolder.solve_simulated_annealing(num_reads=300, seed=seed)

# Plot response and unfolding result and save figures
plotter = QUnfoldPlotter(
    response=response,
    measured=measured,
    truth=truth_mc,
    unfolded=sol,
    error=err,
    covariance=cov,
    binning=binning,
    chi2=True,
)
plotter.save_response("examples/simneal_response.png")
plotter.save_histograms("examples/simneal_result.png", method="SA")
