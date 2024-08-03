import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from qunfold import QUnfolder, QPlotter
from qunfold.utils import normalize_response, lambda_optimizer


# Set random seed for replicability
seed = 42
np.random.seed(seed)

# Set parameters for synthetic data generation
entries = 20000
bins = 14
xmin, xmax = 0.0, 15.0  # xaxis range
kappa, theta = 4.0, 1.0  # gamma distribution
mu_smear, std_smear = -0.1, 0.4  # gaussian smearing
efficiency = 0.7  # constant efficiency

# Generate Monte Carlo truth and reco data (with Gaussian smearing)
truth_mc_data = np.random.gamma(kappa, theta, size=entries)
smearing_mc = np.random.normal(mu_smear, std_smear, size=entries)
eff_mask_mc = np.random.rand(entries) < efficiency
reco_mc_data = (truth_mc_data + smearing_mc)[eff_mask_mc]

# Generate target truth and measured data (with Gaussian smearing)
truth_data = np.random.gamma(kappa, theta, size=entries)
smearing = np.random.normal(mu_smear, std_smear, size=entries)
eff_mask = np.random.rand(entries) < efficiency
measured_data = (truth_data + smearing)[eff_mask]

# Define histograms binning by adaptive KMeans algorithm
kbd = KBinsDiscretizer(bins, encode="ordinal", strategy="kmeans")
kbd.fit(np.clip(measured_data, xmin, xmax).reshape(-1, 1))
bin_edges = kbd.bin_edges_[0].tolist()
binning = np.array([-np.inf] + bin_edges + [np.inf])  # under/over-flow bins

# Generate Monte Carlo response matrix and measured data histogram
response, _, _ = np.histogram2d(reco_mc_data, truth_mc_data[eff_mask_mc], bins=binning)
truth_mc, _ = np.histogram(truth_mc_data, bins=binning)
response = normalize_response(response, truth_mc=truth_mc)
measured, _ = np.histogram(measured_data, bins=binning)

# Find optimal value for regularization parameter
lam = lambda_optimizer(
    response=response,
    measured=measured,
    truth=truth_mc,
    binning=binning,
    num_reps=10,
    seed=seed,
)

# Run simulated annealing to solve unfolding problem as QUBO
unfolder = QUnfolder(response=response, measured=measured, binning=binning, lam=lam)
unfolder.initialize_qubo_model()
sol, cov = unfolder.solve_simulated_annealing(num_reads=400, seed=seed)

# Plot response and unfolding result and save figures
plotter = QPlotter(
    response=response,
    measured=measured,
    truth=truth_mc,
    unfolded=sol,
    covariance=cov,
    binning=binning,
    method="SA",
    normed=True,
)
plotter.save_response("examples/simneal_response.png")
plotter.save_histograms("examples/simneal_result.png")
