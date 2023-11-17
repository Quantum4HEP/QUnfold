import numpy as np
from QUnfold import QUnfoldQUBO, QUnfoldPlotter


# Set parameters for data generation
num_entries = 4000
num_bins = 16
min_bin = -8.0
max_bin = 8.0


# Generate random true data
true_data = np.random.normal(loc=0.0, scale=1.8, size=num_entries)


# Apply gaussian smearing to get measured data
smearing = np.random.normal(loc=-0.3, scale=0.5, size=num_entries)
meas_data = true_data + smearing


# Generate truth and measured histograms
bins = np.linspace(min_bin, max_bin, num_bins + 1)
true, _ = np.histogram(true_data, bins=bins)
measured, _ = np.histogram(meas_data, bins=bins)


# Generate and normalize response matrix
response, _, _ = np.histogram2d(meas_data, true_data, bins=bins)
response /= true + 1e-6


# Run simulated annealing to solve QUBO problem
qubo = QUnfoldQUBO(response=response, measured=measured, lam=0.01)
unfolded = qubo.solve_simulated_annealing(num_reads=100)


# Plot unfolding result
plotter = QUnfoldPlotter(
    response=response, measured=measured, truth=true, unfolded=unfolded, binning=bins
)
plotter.plotResponse()
plotter.plot()
