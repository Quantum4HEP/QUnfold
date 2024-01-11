# Main modules
import sys
import ROOT as r
import numpy as np
from analysis_functions.custom_logger import get_custom_logger
from analysis_functions.generator import generate
from analysis_functions.RooUnfold import (
    RooUnfold_unfolder,
    RooUnfold_plot,
    RooUnfold_plot_response,
)
from analysis_functions.QUnfolder import QUnfold_unfolder_and_plot
from analysis_functions.comparisons import plot_comparisons
from QUnfold.utility import TH1_to_array, TH2_to_array, normalize_response

# Settings
log = get_custom_logger(__name__)
loaded_RooUnfold = r.gSystem.Load("HEP_deps/RooUnfold/libRooUnfold.so")
if not loaded_RooUnfold == 0:
    log.error("RooUnfold not found!")
    sys.exit(0)

# Input variables
distributions = {
    "normal": np.linspace(0, 10, 21),
    "gamma": np.linspace(0, 10, 21),
    "exponential": np.linspace(0, 10, 21),
    "breit-wigner": np.linspace(0, 10, 21),
    "double-peaked": np.linspace(0, 10, 21),
}
samples = 10000
bias = -0.13
smearing = 0.21
eff = 0.7


if __name__ == "__main__":
    # Iterate over distributions
    for distr, binning in distributions.items():
        # Generate data
        log.info("Unfolding the {} distribution".format(distr))
        truth, measured, response = generate(
            distr, binning, samples, bias, smearing, eff
        )

        ########################## Classic ###########################

        # RooUnfold settings
        r_response = response
        RooUnfold_plot_response(r_response, distr)
        r_response.UseOverflow(False)

        # Matrix inversion (MI)
        unfolded_MI, error_MI = RooUnfold_unfolder("MI", r_response, measured)
        RooUnfold_plot(truth, measured, unfolded_MI, distr)

        # Iterative Bayesian unfolding (IBU)
        unfolded_IBU, error_IBU = RooUnfold_unfolder("IBU", r_response, measured)
        RooUnfold_plot(truth, measured, unfolded_IBU, distr)

        ########################## Quantum ###########################

        # QUnfold settings
        truth = TH1_to_array(truth, overflow=False)
        measured = TH1_to_array(measured, overflow=False)
        response = normalize_response(
            TH2_to_array(response.Hresponse()), TH1_to_array(response.Htruth())
        )

        # Simulated annealing (SA)
        unfolded_SA, error_SA = QUnfold_unfolder_and_plot(
            "SA", response, measured, truth, distr, binning
        )

        # # Hybrid solver (HYB)
        unfolded_HYB, error_HYB = QUnfold_unfolder_and_plot(
            "HYB", response, measured, truth, distr, binning
        )

        ########################## Compare ###########################

        # Comparison settings
        data = {
            "MI": TH1_to_array(unfolded_MI, overflow=False),
            "IBU4": TH1_to_array(unfolded_IBU, overflow=False),
            "SA": unfolded_SA,
            "HYB": unfolded_HYB,
        }
        errors = {
            "MI": error_MI,
            "IBU4": error_IBU,
            "SA": error_SA,
            "HYB": error_HYB,
        }

        # Plot comparisons
        plot_comparisons(data, errors, distr, truth, measured, binning)
        log.info("Done\n")
