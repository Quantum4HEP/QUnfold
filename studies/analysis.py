# Main modules
import sys
import ROOT as r
import numpy as np
from analysis_functions.custom_logger import get_custom_logger
from analysis_functions.generator import generate
from analysis_functions.RooUnfold import RooUnfold_unfolder
from analysis_functions.QUnfolder import QUnfold_unfolder
from analysis_functions.comparisons import plot_comparisons
from QUnfold.utility import (
    TH1_to_array,
    TH2_to_array,
    normalize_response,
    TMatrix_to_array,
)

# Settings
log = get_custom_logger(__name__)
loaded_RooUnfold = r.gSystem.Load("HEP_deps/RooUnfold/libRooUnfold.so")
if not loaded_RooUnfold == 0:
    log.error("RooUnfold not found!")
    sys.exit(0)

# Binning
bin_normal = np.array(
    [
        1.0,
        1.5,
        2.0,
        2.5,
        3.0,
        3.5,
        4.0,
        4.5,
        5.0,
        5.5,
        6.0,
        6.5,
        7.0,
        7.5,
        8.0,
        8.5,
        9.0,
        9.5,
    ]
)
bin_gamma = np.array(
    [
        1.0,
        1.5,
        2.0,
        2.5,
        3.0,
        3.5,
        4.0,
        4.5,
        5.0,
        5.5,
        6.0,
        6.5,
        7.0,
        7.5,
        8.0,
        8.5,
        9.0,
        9.5,
        10.0,
        10.5,
        11.0,
        11.5,
        12.0,
        12.5,
        13.0,
    ]
)
bin_exponential = np.array(
    [
        0.5,
        1.0,
        1.5,
        2.0,
        2.5,
        3.0,
        3.5,
        4.0,
        4.5,
        5.0,
        5.5,
        6.0,
        6.5,
        7.0,
        7.5,
        8.0,
        8.5,
        9.0,
        9.5,
        10.0,
    ]
)
bin_bw = np.array(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        2.5,
        3.0,
        3.5,
        4.0,
        4.5,
        5.0,
        5.5,
        6.0,
        6.5,
        7.0,
        7.5,
        8.0,
        8.5,
        9.0,
        9.5,
        10.0,
    ]
)

# Input variables
samples = 1000000
n_toys = 100
quantum = False
distributions = {
    # "normal": bin_normal,
    "gamma": bin_gamma,
    # "exponential": bin_exponential,
    # "breit-wigner": bin_bw,
}
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
        r_response.UseOverflow(True)

        # Iterative Bayesian unfolding (IBU)
        unfolded_IBU, error_IBU, cov_IBU = RooUnfold_unfolder(
            "IBU", r_response, measured, n_toys
        )

        # Matrix inversion (MI)
        unfolded_MI, error_MI, cov_MI = RooUnfold_unfolder(
            "MI", r_response, measured, n_toys
        )

        ########################## Quantum ###########################

        # QUnfold settings
        truth = TH1_to_array(truth, overflow=True)
        measured = TH1_to_array(measured, overflow=True)
        response = normalize_response(
            TH2_to_array(response.Hresponse(), overflow=True),
            TH1_to_array(response.Htruth(), overflow=True),
        )

        # Simulated annealing (SA)
        unfolded_SA, error_SA, cov_SA = QUnfold_unfolder(
            "SA", response, measured, distr, n_toys
        )

        if quantum:
            # # Hybrid solver (HYB)
            unfolded_HYB, error_HYB, cov_HYB = QUnfold_unfolder(
                "HYB", response, measured, distr, n_toys
            )

            # Quantum annealing (QA)
            unfolded_QA, error_QA, cov_QA = QUnfold_unfolder(
                "QA", response, measured, distr, n_toys
            )

        ########################## Compare ###########################

        # Comparison settings
        data = {
            "MI": TH1_to_array(unfolded_MI, overflow=True)[1:-1],
            "IBU4": TH1_to_array(unfolded_IBU, overflow=True)[1:-1],
            "SA": unfolded_SA[1:-1],
        }

        errors = {
            "MI": error_MI,
            "IBU4": error_IBU,
            "SA": error_SA[1:-1],
        }

        cov = {
            "MI": TMatrix_to_array(cov_MI)[1:-1, 1:-1] if n_toys > 1 else None,
            "IBU4": TMatrix_to_array(cov_IBU)[1:-1, 1:-1] if n_toys > 1 else None,
            "SA": cov_SA[1:-1, 1:-1] if n_toys > 1 else None,
        }

        if quantum:
            additional_quantum_data = {
                "HYB": unfolded_HYB[1:-1],
                "QA": unfolded_QA[1:-1],
            }

            additional_quantum_errors = {
                "HYB": error_HYB[1:-1],
                "QA": error_QA[1:-1],
            }

            additional_quantum_cov = {
                "HYB": cov_HYB[1:-1, 1:-1] if n_toys > 1 else None,
                "QA": cov_QA[1:-1, 1:-1] if n_toys > 1 else None,
            }

            data.update(additional_quantum_data)
            errors.update(additional_quantum_errors)
            cov.update(additional_quantum_cov)

            # Plot comparisons
        plot_comparisons(data, errors, cov, distr, truth[1:-1], measured[1:-1], binning)
        log.info("Done\n")
