import sys
import ROOT
import numpy as np
from analysis_functions.custom_logger import get_custom_logger
from QUnfold.utility import TH1_to_array, TMatrix_to_array


log = get_custom_logger(__name__)

loaded_RooUnfold = ROOT.gSystem.Load("HEP_deps/RooUnfold/libRooUnfold.so")
if not loaded_RooUnfold == 0:
    log.error("RooUnfold not found!")
    sys.exit(0)


def run_RooUnfold(method, response, measured, num_toys=None):
    if method == "MI":
        unfolder = ROOT.RooUnfoldInvert("MI", "Matrix Inversion")
    elif method == "B2B":
        unfolder = ROOT.RooUnfoldBinByBin("B2B", "Bin-by-Bin")
    elif method == "SVD":
        unfolder = ROOT.RooUnfoldSvd("SVD", "SVD Tikhonov")
        unfolder.SetKterm(2)
    elif method == "IBU":
        unfolder = ROOT.RooUnfoldBayes("IBU", "Iterative Bayes")
        unfolder.SetIterations(4)
        unfolder.SetSmoothing(0)

    unfolder.SetVerbose(0)
    unfolder.SetResponse(response)
    unfolder.SetMeasured(measured)

    if num_toys is None:
        sol_histo = unfolder.Hunfold(unfolder.kErrors)
        cov_matrix = unfolder.Eunfold(unfolder.kErrors)
    else:
        unfolder.SetNToys(num_toys)
        sol_histo = unfolder.Hunfold(unfolder.kCovToys)
        cov_matrix = unfolder.Eunfold(unfolder.kCovToys)

    sol = TH1_to_array(sol_histo)
    num_bins = sol_histo.GetNbinsX()
    err = np.array([sol_histo.GetBinError(i) for i in range(1, num_bins + 1)])
    cov = TMatrix_to_array(cov_matrix)

    return sol, err, cov
