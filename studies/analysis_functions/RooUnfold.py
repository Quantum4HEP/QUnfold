import sys
import ROOT
from analysis_functions.custom_logger import get_custom_logger
from QUnfold.root2numpy import TH1_to_numpy, TMatrix_to_numpy


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

    sol = TH1_to_numpy(sol_histo, overflow=True)
    err = TH1_to_numpy(sol_histo, error=True, overflow=True)
    cov = TMatrix_to_numpy(cov_matrix)

    return sol, err, cov
