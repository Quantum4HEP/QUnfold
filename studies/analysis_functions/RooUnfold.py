import ROOT as r
import numpy as np


def RooUnfold_unfolder(type, m_response, h_measured, n_toys):
    unfolder = ""
    if type == "MI":
        unfolder = r.RooUnfoldInvert("MI", "Matrix Inversion")
    elif type == "SVD":
        unfolder = r.RooUnfoldSvd("SVD", "SVD Tikhonov")
        unfolder.SetKterm(2)
    elif type == "IBU":
        unfolder = r.RooUnfoldBayes("IBU", "Iterative Bayesian")
        unfolder.SetIterations(4)
        unfolder.SetSmoothing(0)
    elif type == "B2B":
        unfolder = r.RooUnfoldBinByBin("B2B", "Bin-y-Bin")

    unfolder.SetVerbose(0)
    unfolder.SetResponse(m_response)
    unfolder.SetMeasured(h_measured)

    histo = None
    cov_matrix = None
    if n_toys == 1:
        histo = unfolder.Hunfold(unfolder.kErrors)
    elif n_toys > 1:
        unfolder.SetNToys(n_toys)
        histo = unfolder.Hunfold(unfolder.kCovToys)
        cov_matrix = unfolder.Eunfold(unfolder.kCovToys)
    histo.SetName("unfolded_{}".format(type))
    start, stop = 1, histo.GetNbinsX() + 1
    error = np.array([histo.GetBinError(i) for i in range(start, stop)])

    return histo, error, cov_matrix
