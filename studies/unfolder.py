import ROOT
from qunfold.root2numpy import TH1_to_numpy, TMatrix_to_numpy
from qunfold import QUnfolder


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
        sol_histo = unfolder.Hunfold()
        cov_matrix = unfolder.Eunfold()
    else:
        unfolder.SetNToys(num_toys)
        sol_histo = unfolder.Hunfold(unfolder.kCovToys)
        cov_matrix = unfolder.Eunfold(unfolder.kCovToys)

    sol = TH1_to_numpy(sol_histo, overflow=True)
    cov = TMatrix_to_numpy(cov_matrix)
    return sol, cov


def run_QUnfold(method, response, measured, binning, lam, num_reads=None, num_toys=None):
    unfolder = QUnfolder(response, measured, binning=binning, lam=lam)
    unfolder.initialize_qubo_model()

    if method == "GRB":
        sol, cov = unfolder.solve_gurobi_integer()
    elif method == "SA":
        sol, cov = unfolder.solve_simulated_annealing(num_reads=num_reads, num_toys=num_toys)
    elif method == "HYB":
        sol, cov = unfolder.solve_hybrid_sampler()
    elif method == "QA":
        unfolder.set_quantum_device()
        unfolder.set_graph_embedding()
        sol, cov, _ = unfolder.solve_quantum_annealing(num_reads=num_reads, num_toys=num_toys)
    return sol, cov
