# ---------------------- Metadata ----------------------
#
# File name:  paper.py
# Author:     Gianluca Bianco (biancogianluca9@gmail.com)
# Date:       2023-12-05
# Copyright:  (c) 2023 Gianluca Bianco under the MIT license.

# STD modules
import sys, os

# Data science modules
import ROOT

# QUnfold modules
from QUnfold import QUnfoldPlotter, QUnfoldQUBO
from QUnfold.utility import TH1_to_array, TH2_to_array

# RooUnfold settings
loaded_RooUnfold = ROOT.gSystem.Load("HEP_deps/RooUnfold/libRooUnfold.so")
if not loaded_RooUnfold == 0:
    print("RooUnfold not found!")
    sys.exit(0)


# Main function
def main():
    # Read input data
    file = ROOT.TFile("data/simulated/output/unfolding_input.root", "READ")
    reco = file.Get("reco")
    particle = file.Get("particle")

    # Prepare unfolding input
    measured = TH1_to_array(reco.Get("pT_lep1"))
    truth = TH1_to_array(particle.Get("particle_pT_lep1"))
    response = TH2_to_array(reco.Get("particle_pT_lep1_vs_pT_lep1"))
    binning = [
        reco.Get("pT_lep1").GetXaxis().GetBinLowEdge(bin)
        for bin in range(1, reco.Get("pT_lep1").GetNbinsX() + 2)
    ]

    # Unfold
    unfolder = QUnfoldQUBO(measured=measured, response=response, lam=0.005)
    unfolder.initialize_qubo_model(False)
    unfolded = unfolder.solve_simulated_annealing(num_reads=100)

    # Plot result
    plotter = QUnfoldPlotter(
        response=response, measured=measured, truth=truth, unfolded=unfolded, binning=binning
    )

    if not os.path.exists("studies/img/paper"):
        os.makedirs("studies/img/paper")
    plotter.saveResponse("studies/img/paper/response.png")
    plotter.savePlot("studies/img/paper/result.png")


# Main program
if __name__ == "__main__":
    main()
