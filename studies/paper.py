# ---------------------- Metadata ----------------------
#
# File name:  paper.py
# Author:     Gianluca Bianco (biancogianluca9@gmail.com)
# Date:       2023-12-05
# Copyright:  (c) 2023 Gianluca Bianco under the MIT license.

# TODO: fakes (vedi warning)
# TODO: aggiungi altre metriche
# TODO: errori con iterazioni

import argparse as ap
import ROOT
from paper_functions.comparisons import make_comparisons

if __name__ == "__main__":
    # Read input data
    file = ROOT.TFile("data/simulated/output/unfolding_input.root", "READ")
    reco = file.Get("reco")
    particle = file.Get("particle")

    # Make comparisons
    make_comparisons(reco, particle)
