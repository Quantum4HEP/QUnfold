import ROOT
from paper_functions.comparisons import make_comparisons

if __name__ == "__main__":
    # Read input data
    file = ROOT.TFile("data/simulated_final/output/unfolding_input.root", "READ")
    reco = file.Get("reco")
    particle = file.Get("particle")

    # Make comparisons
    make_comparisons(reco, particle)
