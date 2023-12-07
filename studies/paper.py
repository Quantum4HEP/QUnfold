# ---------------------- Metadata ----------------------
#
# File name:  paper.py
# Author:     Gianluca Bianco (biancogianluca9@gmail.com)
# Date:       2023-12-05
# Copyright:  (c) 2023 Gianluca Bianco under the MIT license.

# TODO: fakes (vedi warning)
# TODO: aggiungi altre metriche
# TODO: errori con iterazioni

# STD modules
import argparse as ap

# Data science modules
import ROOT

# My modules
from paper_functions.comparisons import make_comparisons


# Main function
def main():
    # Read input data
    file = ROOT.TFile(args.file, "READ")
    reco = file.Get("reco")
    particle = file.Get("particle")

    # Make comparisons
    make_comparisons(reco, particle)


# Main program
if __name__ == "__main__":
    # Parser settings
    parser = ap.ArgumentParser(description="Parsing input arguments.")
    parser.add_argument(
        "-f",
        "--file",
        default="",
        help="Input data file.",
    )
    args = parser.parse_args()

    # Main part
    main()
