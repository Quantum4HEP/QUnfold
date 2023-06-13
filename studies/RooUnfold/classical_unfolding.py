#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Tue Jun 13 17:10:00 2023
Author: Gianluca Bianco
"""

###########################################
#    Import libraries
###########################################

# Standard modules
import argparse as ap

# Data science modules
from ROOT import * # Bad practice, but necessary for ROOT
import pandas as pd
import numpy as np

# Load RooUnfold
loaded_RooUnfold = gSystem.Load("RooUnfold/libRooUnfold.so")

###########################################
#    read_input
###########################################
def read_input(input_file):
    """
    Read the input file a gives two vectors of truth and pseudo-data respectively.

    Args:
        input_file (csv): the input csv file.

    Returns:
        np.array: the vector of truth elements.
        np.array: the vector of pseudo-data.
    """
    
    input = pd.read_csv(input_file)
    truth = input["Truth"].to_numpy()
    pseudo_data = input["PseudoData"].to_numpy()
    
    return truth, pseudo_data

###########################################
#    Main program
###########################################

# Main function
def main():
    
    # Read unfolding parameters from the input variables file
    truth, pseudo_data = read_input(args.input)
    print("INFO: Signal truth-level: {}".format(truth))
    print("INFO: Pseudo-data truth-level: {}".format(pseudo_data))
    
    # Read the response matrix
    response = np.loadtxt(args.response)
    print("INFO: Response matrix: \n{}".format(response))
    
    # Todo
    # ...

# Execute main
if __name__ == "__main__":
    
    # Parser settings
    parser = ap.ArgumentParser(description = "Parsing unfolding input variables.")
    parser.add_argument("-i", "--input", default = "../../data/distributions/peak.csv", help = "Input data used for unfolding.")   
    parser.add_argument("-l", "--lreg", default = 0.00, help = "Regularization strength.")   
    parser.add_argument("-r", "--response", default = "../../data/responses/nominal.txt", help = "Regularization strength.") 
    args = parser.parse_args()
    
    # Run main function
    main()