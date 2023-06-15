#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Wed Jun 14 17:10:00 2023
Author: Gianluca Bianco
"""

# STD modules
import argparse as ap

# Data science modules
import ROOT as r


def main():
    pass


if __name__ == "__main__":

    # Parser settings
    parser = ap.ArgumentParser(description="Parsing generator input variables.")
    parser.add_argument(
        "-t",
        "--distr",
        choices=["normal", "bw"],
        default="normal",
        help="The type of the distribution to be simulated.",
    )
    args = parser.parse_args()

    # Run main function
    main()
