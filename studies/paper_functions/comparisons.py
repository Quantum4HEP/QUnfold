# ---------------------- Metadata ----------------------
#
# File name:  QUnfolder.py
# Author:     Gianluca Bianco (biancogianluca9@gmail.com)
# Date:       2023-12-05
# Copyright:  (c) 2023 Gianluca Bianco under the MIT license.

# STD modules
import sys, os

# Data science modules
import ROOT
from scipy.stats import chisquare
import numpy as np
import matplotlib.pyplot as plt

# QUnfold modules
from QUnfold import QUnfoldQUBO
from QUnfold.utility import TH1_to_array, TH2_to_array, normalize_response

# RooUnfold settings
loaded_RooUnfold = ROOT.gSystem.Load("HEP_deps/RooUnfold/libRooUnfold.so")
if not loaded_RooUnfold == 0:
    print("RooUnfold not found!")
    sys.exit(0)


def compute_chi2(unfolded, truth):
    # Trick for chi2 convergence
    null_indices = truth == 0
    truth[null_indices] += 1
    unfolded[null_indices] += 1

    # Compute chi2
    chi2, _ = chisquare(
        unfolded,
        np.sum(unfolded) / np.sum(truth) * truth,
    )
    dof = len(unfolded) - 1
    chi2_dof = chi2 / dof
    return chi2_dof


def make_plots(unfolded_SA, unfolded_HYB, unfolded_IBU, unfolded_SVD, truth, binning, var):
    # Divide into subplots
    fig = plt.figure()
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    marker_size = 3.5
    binwidths = np.diff(binning)
    bin_midpoints = binning[:-1] + binwidths / 2

    # Plot truth
    truth_steps = np.append(truth, [truth[-1]])
    ax1.step(binning, truth_steps, label="Truth", where="post", color="tab:blue")
    ax1.fill_between(binning, truth_steps, step="post", alpha=0.3, color="tab:blue")
    ax2.axhline(y=1, color="tab:blue")

    # Plot SA
    chi2 = round(compute_chi2(unfolded_SA, truth), 2)
    label = rf"Unfolded (SA) $\chi^2 = {chi2}$"
    ax1.errorbar(
        x=bin_midpoints,
        y=unfolded_SA,
        yerr=np.sqrt(unfolded_SA),
        label=label,
        marker="o",
        ms=marker_size,
        c="green",
        linestyle="None",
    )

    ax2.errorbar(
        x=bin_midpoints,
        y=unfolded_SA / truth,
        yerr=np.sqrt(unfolded_SA) / truth,
        ms=marker_size,
        fmt="o",
        color="green",
    )

    # Plot HYB
    # chi2 = round(compute_chi2(unfolded_HYB, truth), 2)
    # label = rf"Unfolded (HYB) $\chi^2 = {chi2}$"
    # ax1.errorbar(
    #     x=bin_midpoints,
    #     y=unfolded_HYB,
    #     yerr=np.sqrt(unfolded_HYB),
    #     label=label,
    #     marker="^",
    #     ms=marker_size,
    #     c="purple",
    #     linestyle="None",
    # )

    # ax2.errorbar(
    #     x=bin_midpoints,
    #     y=unfolded_HYB / truth,
    #     yerr=np.sqrt(unfolded_HYB) / truth,
    #     ms=marker_size,
    #     fmt="^",
    #     color="purple",
    # )

    # Plot IBU
    chi2 = round(compute_chi2(unfolded_IBU, truth), 2)
    label = rf"Unfolded (IBU) $\chi^2 = {chi2}$"
    ax1.errorbar(
        x=bin_midpoints,
        y=unfolded_IBU,
        yerr=np.sqrt(unfolded_IBU),
        label=label,
        marker="s",
        ms=marker_size,
        c="red",
        linestyle="None",
    )

    ax2.errorbar(
        x=bin_midpoints,
        y=unfolded_IBU / truth,
        yerr=np.sqrt(unfolded_IBU) / truth,
        ms=marker_size,
        fmt="s",
        color="red",
    )

    # Plot SVD
    chi2 = round(compute_chi2(unfolded_SVD, truth), 2)
    label = rf"Unfolded (SVD) $\chi^2 = {chi2}$"
    ax1.errorbar(
        x=bin_midpoints,
        y=unfolded_SVD,
        yerr=np.sqrt(unfolded_SVD),
        label=label,
        marker="p",
        ms=marker_size,
        c="tab:orange",
        linestyle="None",
    )

    ax2.errorbar(
        x=bin_midpoints,
        y=unfolded_SVD / truth,
        yerr=np.sqrt(unfolded_SVD) / truth,
        ms=marker_size,
        fmt="p",
        color="tab:orange",
    )

    # Plot settings
    ax1.tick_params(axis="x", which="both", bottom=True, top=False, direction="in")
    ax2.tick_params(axis="x", which="both", bottom=True, top=True, direction="in")
    ax1.set_xlim(binning[0], binning[-1])
    ax1.set_ylim(0, ax1.get_ylim()[1])
    ax2.set_yticks([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75])
    ax2.set_yticklabels(["", "0.5", "", "1.0", "", "1.5", ""])
    ax1.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    ax2.set_ylabel("Ratio to\ntruth")
    ax2.set_xlabel("Bins")
    ax1.set_ylabel("Frequency")
    ax1.legend(loc="upper right")
    plt.tight_layout()

    # Save plot
    if not os.path.exists("studies/img/paper"):
        os.makedirs("studies/img/paper")
    plt.savefig("studies/img/paper/comparison_{}.png".format(var))
    plt.close()


def make_comparisons(reco, particle):
    for var in [
        "pT_lep1",
        "pT_lep2",
        "eta_lep1",
        "eta_lep2",
    ]:  # "pT_lep1", "pT_lep2", "eta_lep1", "eta_lep2"
        # Raw input
        m_response = reco.Get("particle_{0}_vs_{0}".format(var))
        h_measured = reco.Get(var)
        h_truth = particle.Get("particle_{0}".format(var))
        h_mc_measured = reco.Get("mc_{}".format(var))
        h_mc_truth = particle.Get("mc_particle_{0}".format(var))

        # Prepare unfolding input
        measured = TH1_to_array(h_measured)
        truth = TH1_to_array(h_truth)
        response = normalize_response(TH2_to_array(m_response), TH1_to_array(h_mc_truth))
        binning = [
            reco.Get(var).GetXaxis().GetBinLowEdge(bin)
            for bin in range(1, reco.Get(var).GetNbinsX() + 2)
        ]

        # Unfold with QUnfold
        if var == "pT_lep1":
            lam = 0.05
        elif var == "pT_lep2":
            lam = 0.02
        elif var == "eta_lep1":
            lam = 0.05
        elif var == "eta_lep2":
            lam = 0.01
        unfolder = QUnfoldQUBO(measured=measured, response=response, lam=lam)
        unfolder.initialize_qubo_model(False)
        unfolded_SA = unfolder.solve_simulated_annealing(num_reads=100)
        # unfolded_HYB = unfolder.solve_hybrid_sampler()
        unfolded_HYB = None

        # Make RooUnfold response
        m_response = ROOT.RooUnfoldResponse(h_mc_measured, h_mc_truth, m_response)

        # Unfold with RooUnfold IBU
        unfolder = ROOT.RooUnfoldBayes("IBU", "Iterative Bayesian")
        unfolder.SetIterations(4)
        unfolder.SetSmoothing(0)
        unfolder.SetVerbose(0)
        unfolder.SetResponse(m_response)
        unfolder.SetMeasured(h_measured)
        unfolded_IBU = unfolder.Hunfold()
        unfolded_IBU = TH1_to_array(unfolded_IBU)

        # Unfold with RooUnfold SVD
        unfolder = ROOT.RooUnfoldSvd("SVD", "SVD Tikhonov")
        unfolder.SetKterm(2)
        unfolder.SetVerbose(0)
        unfolder.SetResponse(m_response)
        unfolder.SetMeasured(h_measured)
        unfolded_SVD = unfolder.Hunfold()
        unfolded_SVD = TH1_to_array(unfolded_SVD)

        # Make plots
        make_plots(unfolded_SA, unfolded_HYB, unfolded_IBU, unfolded_SVD, truth, binning, var)
