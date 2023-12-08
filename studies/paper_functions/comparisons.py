# ---------------------- Metadata ----------------------
#
# File name:  QUnfolder.py
# Author:     Gianluca Bianco (biancogianluca9@gmail.com)
# Date:       2023-12-05
# Copyright:  (c) 2023 Gianluca Bianco under the MIT license.

# TODO: errore chi2 STD o STD mean?
# TODO: calcola chi2 covarianze
# TODO: triangular discriminator
# TODO: barre di errore truth ecc?

# STD modules
import sys, os
import tqdm

# Data science modules
import ROOT
from scipy.stats import chisquare
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# QUnfold modules
from QUnfold import QUnfoldQUBO
from QUnfold.utility import TH1_to_array, TH2_to_array, normalize_response

# RooUnfold settings
loaded_RooUnfold = ROOT.gSystem.Load("HEP_deps/RooUnfold/libRooUnfold.so")
if not loaded_RooUnfold == 0:
    print("RooUnfold not found!")
    sys.exit(0)


def make_plots(
    SA_info, HYB_info, IBU_info, SVD_info, truth, measured, binning, var, ntoys, lam
):
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
    ax1.step(
        binning,
        truth_steps,
        label=r"Truth ($\mathtt{Madgraph}$)",
        where="post",
        color="tab:blue",
    )
    ax1.fill_between(binning, truth_steps, step="post", alpha=0.3, color="tab:blue")
    ax2.axhline(y=1, color="tab:blue")

    # Plot measured
    meas_steps = np.append(measured, [measured[-1]])
    ax1.step(
        binning,
        meas_steps,
        label=r"Measured ($\mathtt{Delphes}$)",
        where="post",
        color="tab:orange",
    )
    ax1.fill_between(binning, meas_steps, step="post", alpha=0.3, color="tab:orange")

    # Plot SA
    SA_chi2_mean = SA_info["chi2_mean"]
    SA_chi2_STD = SA_info["chi2_STD"]
    label = rf"Unfolded (SA) $\chi^2 = {SA_chi2_mean} \pm {SA_chi2_STD}$"
    ax1.errorbar(
        x=bin_midpoints,
        y=SA_info["mean"],
        yerr=SA_info["STD"],
        label=label,
        marker="o",
        ms=marker_size,
        c="green",
        linestyle="None",
    )

    ax2.errorbar(
        x=bin_midpoints,
        y=SA_info["mean"] / truth,
        yerr=SA_info["STD"] / truth,
        ms=marker_size,
        fmt="o",
        color="green",
    )

    # Plot HYB
    # HYB_chi2_mean = HYB_info["chi2_mean"]
    # HYB_chi2_STD = HYB_info["chi2_STD"]
    # label = rf"Unfolded (HYB) $\chi^2 = {HYB_chi2_mean} \pm {HYB_chi2_STD}$"
    # ax1.errorbar(
    #     x=bin_midpoints,
    #     y=HYB_info["mean"],
    #     yerr=SA_info["STD"],
    #     label=label,
    #     marker="^",
    #     ms=marker_size,
    #     c="purple",
    #     linestyle="None",
    # )

    # ax2.errorbar(
    #     x=bin_midpoints,
    #     y=HYB_info["mean"] / truth,
    #     yerr=HYB_info["STD"] / truth,
    #     ms=marker_size,
    #     fmt="^",
    #     color="purple",
    # )

    # Plot IBU
    IBU_chi2_mean = IBU_info["chi2_mean"]
    IBU_chi2_STD = IBU_info["chi2_STD"]
    label = rf"Unfolded (IBU) $\chi^2 = {IBU_chi2_mean} \pm {IBU_chi2_STD}$"
    ax1.errorbar(
        x=bin_midpoints,
        y=IBU_info["mean"],
        yerr=IBU_info["STD"],
        label=label,
        marker="s",
        ms=marker_size,
        c="red",
        linestyle="None",
    )

    ax2.errorbar(
        x=bin_midpoints,
        y=IBU_info["mean"] / truth,
        yerr=IBU_info["STD"] / truth,
        ms=marker_size,
        fmt="s",
        color="red",
    )

    # Plot SVD
    SVD_chi2_mean = SVD_info["chi2_mean"]
    SVD_chi2_STD = SVD_info["chi2_STD"]
    label = rf"Unfolded (SVD) $\chi^2 = {SVD_chi2_mean} \pm {SVD_chi2_STD}$"
    ax1.errorbar(
        x=bin_midpoints,
        y=SVD_info["mean"],
        yerr=SVD_info["STD"],
        label=label,
        marker="p",
        ms=marker_size,
        c="purple",
        linestyle="None",
    )

    ax2.errorbar(
        x=bin_midpoints,
        y=SVD_info["mean"] / truth,
        yerr=SVD_info["STD"] / truth,
        ms=marker_size,
        fmt="p",
        color="purple",
    )

    # Set var name to latex
    variable_labels = {
        "pT_lep1": r"$p_T^{lep1}$ [GeV]",
        "pT_lep2": r"$p_T^{lep2}$ [GeV]",
        "eta_lep1": r"$\eta^{lep1}$",
        "eta_lep2": r"$\eta^{lep2}$",
        "DR_b1b2": r"$\Delta R_{b1b2}$",
        "m_l1l2": r"$m_{l1l2}$ [GeV]",
        "phi_lep1": r"$\phi_{lep1}$",
        "phi_lep2": r"$\phi_{lep2}$",
        "y_lep1": r"$\y_{lep1}$",
        "y_lep2": r"$\y_{lep2}$",
    }
    varname = variable_labels.get(var)

    # Plot settings
    ax1.tick_params(axis="x", which="both", bottom=True, top=False, direction="in")
    ax2.tick_params(axis="x", which="both", bottom=True, top=True, direction="in")
    ax1.set_xlim(binning[0], binning[-1])
    ax1.set_ylim(0, ax1.get_ylim()[1])
    ax2.set_yticks([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75])
    ax2.set_yticklabels(["", "0.5", "", "1.0", "", "1.5", ""])
    ax1.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    ax2.set_ylabel("Ratio to\ntruth")
    ax2.set_xlabel(varname, loc="center")
    ax1.set_ylabel("Entries", loc="center")
    ax1.legend(loc="best")

    # Add text for N toys
    offset = matplotlib.text.OffsetFrom(ax1.get_legend(), (1.0, 0.0))
    ax1.annotate(
        f"Number of Toys: {ntoys}",
        xy=(0, 0),
        size=10,
        xycoords="figure fraction",
        xytext=(0, -10),
        textcoords=offset,
        horizontalalignment="right",
        verticalalignment="top",
    )

    # Add text for lam
    offset = matplotlib.text.OffsetFrom(ax1.get_legend(), (1.0, 0.0))
    ax1.annotate(
        f"Regularization: {lam}",
        xy=(0, 0),
        size=10,
        xycoords="figure fraction",
        xytext=(0, -30),
        textcoords=offset,
        horizontalalignment="right",
        verticalalignment="top",
    )

    # Save plot
    plt.tight_layout()
    if not os.path.exists("studies/img/paper"):
        os.makedirs("studies/img/paper")
    plt.savefig(f"studies/img/paper/comparison_{var}.png")
    plt.close()


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


def make_comparisons(reco, particle):
    # Variables
    # fmt: off
    variables = ["DR_b1b2", "pT_lep1", "pT_lep2", "eta_lep1", "eta_lep2", "m_l1l2", "phi_lep1", "phi_lep2", "y_lep1", "y_lep2"]
    ntoys = 2
    chi2_round = 3

    # RUnning over variables
    for var in variables:
        print(f"Processing the {var} variable:")

        # Variables
        SA_results = []
        HYB_results = []
        IBU_results = []
        SVD_results = []

        SA_chi2 = []
        HYB_chi2 = []
        IBU_chi2 = []
        SVD_chi2 = []

        # Running over toys
        print(f"- Running on {ntoys} toys...")
        for i in tqdm.trange(ntoys, ncols=100):
            # Raw input
            m_response = reco.Get(f"particle_{var}_vs_{var}")
            h_measured = reco.Get(var)
            h_truth = particle.Get(f"particle_{var}")
            h_mc_measured = reco.Get(f"mc_{var}")
            h_mc_truth = particle.Get(f"mc_particle_{var}")

            # Prepare unfolding input
            measured = TH1_to_array(h_measured)
            truth = TH1_to_array(h_truth)
            response = normalize_response(TH2_to_array(m_response), TH1_to_array(h_mc_truth))
            binning = [
                reco.Get(var).GetXaxis().GetBinLowEdge(bin)
                for bin in range(1, reco.Get(var).GetNbinsX() + 2)
            ]

            # Unfold with QUnfold
            lam = 0.0
            if var == "pT_lep1":
                lam = 0.04
            elif var == "pT_lep2":
                lam = 0.3
            elif var == "eta_lep1":
                lam = 0.05
            elif var == "eta_lep2":
                lam = 0.01
            unfolder = QUnfoldQUBO(measured=measured, response=response, lam=lam)
            unfolder.initialize_qubo_model(optimize_vars_range=False)
            unfolded_SA = unfolder.solve_simulated_annealing(num_reads=100)
            SA_results.append(unfolded_SA)
            SA_chi2.append(compute_chi2(unfolded_SA, truth))

            # unfolded_HYB = unfolder.solve_hybrid_sampler()
            # HYB_results.append(unfolded_HYB)
            # HYB_chi2.append(compute_chi2(unfolded_HYB, truth))

            # Make RooUnfold response
            m_response = ROOT.RooUnfoldResponse(h_mc_measured, h_mc_truth, m_response)

            # Unfold with RooUnfold IBU
            unfolder = ROOT.RooUnfoldBayes("IBU", "Iterative Bayesian")
            unfolder.SetIterations(4)
            # unfolder.SetSmoothing(0)
            unfolder.SetVerbose(0)
            unfolder.SetResponse(m_response)
            unfolder.SetMeasured(h_measured)
            unfolded_IBU = TH1_to_array(unfolder.Hunfold(3))
            IBU_results.append(unfolded_IBU)
            IBU_chi2.append(compute_chi2(unfolded_IBU, truth))

            # Unfold with RooUnfold SVD
            unfolder = ROOT.RooUnfoldSvd("SVD", "SVD Tikhonov")
            unfolder.SetKterm(2)
            unfolder.SetVerbose(0)
            unfolder.SetResponse(m_response)
            unfolder.SetMeasured(h_measured)
            unfolded_SVD = TH1_to_array(unfolder.Hunfold(3))
            SVD_results.append(unfolded_SVD)
            SVD_chi2.append(compute_chi2(unfolded_SVD, truth))

        # SA results
        SA_info = {
            "mean": np.mean(SA_results, axis=0),
            "STD": np.std(SA_results, axis=0),
            "chi2_mean": np.round(np.mean(SA_chi2), chi2_round),
            "chi2_STD": np.round(np.std(SA_chi2), chi2_round),
        }

        # HYB results
        # HYB_info = {
        #     "mean": np.mean(HYB_results, axis=0),
        #     "STD": np.std(HYB_results, axis=0),
        #     "chi2_mean": np.round(np.mean(HYB_chi2), chi2_round),
        #     "chi2_STD": np.round(np.std(HYB_chi2), chi2_round)
        # }
        HYB_info = {}

        # IBU results
        IBU_info = {
            "mean": np.mean(IBU_results, axis=0),
            "STD": np.std(IBU_results, axis=0),
            "chi2_mean": np.round(np.mean(IBU_chi2), chi2_round),
            "chi2_STD": np.round(np.std(IBU_chi2), chi2_round),
        }

        # SVD results
        SVD_info = {
            "mean": np.mean(SVD_results, axis=0),
            "STD": np.std(SVD_results, axis=0),
            "chi2_mean": np.round(np.mean(SVD_chi2), chi2_round),
            "chi2_STD": np.round(np.std(SVD_chi2), chi2_round),
        }

        # Make plots
        make_plots(SA_info, HYB_info, IBU_info, SVD_info, truth, measured, binning, var, ntoys, lam)
        print()
