# TODO: errore chi2 STD o STD mean?
# TODO: calcola chi2 covarianze
# TODO: triangular discriminator
# TODO: barre di errore truth ecc?

# Main modules
import sys, os
import tqdm
import ROOT
from scipy.stats import chisquare
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from QUnfold import QUnfoldQUBO
from QUnfold.utility import TH1_to_array, TH2_to_array, normalize_response

# Settings
loaded_RooUnfold = ROOT.gSystem.Load("HEP_deps/RooUnfold/libRooUnfold.so")
if not loaded_RooUnfold == 0:
    print("RooUnfold not found!")
    sys.exit(0)


def make_plots(SA_info, IBU_info, SVD_info, truth, measured, binning, var, ntoys, lam):
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
    SA_chi2 = SA_info["chi2"]
    label = rf"Unfolded (SA) $\chi^2 = {SA_chi2}$"
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

    # Plot IBU
    IBU_chi2 = IBU_info["chi2"]
    label = rf"Unfolded (IBU) $\chi^2 = {IBU_chi2}$"
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
    SVD_chi2 = SVD_info["chi2"]
    label = rf"Unfolded (SVD) $\chi^2 = {SVD_chi2}$"
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
        "m_b1b2": r"$m_{b1b2 [GeV]$",
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
        f"Number of reads: {ntoys}",
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
        os.makedirs("studies/img/paper/png")
        os.makedirs("studies/img/paper/pdf")
    plt.savefig(f"studies/img/paper/png/comparison_{var}.png")
    plt.savefig(f"studies/img/paper/pdf/comparison_{var}.pdf")
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
    variables = [
        "pT_lep1",
        "pT_lep2",
        "DR_b1b2",
        "eta_lep1",
        "eta_lep2",
        "m_l1l2",
        "phi_lep1",
        "phi_lep2",
        "m_b1b2",
    ]

    # RUnning over variables
    for var in variables:
        print(f"Processing the {var} variable...")

        # Raw input
        m_response = reco.Get(f"particle_{var}_vs_{var}")
        h_measured = reco.Get(var)
        h_truth = particle.Get(f"particle_{var}")
        h_mc_measured = reco.Get(f"mc_{var}")
        h_mc_truth = particle.Get(f"mc_particle_{var}")

        # Prepare unfolding input
        measured = TH1_to_array(h_measured)
        truth = TH1_to_array(h_truth)
        response = normalize_response(
            TH2_to_array(m_response), TH1_to_array(h_mc_truth)
        )
        binning = [
            reco.Get(var).GetXaxis().GetBinLowEdge(bin)
            for bin in range(1, reco.Get(var).GetNbinsX() + 2)
        ]

        # Unfold with QUnfold
        lam = 0.0
        num_reads = 100
        unfolder = QUnfoldQUBO(measured=measured, response=response, lam=lam)
        unfolder.initialize_qubo_model(optimize_vars_range=False)
        unfolded_SA, error_SA = unfolder.solve_simulated_annealing(num_reads=num_reads)
        chi2_SA = compute_chi2(unfolded_SA, truth)

        # Make RooUnfold response
        m_response = ROOT.RooUnfoldResponse(h_mc_measured, h_mc_truth, m_response)

        # Unfold with RooUnfold IBU
        unfolder = ROOT.RooUnfoldBayes("IBU", "Iterative Bayesian")
        unfolder.SetIterations(4)
        unfolder.SetVerbose(0)
        unfolder.SetResponse(m_response)
        unfolder.SetMeasured(h_measured)
        unfolded_IBU_histo = unfolder.Hunfold(unfolder.kCovToys)
        unfolded_IBU = TH1_to_array(unfolded_IBU_histo)
        start, stop = 1, unfolded_IBU_histo.GetNbinsX() + 1
        error_IBU = np.array(
            [unfolded_IBU_histo.GetBinError(i) for i in range(start, stop)]
        )
        chi2_IBU = compute_chi2(unfolded_IBU, truth)

        # Unfold with RooUnfold SVD
        unfolder = ROOT.RooUnfoldSvd("SVD", "SVD Tikhonov")
        unfolder.SetKterm(2)
        unfolder.SetVerbose(0)
        unfolder.SetResponse(m_response)
        unfolder.SetMeasured(h_measured)
        unfolded_SVD_histo = unfolder.Hunfold(unfolder.kCovToys)
        unfolded_SVD = TH1_to_array(unfolded_SVD_histo)
        start, stop = 1, unfolded_IBU_histo.GetNbinsX() + 1
        error_SVD = np.array(
            [unfolded_SVD_histo.GetBinError(i) for i in range(start, stop)]
        )
        chi2_SVD = compute_chi2(unfolded_SVD, truth)

        # SA results
        SA_info = {
            "mean": unfolded_SA,
            "STD": error_SA,
            "chi2": np.round(chi2_SA, 3),
        }

        # IBU results
        IBU_info = {
            "mean": unfolded_IBU,
            "STD": error_IBU,
            "chi2": np.round(chi2_IBU, 3),
        }

        # SVD results
        SVD_info = {
            "mean": unfolded_SVD,
            "STD": error_SVD,
            "chi2": np.round(chi2_SVD, 3),
        }

        # Make plots
        make_plots(
            SA_info, IBU_info, SVD_info, truth, measured, binning, var, num_reads, lam
        )
