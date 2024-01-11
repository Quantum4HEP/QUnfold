import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chisquare


def plot_errorbar(
    ax1, ax2, bin_edges, bin_contents, bin_errors, color, mark, method, chi2, truth
):
    """
    Plot error bars on two subplots representing data and the ratio of data to truth.

    Args:
        ax1 (matplotlib.axes._axes.Axes): The first subplot for the data plot.
        ax2 (matplotlib.axes._axes.Axes): The second subplot for the ratio plot.
        bin_edges (array-like): The bin edges for the histogram.
        bin_contents (array-like): The content of each bin in the histogram.
        bin_errors (array-like): The errors associated with each bin.
        color (str): The color of the error bars and markers in the plot.
        marker (str): The marker style for the data points in the plot.
        method (str): The method used to obtain the data, for labeling purposes.
        chi2 (float): The chi-squared value associated with the data fit.
        truth (array-like): The true values corresponding to each bin, used for the ratio plot.
    """

    ax1.errorbar(
        x=bin_edges,
        y=bin_contents,
        yerr=bin_errors,
        color=color,
        marker=mark,
        ms=3.5,
        label=r"{} ($\chi^2 = {:.2f}$)".format(method, chi2),
        linestyle="None",
    )

    ax2.errorbar(
        x=bin_edges,
        y=bin_contents / truth,
        yerr=bin_errors / truth,
        color=color,
        marker=mark,
        ms=3.5,
        linestyle="None",
    )


def compute_chi2_dof(bin_contents, truth_bin_contents):
    """
    Compute the chi-squared per degree of freedom (chi2/dof) between two distributions.

    Args:
        bin_contents (numpy.array): The observed bin contents.
        truth_bin_contents (numpy.array): The expected bin contents.

    Returns:
        float: The chi-squared per degree of freedom.
    """

    # Trick for chi2 convergence
    null_indices = truth_bin_contents == 0
    truth_bin_contents[null_indices] += 1
    bin_contents[null_indices] += 1

    # Compute chi2
    chi2, _ = chisquare(
        bin_contents,
        np.sum(bin_contents) / np.sum(truth_bin_contents) * truth_bin_contents,
    )
    dof = len(bin_contents) - 1
    chi2_dof = chi2 / dof

    return chi2_dof


def plot_comparisons(data, errors, distr, truth, measured, binning):
    """
    Plots the unfolded distributions for different unfolding methods.

    Args:
        data (dict): A dictionary containing the data for each unfolding method.
                     Keys represent method names, and values represent corresponding file paths.
        errors (dict): A dictionary containing the errors for each unfolding method.
                       Keys are method names, and values are corresponding error arrays.
        distr (array-like): The generated distribution.
        truth (array-like): The true distribution used for comparison.
        measured (array-like): The measured distribution used for unfolding.
        bins (int or array-like): The number of bins or bin edges for the histograms.
        min_bin (float): The minimum value of the histogram range.
        max_bin (float): The maximum value of the histogram range.
    """

    # Binning
    bin_edges = binning
    binwidths = np.diff(bin_edges)
    bin_midpoints = bin_edges[:-1] + binwidths / 2

    # Divide into subplots
    fig = plt.figure(figsize=(7.6, 6.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    # Plot truth
    truth_steps = np.append(truth, [truth[-1]])
    ax1.step(
        bin_edges,
        truth_steps,
        label="Truth",
        where="post",
        color="tab:blue",
    )
    ax1.fill_between(bin_edges, truth_steps, step="post", alpha=0.3, color="tab:blue")
    ax2.axhline(y=1, color="tab:blue")

    # Plot measured
    meas_steps = np.append(measured, [measured[-1]])
    ax1.step(
        bin_edges,
        meas_steps,
        label="Measured",
        where="post",
        color="tab:orange",
    )
    ax1.fill_between(bin_edges, meas_steps, step="post", alpha=0.3, color="tab:orange")

    # Iterate over the unfolding methods
    for method, unfolded in data.items():
        # Plot each unfolding method
        chi2_dof = compute_chi2_dof(unfolded, truth)
        if method == "MI":
            plot_errorbar(
                ax1,
                ax2,
                bin_midpoints,
                unfolded,
                errors[method],
                "green",
                "s",
                r"$\mathtt{RooUnfold}$ (MI)",
                chi2_dof,
                truth,
            )
        elif method == "IBU4":
            plot_errorbar(
                ax1,
                ax2,
                bin_midpoints,
                unfolded,
                errors[method],
                "red",
                "o",
                r"$\mathtt{RooUnfold}$ (IBU)",
                chi2_dof,
                truth,
            )
        elif method == "SA":
            plot_errorbar(
                ax1,
                ax2,
                bin_midpoints,
                unfolded,
                errors[method],
                "purple",
                "*",
                r"$\mathtt{QUnfold}$ (SIM)",
                chi2_dof,
                truth,
            )
        elif method == "HYB":
            plot_errorbar(
                ax1,
                ax2,
                bin_midpoints,
                unfolded,
                errors[method],
                "orange",
                "*",
                r"$\mathtt{QUnfold}$ (HYB)",
                chi2_dof,
                truth,
            )

        # Plot settings
        ax1.tick_params(axis="x", which="both", bottom=True, top=False, direction="in")
        ax2.tick_params(axis="x", which="both", bottom=True, top=True, direction="in")
        ax1.set_xlim(bin_edges[0], bin_edges[-1])
        ax1.set_ylim(0, ax1.get_ylim()[1])
        ax2.set_yticks([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75])
        ax2.set_yticklabels(["", "0.5", "", "1.0", "", "1.5", ""])
        ax1.tick_params(
            axis="x", which="both", bottom=False, top=False, labelbottom=False
        )
        ax2.set_ylabel("Ratio to\ntruth")
        ax2.set_xlabel("Bins", loc="center")
        ax1.set_ylabel("Entries", loc="center")
        ax1.legend(loc="best")

        # Save plot
        plt.tight_layout()
        if not os.path.exists("studies/img/comparisons"):
            os.makedirs("studies/img/comparisons")
        plt.savefig("studies/img/comparisons/{}.png".format(distr))

    plt.close()
