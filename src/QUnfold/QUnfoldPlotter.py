import numpy as np
import pylab as plt
from QUnfold.utils import compute_chi2


############ PLOTTING CONFIGURATIONS ############
label2color = {
    "Truth": "tab:blue",
    "Measured": "tab:orange",
    "MI": "tab:green",  # matrix inversion
    "IBU": "tab:red",  # iterative bayesian
    "SA": "black",  # simulated annealing
    "HYB": "tab:purple",  # hybrid solver
    "QA": "tab:olive",  # quantum annealing
    "GRB": "tab:pink",  # gurobi integer
}
figsize = (12, 9)
chi2_ndigits = 2
alpha = 0.3
marker = "o"
markersize = 3.5
ticks_fontsize = 8
labels_fontsize = 14
legend_fontsize = 12
#################################################


def histogram_plot(ax, x, y, label):
    y = np.append(y, [y[-1]])
    color = label2color[label]
    ax.step(x, y, label=label, color=color, where="post")
    ax.fill_between(x, y, color=color, alpha=alpha, step="post")


def errorbar_plot(ax, x, y, yerr, method, chi2, binning):
    rounded_chi2 = round(chi2, ndigits=chi2_ndigits)
    label = rf"{method} ($\chi^2 = {rounded_chi2}$)"
    color = label2color[method]
    ax.errorbar(
        x=x,
        y=y,
        yerr=yerr,
        label=label,
        color=color,
        marker=marker,
        ms=markersize,
        linestyle="None",
    )
    ax.set_xlim(left=binning[0], right=binning[-1])
    ax.tick_params(
        reset=True,
        direction="inout",
        labelsize=ticks_fontsize,
        top=False,
        right=False,
        labelbottom=False,
    )
    yticks = [-np.inf, -np.inf] + ax.get_yticks().tolist()[2:]
    yticklabels = ["", ""] + ax.get_yticklabels()[2:]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_ylabel(ylabel="Entries", fontsize=labels_fontsize)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=legend_fontsize)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def ratio_plot(ax, x, y, yerr, method, binning, xlabel):
    ax.axhline(y=1, color=label2color["Truth"])
    color = label2color[method]
    ax.errorbar(
        x=x,
        y=y,
        yerr=yerr,
        color=color,
        marker=marker,
        ms=markersize,
        linestyle="None",
    )
    ax.tick_params(
        reset=True,
        direction="inout",
        labelsize=ticks_fontsize,
        right=False,
    )
    ax.set_xticks(binning)
    ax.set_xlabel(xlabel=xlabel, fontsize=labels_fontsize)
    ax.set_ylabel(ylabel="Ratio to\nTruth", fontsize=labels_fontsize)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


class QUnfoldPlotter:
    def __init__(
        self, response, measured, truth, unfolded, error, covariance, binning, chi2=True
    ):
        self.response = response[1:-1, 1:-1]
        self.measured = measured[1:-1]
        self.truth = truth[1:-1]
        self.unfolded = unfolded[1:-1]
        self.error = error[1:-1]
        self.covariance = covariance[1:-1, 1:-1]
        self.binning = binning[1:-1]
        self.chi2 = chi2

    def _plot_response(self):
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(
            np.transpose(self.response),
            cmap="viridis",
            extent=[
                self.binning[0],
                self.binning[-1],
                self.binning[0],
                self.binning[-1],
            ],
            origin="lower",
        )
        ax.tick_params(
            reset=True,
            direction="inout",
            labelsize=ticks_fontsize,
            top=False,
            right=False,
        )
        ax.set_xlabel("Truth", fontsize=labels_fontsize)
        ax.set_ylabel("Measured", fontsize=labels_fontsize)
        fig.tight_layout()

    def _plot_histograms(self, method):
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[3, 1], hspace=0)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)

        histogram_plot(ax=ax1, x=self.binning, y=self.truth, label="Truth")
        histogram_plot(ax=ax1, x=self.binning, y=self.measured, label="Measured")

        binning = self.binning
        bin_midpoints = binning[:-1] + np.diff(binning) / 2
        chi2 = compute_chi2(self.unfolded, self.truth, self.covariance)
        errorbar_plot(
            ax=ax1,
            x=bin_midpoints,
            y=self.unfolded,
            yerr=self.error,
            method=method,
            chi2=chi2,
            binning=binning,
        )

        ratio_sol = self.unfolded / self.truth
        ratio_err = self.error / self.truth
        ratio_plot(
            ax=ax2,
            x=bin_midpoints,
            y=ratio_sol,
            yerr=ratio_err,
            method=method,
            binning=binning,
        )
        fig.tight_layout()

    def show_response(self):
        self._plot_response()
        plt.show()
        plt.close()

    def save_response(self, path):
        self._plot_response()
        plt.savefig(path)
        plt.close()

    def show_histograms(self, method):
        self._plot_histograms(method)
        plt.show()
        plt.close()

    def save_histograms(self, path, method):
        self._plot_histograms(method)
        plt.savefig(path)
        plt.close()
