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
alpha = 0.25
marker = "o"
markersize = 4
ticks_fontsize = 9
labels_fontsize = 15
legend_fontsize = 12
#################################################


def histogram_plot(ax, x, y, label):
    y = np.append(y, [y[-1]])
    color = label2color[label]
    ax.step(x, y, label=label, color=color, where="post")
    ax.fill_between(x, y, color=color, alpha=alpha, step="post")


def errorbar_plot(ax, x, y, yerr, binning, method, chi2):
    rchi2 = round(chi2, ndigits=chi2_ndigits)
    label = rf"Unfolded {method} ($\chi^2 = {rchi2}$)"
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


def ratio_plot(ax, x, y, yerr, binning, method, xlabel=None):
    if xlabel is None:
        xlabel = "Bins"
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
        self, response, measured, truth, unfolded, covariance, binning, method
    ):
        self.response = response[1:-1, 1:-1]
        self.measured = measured[1:-1]
        self.truth = truth[1:-1]
        self.unfolded = unfolded[1:-1]
        self.covariance = covariance[1:-1, 1:-1]
        self.binning = binning[1:-1]
        self.qunfold_method = method

    def _plot_response(self):
        fig, ax = plt.subplots(figsize=figsize)
        heatmap = ax.imshow(
            np.transpose(self.response),
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
        ax.set_xlabel("Measured", fontsize=labels_fontsize)
        ax.set_ylabel("Truth", fontsize=labels_fontsize)
        fig.colorbar(heatmap, ax=ax)
        fig.tight_layout()

    def _plot_histograms(self):
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[3, 1], hspace=0)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)

        histogram_plot(ax=ax1, x=self.binning, y=self.truth, label="Truth")
        histogram_plot(ax=ax1, x=self.binning, y=self.measured, label="Measured")

        binning = self.binning
        method = self.qunfold_method
        xmid = binning[:-1] + np.diff(binning) / 2
        obs, exp, cov = self.unfolded, self.truth, self.covariance
        err = np.sqrt(np.diag(cov))
        chi2 = compute_chi2(obs, exp, cov)
        errorbar_plot(
            ax=ax1, x=xmid, y=obs, yerr=err, method=method, binning=binning, chi2=chi2
        )
        ratio_sol = obs / self.truth
        ratio_err = err / self.truth
        ratio_plot(
            ax=ax2, x=xmid, y=ratio_sol, yerr=ratio_err, method=method, binning=binning
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

    def show_histograms(self):
        self._plot_histograms()
        plt.show()
        plt.close()

    def save_histograms(self, path):
        self._plot_histograms()
        plt.savefig(path)
        plt.close()
