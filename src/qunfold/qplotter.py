import numpy as np
import pylab as plt
from qunfold.utils import compute_chi2


################### PLOTTING CONFIGURATIONS ###################
# https://matplotlib.org/stable/gallery/color/named_colors.html
label2color = {
    "Truth": "tab:blue",
    "Measured": "tab:orange",
    "MI": "purple",  # matrix inversion
    "IBU": "red",  # iterative bayesian
    "SA": "green",  # simulated annealing
    "HYB": "dodgerblue",  # hybrid solver
    "QA": "orchid",  # quantum annealing
}
default_color = "black"
figsize = (12, 9)
chi2_ndigits = 2
alpha = 0.25
marker = "o"
markersize = 4
ticks_fontsize = 8
labels_fontsize = 14
legend_fontsize = 10
###############################################################


class QPlotter:
    def __init__(self, response, measured, truth, unfolded, covariance, binning, method, ybottom=0.0, normed=True):
        self.response = response[1:-1, 1:-1]
        self.measured = measured[1:-1]
        self.truth = truth[1:-1]
        self.unfolded = unfolded[1:-1]
        self.covariance = covariance[1:-1, 1:-1]
        self.binning = binning[1:-1]
        self.method = method
        self.ybottom = ybottom
        self.normed = normed

    @staticmethod
    def histogram_plot(ax, xedges, hist, label, ylabel, norm=None):
        if norm is not None:
            hist = hist / norm
        hist = np.append(hist, [hist[-1]])
        color = label2color.get(label, default_color)
        ax.step(xedges, hist, label=label, color=color, where="post")
        ax.fill_between(xedges, hist, color=color, alpha=alpha, step="post")
        ax.set_ylabel(ylabel, fontsize=labels_fontsize)

    @staticmethod
    def errorbar_plot(ax, xmid, hist, err, xlims, label, chi2, norm=None):
        if norm is not None:
            hist = hist / norm
            err = err / norm
        color = label2color.get(label, default_color)
        rchi2 = round(chi2, ndigits=chi2_ndigits)
        label = rf"Unfolded {label} ($\chi^2 = {rchi2}$)"
        ax.errorbar(x=xmid, y=hist, yerr=err, label=label, color=color, marker=marker, ms=markersize, linestyle="None")
        ax.tick_params(labelsize=ticks_fontsize, top=False, right=False, labelbottom=False, reset=True)
        yticks = [-1, -1] + ax.get_yticks().tolist()[2:]
        yticklabels = ["", ""] + ax.get_yticklabels()[2:]
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
        ax.legend(fontsize=legend_fontsize)
        ax.set_xlim(left=xlims[0], right=xlims[1])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    @staticmethod
    def ratio_plot(ax, xmid, ratio, err, label, xticks, xlabel):
        ax.axhline(y=1, color=label2color["Truth"])
        color = label2color.get(label, default_color)
        ax.errorbar(x=xmid, y=ratio, yerr=err, color=color, marker=marker, ms=markersize, linestyle="None")
        ax.tick_params(labelsize=ticks_fontsize, right=False, reset=True)
        ax.set_xticks(xticks)
        ax.set_xlabel(xlabel, fontsize=labels_fontsize)
        ax.set_ylabel("Ratio to\nTruth", fontsize=labels_fontsize)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    def _plot_response(self):
        fig, ax = plt.subplots(figsize=figsize)
        xs = ys = self.binning
        R = self.response
        heatmap = ax.pcolormesh(xs, ys, R.T)
        ax.set_xticks(xs)
        ax.set_yticks(ys)
        ax.tick_params(labelsize=ticks_fontsize, reset=True)
        ax.set_xlabel("Measured", fontsize=labels_fontsize)
        ax.set_ylabel("Truth", fontsize=labels_fontsize)
        fig.colorbar(heatmap, ax=ax)
        fig.tight_layout()

    def _plot_histograms(self):
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[3, 1], hspace=0)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)

        measured, truth = self.measured, self.truth
        sol, cov = self.unfolded, self.covariance
        err = np.sqrt(np.diag(cov))
        sol_ratio = sol / truth
        err_ratio = err / truth
        label = self.method
        widths = np.diff(self.binning)
        xmid = self.binning[:-1] + 0.5 * widths
        xlims = (self.binning[0], self.binning[-1])
        ylabel = "Frequency" if self.normed else "Entries"
        chi2 = compute_chi2(sol, truth, covariance=cov)
        norm = np.sum(truth) if self.normed else None

        self.histogram_plot(ax=ax1, xedges=self.binning, hist=truth, label="Truth", ylabel=ylabel, norm=norm)
        self.histogram_plot(ax=ax1, xedges=self.binning, hist=measured, label="Measured", ylabel=ylabel, norm=norm)
        self.errorbar_plot(ax=ax1, xmid=xmid, hist=sol, err=err, xlims=xlims, label=label, chi2=chi2, norm=norm)
        self.ratio_plot(
            ax=ax2, xmid=xmid, ratio=sol_ratio, err=err_ratio, label=label, xticks=self.binning, xlabel="Bins"
        )
        ax1.set_ylim(bottom=self.ybottom)
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
