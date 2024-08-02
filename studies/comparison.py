import numpy as np
import pylab as plt
from QUnfold.utils import compute_chi2
from QUnfold.QUnfoldPlotter import histogram_plot, errorbar_plot, ratio_plot


def plot_comparison(method2sol, method2cov, truth, measured, binning, xlabel=None):
    fig = plt.figure(figsize=(9, 7))
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[3, 1], hspace=0)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    truth = truth[1:-1]
    measured = measured[1:-1]
    binning = binning[1:-1]
    histogram_plot(ax=ax1, x=binning, y=truth, label="Truth")
    histogram_plot(ax=ax1, x=binning, y=measured, label="Measured")

    num_points = len(method2sol)
    x_shifts = np.diff(binning) / (num_points + 7)
    unfolding_methods = method2sol.keys()
    for i, method in enumerate(unfolding_methods):
        xbin = binning[:-1] + (i + 4) * x_shifts
        sol = method2sol[method][1:-1]
        cov = method2cov[method][1:-1, 1:-1]
        err = np.sqrt(np.diag(cov))
        chi2 = compute_chi2(sol, truth, cov)
        errorbar_plot(
            ax=ax1,
            x=xbin,
            y=sol,
            yerr=err,
            binning=binning,
            method=method,
            chi2=chi2,
        )
        ratio_sol = sol / truth
        ratio_err = err / truth
        xmid = binning[:-1] + (np.diff(binning) / 2)
        ratio_plot(
            ax=ax2,
            x=xmid,
            y=ratio_sol,
            yerr=ratio_err,
            method=method,
            binning=binning,
            xlabel=xlabel,
        )
    return fig
