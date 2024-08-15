import numpy as np
import pylab as plt
from qunfold import QPlotter
from qunfold.utils import compute_chi2


def plot_comparison(method2sol, method2cov, truth, measured, binning, xlabel):
    fig = plt.figure(figsize=(9, 7))
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[3, 1], hspace=0)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    truth = truth[1:-1]
    measured = measured[1:-1]
    binning = binning[1:-1]
    widths = np.diff(binning)
    QPlotter.histogram_plot(ax=ax1, xedges=binning, hist=truth, label="Truth")
    QPlotter.histogram_plot(ax=ax1, xedges=binning, hist=measured, label="Measured")

    num_points = len(method2sol)
    xshift = widths / (num_points + 9)
    xlims = (binning[0], binning[-1])
    for i, method in enumerate(method2sol):
        xpt = binning[:-1] + (i + 5) * xshift
        sol = method2sol[method][1:-1]
        cov = method2cov[method][1:-1, 1:-1]
        err = np.sqrt(np.diag(cov))
        chi2 = compute_chi2(observed=sol, expected=truth)
        QPlotter.errorbar_plot(ax=ax1, xmid=xpt, hist=sol, err=err, xlims=xlims, label=method, chi2=chi2)
        yticks = [tick for tick in ax1.get_yticks() if tick != 0]
        ax1.set_yticks(yticks)
        sol_ratio = sol / truth
        err_ratio = err / truth
        QPlotter.ratio_plot(
            ax=ax2, xmid=xpt, ratio=sol_ratio, err=err_ratio, label=method, xticks=binning, xlabel=xlabel
        )
    return fig
