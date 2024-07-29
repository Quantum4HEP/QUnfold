import os
import numpy as np
import pylab as plt
from QUnfold.utils import compute_chi2_toys
from QUnfold.QUnfoldPlotter import histogram_plot, errorbar_plot, ratio_plot


def plot_comparisons(
    method2sol, method2err, method2cov, truth, measured, binning, distr
):
    fig = plt.figure(figsize=(9, 7))
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[3, 1], hspace=0)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    truth = truth[1:-1]
    measured = measured[1:-1]
    histogram_plot(ax=ax1, x=binning, y=truth, label="Truth")
    histogram_plot(ax=ax1, x=binning, y=measured, label="Measured")

    num_points = len(method2sol)
    x_shifts = np.diff(binning) / (num_points + 7)
    unfolding_methods = method2sol.keys()
    for i, method in enumerate(unfolding_methods):
        x = binning[:-1] + (i + 4) * x_shifts
        sol = method2sol[method][1:-1]
        err = method2err[method][1:-1]
        cov = method2cov[method][1:-1, 1:-1]
        chi2 = compute_chi2_toys(sol, truth, cov)
        errorbar_plot(
            ax=ax1,
            x=x,
            y=sol,
            yerr=err,
            method=method,
            chi2=chi2,
            binning=binning,
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
        )

        for ext in ["pdf"]:
            dirpath = f"studies/img/analysis/{ext}"
            if not os.path.exists(dirpath):
                os.makedirs(dirpath)
            fig.tight_layout()
            fig.savefig(f"{dirpath}/{distr}.{ext}")
        plt.close()
