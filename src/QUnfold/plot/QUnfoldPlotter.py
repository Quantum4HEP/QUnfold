import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chisquare


class QUnfoldPlotter:
    """
    Class used to plot QUnfold data and results.
    """

    def __init__(self, response, measured, truth, unfolded, error, binning):
        """
        Constructs a QUnfoldPlotter class for visualizing unfolding results.

        Args:
            response (np.array): The response matrix used in the unfolding process.
            measured (np.array): The measured (observed) data distribution.
            truth (np.array): The true distribution.
            unfolded (np.array): The unfolded distribution.
            error (np.array): The error on the unfolded distribution.
            binning (np.array): The binning information for the histograms.
        """

        self.response = response
        self.measured = measured
        self.truth = truth
        self.unfolded = unfolded
        self.error = error
        self.binning = binning

    def __plotResponseSetup(self):
        """
        Set up the response matrix plot for drawing or saving: the response matrix is set up as a heatmap, with the column representing the measured variable and the row representing the truth variable.
        """

        # Set up plot
        plt.imshow(
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

        # Plot settings
        plt.colorbar()
        plt.xlabel("Truth")
        plt.ylabel("Measured")

    def plotResponse(self):
        """
        Plot the response matrix with matplotlib style.
        """

        self.__plotResponseSetup()
        plt.show()
        plt.close()

    def saveResponse(self, path: str):
        """
        Save the plotted response matrix with matplotlib style to a file.

        Args:
            path (str): The file path to save the plot.
        """

        self.__plotResponseSetup()
        plt.savefig(path)
        plt.close()

    def _compute_chi2_dof(self, unfolded, truth):
        """
        Compute the chi-squared per degree of freedom (chi2/dof) between two distributions.

        Args:
            unfolded (numpy.array): The unfolded distribution bin contents.
            truth (numpy.array): The truth distribution bin contents.

        Returns:
            float: The chi-squared per degree of freedom.
        """

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

    def __plotSetup(self, method):
        """
        Create an histogram comparison among measured, truth and unfolded distributions, with the chi2 among unfolded and truth distribution.

        Args:
            method (str): Unfolding method type.
        """

        # Divide into subplots
        fig = plt.figure()
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        marker_size = 3.5

        # Plot Truth
        truth_steps = np.append(self.truth, [self.truth[-1]])
        ax1.step(
            self.binning, truth_steps, label="Truth", where="post", color="tab:blue"
        )
        ax1.fill_between(
            self.binning, truth_steps, step="post", alpha=0.3, color="tab:blue"
        )

        # Plot measured
        meas_steps = np.append(self.measured, [self.measured[-1]])
        ax1.step(
            self.binning, meas_steps, label="Measured", where="post", color="tab:orange"
        )

        # Plot unfolded histogram with chi2 test
        binwidths = np.diff(self.binning)
        bin_midpoints = self.binning[:-1] + binwidths / 2
        chi2 = round(self._compute_chi2_dof(self.unfolded, self.truth), 2)
        label = rf"Unfolded {method} ($\chi^2 = {chi2}$)"
        ax1.errorbar(
            x=bin_midpoints,
            y=self.unfolded,
            yerr=self.error,
            label=label,
            marker="o",
            ms=marker_size,
            c="green",
            linestyle="None",
        )

        # Plot ratio QUnfold to truth
        ax2.axhline(y=1, color="tab:blue")
        ax2.errorbar(
            x=bin_midpoints,
            y=self.unfolded / self.truth,
            yerr=self.error / self.truth,
            ms=marker_size,
            fmt="o",
            color="g",
        )

        # Plot style settings
        ax1.tick_params(axis="x", which="both", bottom=True, top=False, direction="in")
        ax2.tick_params(axis="x", which="both", bottom=True, top=True, direction="in")
        ax1.set_xlim(self.binning[0], self.binning[-1])
        ax1.set_ylim(0, ax1.get_ylim()[1])
        ax2.set_yticks([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75])
        ax2.set_yticklabels(["", "0.5", "", "1.0", "", "1.5", ""])
        ax1.tick_params(
            axis="x", which="both", bottom=False, top=False, labelbottom=False
        )

        # Plot settings
        ax2.set_ylabel("Ratio to\ntruth")
        ax2.set_xlabel("Bins")
        ax1.set_ylabel("Entries")
        ax1.legend(loc="upper right")
        plt.tight_layout()

    def plot(self, method=""):
        """
        Plot the measured distribution histogram.

        Args:
            method (str): Unfolding method type. Default "".
        """

        self.__plotSetup(method)
        plt.show()
        plt.close()

    def savePlot(self, path: str, method=""):
        """
        Save the plot of the measured distribution histogram into path.

        Args:
            path (str): The file path to save the plot.
            method (str): Unfolding method type. Default "".
        """

        self.__plotSetup(method)
        plt.savefig(path)
        plt.close()
