import os
import numpy as np
from QUnfold import QUnfoldQUBO, QUnfoldPlotter


def QUnfold_unfolder_and_plot(
    unf_type, response, measured, truth, distr, bins, min_bin, max_bin
):
    """
    TODO: docstring
    """

    # Create dirs
    if not os.path.exists("studies/img/QUnfold/{}".format(distr)):
        os.makedirs("studies/img/QUnfold/{}".format(distr))

    # Unfolder
    unfolder = QUnfoldQUBO(response, measured, lam=0.05)
    unfolder.initialize_qubo_model()
    unfolded = None
    error = None

    # Unfold with simulated annealing
    if unf_type == "SA":
        unfolded, error = unfolder.solve_simulated_annealing(num_reads=10)
        plotter = QUnfoldPlotter(
            response=response,
            measured=measured,
            truth=truth,
            unfolded=unfolded,
            error=error,
            binning=np.linspace(min_bin, max_bin, bins + 1),
        )
        plotter.savePlot("studies/img/QUnfold/{}/unfolded_SA.png".format(distr), "SA")
        plotter.saveResponse("studies/img/QUnfold/{}/response.png".format(distr))
        print(
            "The png file studies/img/QUnfold/{}/unfolded_SA.png has been created".format(
                distr
            )
        )

    # Unfold with hybrid solver
    elif unf_type == "HYB":
        unfolded = unfolder.solve_hybrid_sampler()
        plotter = QUnfoldPlotter(
            response=response,
            measured=measured,
            truth=truth,
            unfolded=unfolded,
            error=error,
            binning=np.linspace(min_bin, max_bin, bins + 1),
        )
        plotter.savePlot("studies/img/QUnfold/{}/unfolded_HYB.png".format(distr), "HYB")
        plotter.saveResponse("studies/img/QUnfold/{}/response.png".format(distr))
        print(
            "The png file studies/img/QUnfold/{}/unfolded_HYB.png has been created".format(
                distr
            )
        )

    return unfolded, error
