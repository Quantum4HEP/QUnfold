import numpy as np
from analysis_functions.custom_logger import get_custom_logger
from analysis_functions.generator import generate
from analysis_functions.RooUnfold import run_RooUnfold
from analysis_functions.QUnfolder import run_QUnfold
from analysis_functions.comparisons import plot_comparisons
from QUnfold.root2numpy import TH1_to_numpy, TH2_to_numpy
from QUnfold.utils import normalize_response, lambda_optimizer


log = get_custom_logger(__name__)

samples = 100000
bins = 20
min_bin = 0.0
max_bin = 10.0
binning = np.linspace(min_bin, max_bin, bins + 1)

distributions = ["normal", "gamma", "exponential", "breit-wigner", "double-peaked"]
bias = -0.13
smearing = 0.21
eff = 0.7

num_reads = 300
num_toys = None
enable_hybrid = False
enable_quantum = False


if __name__ == "__main__":
    for distr in distributions:
        log.info(f"Unfolding the {distr} distribution")
        th1_truth, th1_measured, roounfold_response = generate(
            distr=distr,
            binning=binning,
            samples=samples,
            bias=bias,
            smearing=smearing,
            eff=eff,
        )

        ######################### RooUnfold #########################
        roounfold_response.UseOverflow(True)

        sol_IBU, err_IBU, cov_IBU = run_RooUnfold(
            method="IBU",
            response=roounfold_response,
            measured=th1_measured,
            num_toys=num_toys,
        )

        sol_MI, err_MI, cov_MI = run_RooUnfold(
            method="MI",
            response=roounfold_response,
            measured=th1_measured,
            num_toys=num_toys,
        )

        ########################## QUnfold ##########################
        truth = TH1_to_numpy(th1_truth, overflow=True)
        measured = TH1_to_numpy(th1_measured, overflow=True)
        response = normalize_response(
            response=TH2_to_numpy(roounfold_response.Hresponse(), overflow=True),
            truth_mc=TH1_to_numpy(roounfold_response.Htruth(), overflow=True),
        )
        lam = lambda_optimizer(
            response=response, measured=measured, truth=truth, num_reps=20
        )

        sol_SA, err_SA, cov_SA = run_QUnfold(
            method="SA",
            response=response,
            measured=measured,
            lam=lam,
            num_reads=num_reads,
            num_toys=num_toys,
        )

        if enable_hybrid:
            sol_HYB, err_HYB, cov_HYB = run_QUnfold(
                method="HYB",
                response=response,
                measured=measured,
                lam=lam,
                num_toys=num_toys,
            )

        if enable_quantum:
            sol_QA, err_QA, cov_QA = run_QUnfold(
                method="QA",
                response=response,
                measured=measured,
                lam=lam,
                num_reads=num_reads,
                num_toys=num_toys,
            )

        ######################### Comparison #########################
        solution = {"MI": sol_MI, "IBU": sol_IBU, "SA": sol_SA}
        error = {"MI": err_MI, "IBU": err_IBU, "SA": err_SA}
        covariance = {"MI": cov_MI, "IBU": cov_IBU, "SA": cov_SA}

        if enable_hybrid:
            solution.update({"HYB": sol_HYB})
            error.update({"HYB": err_HYB})
            covariance.update({"HYB": cov_HYB})
        if enable_quantum:
            solution.update({"QA": sol_QA})
            error.update({"QA": err_QA})
            covariance.update({"QA": cov_QA})

        plot_comparisons(solution, error, covariance, distr, truth, measured, binning)
