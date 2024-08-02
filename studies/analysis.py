import os
import ROOT
import numpy as np
from QUnfold.root2numpy import TH1_to_numpy, TH2_to_numpy
from QUnfold.utils import normalize_response, lambda_optimizer
from generator import generate
from unfolder import run_RooUnfold, run_QUnfold
from comparison import plot_comparison


roounfold_lib_path = "./HEP_deps/RooUnfold/libRooUnfold.so"
roounfold_error = ROOT.gSystem.Load(roounfold_lib_path)
if roounfold_error:
    raise ImportError("RooUnfold was not loaded successfully!")


samples = 100000
bins = 20
xrange = np.linspace(start=0.0, stop=10.0, num=bins + 1).tolist()
binning = np.array([-np.inf] + xrange + [np.inf])

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
        print(f"Unfolding '{distr}' distribution...")

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

        sol_MI, cov_MI = run_RooUnfold(
            method="MI",
            response=roounfold_response,
            measured=th1_measured,
            num_toys=num_toys,
        )

        sol_IBU, cov_IBU = run_RooUnfold(
            method="IBU",
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
            response=response,
            measured=measured,
            binning=binning,
            truth=truth,
            num_reps=20,
        )

        sol_SA, cov_SA = run_QUnfold(
            method="SA",
            response=response,
            measured=measured,
            binning=binning,
            lam=lam,
            num_reads=num_reads,
            num_toys=num_toys,
        )

        if enable_hybrid:
            sol_HYB, cov_HYB = run_QUnfold(
                method="HYB",
                response=response,
                measured=measured,
                binning=binning,
                lam=lam,
                num_toys=num_toys,
            )

        if enable_quantum:
            sol_QA, cov_QA = run_QUnfold(
                method="QA",
                response=response,
                measured=measured,
                binning=binning,
                lam=lam,
                num_reads=num_reads,
                num_toys=num_toys,
            )

        ######################### Comparison #########################
        solution = {"MI": sol_MI, "IBU": sol_IBU, "SA": sol_SA}
        covariance = {"MI": cov_MI, "IBU": cov_IBU, "SA": cov_SA}

        if enable_hybrid:
            solution.update({"HYB": sol_HYB})
            covariance.update({"HYB": cov_HYB})
        if enable_quantum:
            solution.update({"QA": sol_QA})
            covariance.update({"QA": cov_QA})

        fig = plot_comparison(
            solution, covariance, truth=truth, measured=measured, binning=binning
        )

        for ext in ["png", "pdf"]:
            dirpath = f"studies/img/analysis/{ext}"
            if not os.path.exists(dirpath):
                os.makedirs(dirpath)
            fig.tight_layout()
            fig.savefig(f"{dirpath}/{distr}.{ext}")
