import ROOT
import numpy as np
from array import array

ROOT.gROOT.SetBatch(True)
ROOT.gRandom.SetSeed(12345)


def smear(xt, bias, smearing, eff=1):
    x = ROOT.gRandom.Rndm()
    if x > eff:
        return None
    xsmear = ROOT.gRandom.Gaus(bias, smearing)
    return xt + xsmear


def generate_standard(truth, measured, response, type, distr, samples, bias, smearing, eff):
    # Data generation
    if type == "data":
        for _ in range(samples):
            xt = 0
            if distr == "breit-wigner":
                xt = ROOT.gRandom.BreitWigner(5.3, 2.1)
            elif distr == "normal":
                xt = ROOT.gRandom.Gaus(5.3, 1.4)
            elif distr == "exponential":
                xt = ROOT.gRandom.Exp(2.5)
            elif distr == "gamma":
                xt = np.random.gamma(5, 1)
            truth.Fill(xt)
            x = smear(xt, bias, smearing, eff)
            if x != None:
                measured.Fill(x)

        return truth, measured

    # Response generation
    elif type == "response":
        for _ in range(samples):
            xt = 0
            if distr == "breit-wigner":
                xt = ROOT.gRandom.BreitWigner(5.3, 2.1)
            elif distr == "normal":
                xt = ROOT.gRandom.Gaus(5.3, 1.4)
            elif distr == "exponential":
                xt = ROOT.gRandom.Exp(2.5)
            elif distr == "gamma":
                xt = np.random.gamma(5, 1)
            x = smear(xt, bias, smearing, eff)
            if x != None:
                response.Fill(x, xt)
            else:
                response.Miss(xt)

        return response


def generate_double_peaked(truth, measured, response, type, samples, bias, smearing, eff):
    # Data generation
    if type == "data":
        for _ in range(samples):
            xt = ROOT.gRandom.Gaus(3.3, 0.9)
            truth.Fill(xt)
            x = smear(xt, bias, smearing, eff)
            if x != None:
                measured.Fill(x)
        for _ in range(samples):
            xt = ROOT.gRandom.Gaus(6.4, 1.2)
            truth.Fill(xt)
            x = smear(xt, bias, smearing, eff)
            if x != None:
                measured.Fill(x)

        return truth, measured

    # Response generation
    elif type == "response":
        for _ in range(samples):
            xt = ROOT.gRandom.Gaus(3.3, 0.9)
            x = smear(xt, bias, smearing, eff)
            if x != None:
                response.Fill(x, xt)
            else:
                response.Miss(xt)
        for _ in range(samples):
            xt = ROOT.gRandom.Gaus(6.4, 1.2)
            x = smear(xt, bias, smearing, eff)
            if x != None:
                response.Fill(x, xt)
            else:
                response.Miss(xt)

        return response


def generate(distr, binning, samples, bias, smearing, eff):
    if binning[0] == -np.inf:
        binning = binning[1:]
    if binning[-1] == np.inf:
        binning = binning[:-1]
    bins = len(binning) - 1
    truth = ROOT.TH1F(f"Truth_{distr}", "", bins, array("d", binning))
    measured = ROOT.TH1F(f"Measured_{distr}", "", bins, array("d", binning))
    response = ROOT.RooUnfoldResponse(bins, binning[0], binning[-1])

    if any(d in distr for d in ["normal", "breit-wigner", "exponential", "gamma"]):
        truth, measured = generate_standard(truth, measured, response, "data", distr, samples, bias, smearing, eff)
        response = generate_standard(truth, measured, response, "response", distr, samples, bias, smearing, eff)
    elif any(d in distr for d in ["double-peaked"]):
        truth, measured = generate_double_peaked(truth, measured, response, "data", samples, bias, smearing, eff)
        response = generate_double_peaked(truth, measured, response, "response", samples, bias, smearing, eff)

    return truth, measured, response
