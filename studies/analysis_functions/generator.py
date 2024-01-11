# Main modules
import ROOT as r
import numpy as np
from array import array

# Settings
r.gROOT.SetBatch(True)


def smear(xt, bias, smearing, eff=1):
    """
    Applies a Gaussian smearing effect to a given input value, xt.

    Args:
        xt (float): The input value to apply the smearing to.
        eff (float): smearing efficiency.

    Returns:
        float or None: The resulting value after applying the smearing. Returns None if the value is filtered based on efficiency.
    """
    x = r.gRandom.Rndm()
    if x > eff:
        return None
    xsmear = r.gRandom.Gaus(bias, smearing)
    return xt + xsmear


def generate_standard(
    truth, measured, response, type, distr, samples, bias, smearing, eff
):
    """
    Generate data for standard distributions.

    Args:
        truth (ROOT.TH1F): truth histogram.
        measured (ROOT.TH1F): measured histogram.
        response (ROOT.TH2F): response matrix.
        type (str): type of data generation (data or response).
        distr (distr): the distribution to be generated.
        samples (int): number of samples to be generated.
        bias (float): bias of the distortion.
        smearing (float): smearing of the distortion.
        eff (float): reconstruction efficiency.

    Returns:
        ROOT.TH1F: the filled truth histogram.
        ROOT.TH1F: the filled measured histogram.
        ROOT.TH2F: the filled response matrix.
    """

    # Data generation
    if type == "data":
        r.gRandom.SetSeed(12345)
        for i in range(samples):
            xt = 0
            if distr == "breit-wigner":
                xt = r.gRandom.BreitWigner(5.3, 2.1)
            elif distr == "normal":
                xt = r.gRandom.Gaus(5.3, 1.4)
            elif distr == "exponential":
                xt = r.gRandom.Exp(2.5)
            elif distr == "gamma":
                xt = np.random.gamma(5, 1)
            truth.Fill(xt)
            x = smear(xt, bias, smearing, eff)
            if x != None:
                measured.Fill(x)

        return truth, measured

    # Response generation
    elif type == "response":
        r.gRandom.SetSeed(556)
        for i in range(samples):
            xt = 0
            if distr == "breit-wigner":
                xt = r.gRandom.BreitWigner(5.3, 2.1)
            elif distr == "normal":
                xt = r.gRandom.Gaus(5.3, 1.4)
            elif distr == "exponential":
                xt = r.gRandom.Exp(2.5)
            elif distr == "gamma":
                xt = np.random.gamma(5, 1)
            x = smear(xt, bias, smearing, eff)
            if x != None:
                response.Fill(x, xt)
            else:
                response.Miss(xt)

        return response


def generate_double_peaked(
    truth, measured, response, type, samples, bias, smearing, eff
):
    """
    Generate data for the double peaked distributions.

    Args:
        truth (ROOT.TH1F): truth histogram.
        measured (ROOT.TH1F): measured histogram.
        response (ROOT.TH2F): response matrix.
        type (str): type of data generation (data or response).
        samples (int): number of samples to be generated.
        bias (float): bias of the distortion.
        smearing (float): smearing of the distortion.
        eff (float): reconstruction efficiency.

    Returns:
        ROOT.TH1F: the filled truth histogram.
        ROOT.TH1F: the filled measured histogram.
        ROOT.TH2F: the filled response matrix.
    """

    # Data generation
    if type == "data":
        r.gRandom.SetSeed(12345)
        for i in range(samples):
            xt = r.gRandom.Gaus(3.3, 0.9)
            truth.Fill(xt)
            x = smear(xt, bias, smearing, eff)
            if x != None:
                measured.Fill(x)
        for i in range(samples):
            xt = r.gRandom.Gaus(6.4, 1.2)
            truth.Fill(xt)
            x = smear(xt, bias, smearing, eff)
            if x != None:
                measured.Fill(x)

        return truth, measured

    # Response generation
    elif type == "response":
        r.gRandom.SetSeed(556)
        for i in range(samples):
            xt = r.gRandom.Gaus(3.3, 0.9)
            x = smear(xt, bias, smearing, eff)
            if x != None:
                response.Fill(x, xt)
            else:
                response.Miss(xt)
        for i in range(samples):
            xt = r.gRandom.Gaus(6.4, 1.2)
            x = smear(xt, bias, smearing, eff)
            if x != None:
                response.Fill(x, xt)
            else:
                response.Miss(xt)

        return response


def generate(distr, binning, samples, bias, smearing, eff):
    """
    Generate simulated data and response for a given distribution.

    Args:
        distr (str): The name of the distribution to generate data from.
        bins (int): The number of bins in the histograms.
        min_bin (float): The minimum value of the histogram range.
        max_bin (float): The maximum value of the histogram range.
        samples (int): The number of data samples to generate.
        bias (float): Bias introduced in the measured distribution.
        smearing (float): Smearing introduced in the measured distribution.
        eff (float): Reconstruction efficiency for the measured distribution.

    Returns:
        ROOT.TH1F: The histogram representing the truth distribution.
        ROOT.TH1F: The histogram representing the measured distribution.
        ROOT.RooUnfoldResponse: The response object used for unfolding.
    """

    # Initialize variables
    bins = len(binning) - 1
    truth = r.TH1F(
        "Truth",
        "",
        bins,
        array("d", binning),
    )
    measured = r.TH1F(
        "Measured",
        "",
        bins,
        array("d", binning),
    )
    response = r.RooUnfoldResponse(bins, binning[0], binning[-1])

    # Fill histograms
    if any(d in distr for d in ["normal", "breit-wigner", "exponential", "gamma"]):
        truth, measured = generate_standard(
            truth, measured, response, "data", distr, samples, bias, smearing, eff
        )
        response = generate_standard(
            truth, measured, response, "response", distr, samples, bias, smearing, eff
        )
    elif any(d in distr for d in ["double-peaked"]):
        truth, measured = generate_double_peaked(
            truth, measured, response, "data", samples, bias, smearing, eff
        )
        response = generate_double_peaked(
            truth, measured, response, "response", samples, bias, smearing, eff
        )

    return truth, measured, response
