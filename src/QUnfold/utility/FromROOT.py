import numpy as np


def TH1_to_array(histo, overflow=False):
    """
    Convert a ROOT.TH1F into a numpy.array.

    Args:
        histo (ROOT.TH1F): the input ROOT.TH1F to be converted.
        overflow (bool): enable disable first and last bins overflow.

    Returns:
        numpy.array: a numpy.array of the histo bin contents.
    """

    if overflow:
        start, stop = 0, histo.GetNbinsX() + 2
    else:
        start, stop = 1, histo.GetNbinsX() + 1
    return np.array([histo.GetBinContent(i) for i in range(start, stop)])


def TH2_to_array(histo, overflow=False):
    """
    Convert a ROOT.TH2F object into a numpy.array.

    Parameters:
        hist (ROOT.TH2F): The TH2F object to convert.
        overflow (bool): enable disable first and last bins overflow.

    Returns:
        numpy_array (numpy.array): The numpy.array representing the contents of the TH2F.

    """

    if overflow:
        x_start, x_stop = 0, histo.GetNbinsX() + 2
        y_start, y_stop = 0, histo.GetNbinsY() + 2
    else:
        x_start, x_stop = 1, histo.GetNbinsX() + 1
        y_start, y_stop = 1, histo.GetNbinsY() + 1

    return np.array(
        [
            [histo.GetBinContent(i, j) for j in range(y_start, y_stop)]
            for i in range(x_start, x_stop)
        ]
    )


def TVector_to_array(vector):
    """
    Convert a ROOT.TVectorD object into a numpy.array.

    Parameters:
        vector (ROOT.TVectorD): input TVectorD object to be converted.

    Returns:
        numpy.array: array representing the content of the TVectorD.
    """
    return np.array([vector[i] for i in range(vector.GetNoElements())])


def TMatrix_to_array(matrix):
    """
    Convert a ROOT.TMatrixD object into a numpy.array.

    Parameters:
        matrix (ROOT.TMatrixD): input TMatrixD object to be converted.

    Returns:
        numpy.array: array representing the content of the TMatrixD.
    """
    return np.array(
        [
            [matrix[i][j] for j in range(matrix.GetNcols())]
            for i in range(matrix.GetNrows())
        ]
    )
