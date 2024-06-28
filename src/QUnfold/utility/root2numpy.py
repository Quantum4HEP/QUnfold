import numpy as np


def TH1_to_numpy(histo, error=False, overflow=False, dtype=np.float64):
    """
    Convert a ROOT.TH1 object into a 1D numpy array.

    Args:
        histo (ROOT.TH1): the input TH1 histogram to convert.
        error (bool): if True, it gets the bin error instead of the bin content.
        overflow (bool): if True, it considers also the overflow bins.
        dtype (type): the data type of the output array.

    Returns:
        array (numpy.ndarray): the output 1D numpy array.
    """
    get_entry = histo.GetBinError if error else histo.GetBinContent
    if overflow:
        irange = range(0, histo.GetNbinsX() + 2)
    else:
        irange = range(1, histo.GetNbinsX() + 1)
    array = np.array([get_entry(i) for i in irange])
    return array.astype(dtype=dtype)


def TH2_to_numpy(histo, overflow=False, dtype=np.float64):
    """
    Convert a ROOT.TH2 object into a 2D numpy array.

    Args:
        histo (ROOT.TH2): the input TH2 histogram to convert.
        overflow (bool): if True, it considers also the overflow bins.
        dtype (type): the data type of the output array.

    Returns:
        array (numpy.ndarray): the output 2D numpy array.

    """
    if overflow:
        irange = range(0, histo.GetNbinsX() + 2)
        jrange = range(0, histo.GetNbinsY() + 2)
    else:
        irange = range(1, histo.GetNbinsX() + 1)
        jrange = range(1, histo.GetNbinsY() + 1)
    array = np.array([[histo.GetBinContent(i, j) for j in jrange] for i in irange])
    return array.astype(dtype=dtype)


def TVector_to_numpy(vector, dtype=np.float64):
    """
    Convert a ROOT.TVector object into a 1D numpy array.

    Args:
        vector (ROOT.TVector): the input TVector to convert.
        dtype (type): the data type of the output array.

    Returns:
        array (numpy.ndarray): the output 1D numpy array.
    """
    num_elements = vector.GetNoElements()
    array = np.array([vector[i] for i in range(num_elements)])
    return array.astype(dtype=dtype)


def TMatrix_to_numpy(matrix, dtype=np.float64):
    """
    Convert a ROOT.TMatrix object into a 2D numpy array.

    Parameters:
        matrix (ROOT.TMatrix): the input TMatrix to convert.
        dtype (type): the data type of the output array.

    Returns:
        array (numpy.ndarray): the output 2D numpy array.
    """
    num_rows = matrix.GetNrows()
    num_cols = matrix.GetNcols()
    array = np.array([[matrix[i][j] for j in range(num_cols)] for i in range(num_rows)])
    return array.astype(dtype=dtype)
