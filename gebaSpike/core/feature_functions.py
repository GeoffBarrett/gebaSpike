import numpy as np


def feature_Energy(data):
    """
    Normlized energy calculation discussed from:
    Quantitative measures of cluster quality for use in extracellular recordings by Redish et al.
    Args:
        data: ndarray representing spike data (num_channels X num_spikes X samples_per_spike)
        iPC (optional): the number
        norm (optional): normalize the waveform (True), or not (False).
    Returns:
        E:
    """
    # energy sqrt of the sum of the squares of each point of the waveform, divided by number of samples in waveform
    # energy and first principal component coefficient
    nSamp = data.shape[-1]

    sum_squared = np.sum(data ** 2, axis=-1)

    if len(np.where(sum_squared < 0)[0]) > 0:
        raise ValueError('summing squared values produced negative number!')

    E = np.divide(np.sqrt(sum_squared), np.sqrt(nSamp))  # shape: channel numbers x spike number
    # E[np.where(E == 0)] = 1  # remove any 0's so we avoid dividing by zero

    return E.T


def feature_WavePCX(data, iPC=1, norm=True):
    """Creates the principal components for the waveforms
    Args:
        data: ndarray representing spike data (num_channels X num_spikes X samples_per_spike)
        iPC (optional): the number
        norm (optional): normalize the waveform (True), or not (False).
    Returns:
        FD:
    """

    nCh, nSpikes, nSamp = data.shape

    wavePCData = np.zeros((nSpikes, nCh))

    if norm:
        l2norms = np.sqrt(np.sum(data ** 2, axis=-1))
        l2norms = l2norms.reshape((nCh, -1, 1))

        data = np.divide(data, l2norms)

        # removing NaNs from potential division by zero
        data[np.where(np.isnan(data) == True)] = 0

    for i, w in enumerate(data):
        av = np.mean(w, axis=0)  # row wise average
        # covariance matrix
        cv = np.cov(w.T)
        sd = np.sqrt(np.diag(cv)).T  # row wise standard deviation
        pc, _, _, _ = wavePCA(cv)

        # standardize data to zero mean and unit variance
        wstd = (w - av) / (sd)

        # project the data onto principal component axes
        wpc = np.dot(wstd, pc)

        wavePCData[:, i] = wpc[:, iPC - 1]

    return wavePCData


def wavePCA(cv):
    """
    Principal Component Analysis of standardized waveforms from a
    given (unstandardized) waveform covariance matrix cv(nSamp,nSamp).
    Args:
        cv: nSamp x nSamp wavefrom covariance matrix (unnormalized)
    Returns:
        pc: column oriented principal components (Eigenvectors)
        rpc: column oriented Eigenvectors weighted with their relative amplitudes
        ev: eigenvalues of SVD (= std deviation of data projected onto pc)
        rev: relative eigenvalues so that their sum = 1
    """
    sd = np.sqrt(np.diag(cv)).reshape((-1, 1))  # row wise standard deviation

    # standardized covariance matrix
    cvn = np.divide(cv, np.multiply(sd, sd.T))

    u, ev, pc = np.linalg.svd(cvn)

    # the pc is transposed in the matlab version
    pc = pc.T

    ev = ev.reshape((-1, 1))

    rev = ev / np.sum(ev)  # relative eigne values so that their sum = 1

    rpc = np.multiply(pc, rev.T)

    return pc, rpc, ev, rev


def feature_Amplitude(data):
    amplitude = np.amax(data, axis=2) - np.amin(data, axis=2)

    return amplitude.T


def feature_Peak(data):
    peak = np.amax(data, axis=2)

    return peak.T


def feature_Trough(data):
    trough = np.amin(data, axis=2)

    return trough.T


def feature_TroughTime(data):
    troughT = np.argmin(data, axis=2)

    return troughT.T


def feature_PeakTime(data):
    peakT = np.argmax(data, axis=2)

    return peakT.T


def feature_voltageTime(data, time):
    voltageTime = data[:, :, time]

    return voltageTime.T


def CreateFeatures(data, featuresToCalculate=['Energy', 'WavePCX!1']):
    """Creates the features to be analyzed
    Args:
        featuresToCalculate:
    Returns:
        FD:
    """

    FD = np.array([])

    for feature_name in featuresToCalculate:
        if 'WavePCX' in feature_name:
            # instead of creating a function for every PCX you want to use, I just
            # created one. Use WavePCX!1 for PC1, WavePCX!2 for PC2, etc.

            # all the feature functions are named 'feature_featureName'
            # WavePCX is special though since you need a PC number so separate that
            feature_name, pc_number = feature_name.split('!')
            variables = 'data, %s' % pc_number
        elif 'voltageTime' in feature_name:
            # here we will use voltageTime!index_number
            feature_name, time = feature_name.split('!')
            variables = 'data, %s' % time
        else:
            variables = 'data'

        fnc_name = 'feature_%s' % feature_name
        current_FD = eval("%s(%s)" % (fnc_name, variables))

        if len(FD) == 0:
            FD = current_FD
        else:
            FD = np.hstack((FD, current_FD))

        current_FD = None

    return FD
