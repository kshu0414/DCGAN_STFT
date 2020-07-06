import os
import numpy as np


def pdist(sample_1, sample_2, norm=2, eps=1e-5):
    r"""Compute the matrix of all squared pairwise distances.
    Arguments
    ---------
    sample_1 : numpy.array
        The first sample, should be of shape ``(n_1, d)``.
    sample_2 : numpy.array
        The second sample, should be of shape ``(n_2, d)``.
    norm : float
        The l_p norm to be used.
    Returns
    -------
    numpy.array
        Matrix of shape (n_1, n_2). The [i, j]-th entry is equal to
        ``|| sample_1[i, :] - sample_2[j, :] ||_p``."""
    n_1, n_2 = np.shape(sample_1)[0], np.shape(sample_2)[0]
    if np.shape(sample_1)[-1] == 1:
        sample_1 = np.squeeze(sample_1,axis=-1)
    if np.shape(sample_2)[-1] == 1:
        sample_2 = np.squeeze(sample_2,axis=-1)
    norm = float(norm)
    
    if norm == 2.:
        norms_1 = np.sum(sample_1**2, axis=1, keepdims=True)
        norms_2 = np.sum(sample_2**2, axis=1, keepdims=True)
        norms1 = np.tile(norms_1, n_2)
        norms2 = np.transpose(np.tile(norms_2,n_1))
        norms = norms1+norms2
        distances_squared = norms - 2 * sample_1.dot(np.transpose(sample_2))
        return np.sqrt(eps + np.abs(distances_squared))
    else:
        dim = np.shape(sample_1)[1]
        expanded_1 = np.tile(np.expand_dims(sample_1,axis=1),(n_1,n_2,dim))
        expanded_2 = np.tile(np.expand_dims(sample_2,axis=0),(n_1,n_2,dim))
        differences = np.abs(expanded_1 - expanded_2) ** norm
        inner = np.sum(differences, axis=2, keepdims=False)
        return (eps + inner) ** (1. / norm)

def MMD(sample_1,sample_2,alphas):
    r"""Evaluate the statistic.
    The kernel used is
    .. math::
        k(x, x') = \sum_{j=1}^k e^{-\alpha_j \|x - x'\|^2},
    for the provided ``alphas``.
    Arguments
    ---------
    sample_1: :numpy.array
        The first sample, of size ``(n_1, d)``.
    sample_2: numpy.array
        The second sample, of size ``(n_2, d)``.
    alphas : list of :class:`float`
        The kernel parameters.
    ret_matrix: bool
        If set, the call with also return a second variable.
        This variable can be then used to compute a p-value using
        :py:meth:`~.MMDStatistic.pval`.
    Returns
    -------
    :class:`float`
        The test statistic.
    :class:`torch:torch.autograd.Variable`
        Returned only if ``ret_matrix`` was set to true."""
    
    n_1, n_2 = np.shape(sample_1)[0], np.shape(sample_2)[0]
    a00 = 1. / (n_1 * (n_1 - 1))
    a11 = 1. / (n_2 * (n_2 - 1))
    a01 = - 1. / (n_1 * n_2)

    sample_12 = np.concatenate((sample_1, sample_2), axis=0)
    distances = pdist(sample_12, sample_12, norm=2)

    kernels = None
    for alpha in alphas:
        kernels_a = np.exp(- alpha * distances ** 2)
        if kernels is None:
            kernels = kernels_a
        else:
            kernels = kernels + kernels_a

    k_1 = kernels[:n_1, :n_1]
    k_2 = kernels[n_1:, n_1:]
    k_12 = kernels[:n_1, n_1:]

    mmd = (2 * a01 * np.sum(k_12) +
            a00 * (np.sum(k_1) - np.trace(k_1)) +
            a11 * (np.sum(k_2) - np.trace(k_2)))
    return mmd

def pairwisedistances(X,Y,norm=2):
    dist = pdist(X,Y,norm)
    return np.median(dist)

def RMSE(sample_1,sample_2):
    n_1, n_2 = np.shape(sample_1)[0], np.shape(sample_2)[0]
    if not n_1 == n_2:
        raise Exception('Size of samples must be same to compute RMSE')
    norm = np.sum((sample_1-sample_2)**2,axis=1,keepdims=True)
    RMSE = np.sqrt(np.mean(norm,axis=0))
    return RMSE