""" calculate the effective sample size"""
import numpy as np


def batch_means_ess(x):
    """ Estimate the effective sample size as the ratio of the variance
    of the batch means to the variance of the chain. As explained here:
    https://arxiv.org/pdf/1011.0175.pdf. We expect the chain in the format
    Time-Steps, Num-Chains, Dimension (T, M, D) """

    T, M, D = x.shape
    num_batches = int(np.floor(T**(1/3)))
    batch_size = int(np.floor(num_batches**2))
    batch_means = []
    for i in range(num_batches):
        batch = x[batch_size*i:batch_size*i + batch_size]
        batch_means.append(np.mean(batch, axis=0))
    batch_variance = np.var(np.array(batch_means), axis=0)
    chain_variance = np.var(x, axis=0)

    act = batch_size * batch_variance/(chain_variance + 1e-20)

    return 1/act


def acceptance_rate(z):
    cnt = z.shape[0] * z.shape[1]
    for i in range(0, z.shape[0]):
        for j in range(0, z.shape[1]):
            if np.all(np.equal(z[i-1, j], z[i, j])):
                cnt -= 1
    return cnt / float(z.shape[0] * z.shape[1])


def acceptance_rate_2(z):
    slice_1 = z[1:]
    slice_2 = z[:-1]
    return np.mean(slice_1 != slice_2)


def gelman_rubin_diagnostic(x, mu=None):
    m, n = x.shape[0], x.shape[1]
    theta = np.mean(x, axis=1)
    sigma = np.var(x, axis=1)
    # theta_m = np.mean(theta, axis=0)
    theta_m = mu if mu is not None else np.mean(theta, axis=0)
    b = float(n) / float(m-1) * np.sum((theta - theta_m) ** 2)
    w = 1. / (float(m) * np.sum(sigma, axis=0) + 1e-5)
    v = float(n-1) / float(n) * w + float(m+1) / float(m * n) * b
    r_hat = np.sqrt(v / w)
    return r_hat


def autocor_ESS(A):
    A = A * (A > 0.05)
    return 1. / (1. + 2 * np.sum(A[1:]))


def autocovariance(X, tau=0):
  dT, dN, dX = np.shape(X)
  s = 0.
  for t in range(dT - tau):
    x1 = X[t, :, :]
    x2 = X[t+tau, :, :]

    s += np.sum(x1 * x2) / dN

  return s / (dT - tau)


def acl_spectrum(X, scale):
    n = X.shape[0]
    return np.array([autocovariance(X / scale, tau=t) for t in range(n-1)])