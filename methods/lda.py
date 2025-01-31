import numpy as np


def linear_discriminant(x0, x1, equal_prior=False):
    """ Compute the slope and intercept of Fisher's linear discriminant.
    
    This returns the slope and intercept for an affine function that
    maps a new observation x onto a posterior log-odds ratio in favor
    of hypothesis 1 over hypothesis 0. The slope is a D-vector, and
    the intecept is a scalar.

    The assumptions behind the model is that the samples from the two
    clusters both come from multivariate Gaussians, and that they share
    the same covariance matrix, but have different means.
    """

    n0, dim0 = x1.shape
    n1, dim1 = x1.shape
    assert dim0 == dim1
    mu0 = np.mean(x0, axis=0)
    mu1 = np.mean(x1, axis=0)
    cov0 = np.cov(x0.T, ddof=0)
    cov1 = np.cov(x1.T, ddof=0)
    cov = n0/(n0 + n1)*cov0 + n1/(n0 + n1)*cov1
    mahalanobis0 = mu0 @ np.linalg.solve(cov, mu0)
    mahalanobis1 = mu1 @ np.linalg.solve(cov, mu1)

    slope = np.linalg.solve(cov, mu1 - mu0)
    intercept = 0.5 * (mahalanobis0 - mahalanobis1)

    if not equal_prior:
        intercept += np.log(n1) - np.log(n0)

    return slope, intercept
