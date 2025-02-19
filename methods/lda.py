import numpy as np


def safe_stats(vectors):
    length, dim = vectors.shape
    mean = np.mean(vectors, axis=0)
    cov = np.cov(vectors.T, ddof=0)
    w = dim / (10*length + dim)
    mean = (1 - w)*mean
    cov = w*np.eye(dim) + (1 - w)*cov + w*(1 - w)*np.outer(mean, mean)
    return mean, cov



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

    n0, dim0 = x0.shape
    n1, dim1 = x1.shape
    assert dim0 == dim1
    mu0, cov0 = safe_stats(x0)
    mu1, cov1 = safe_stats(x1)
    cov = n0/(n0 + n1)*cov0 + n1/(n0 + n1)*cov1
    mahalanobis0 = mu0 @ np.linalg.solve(cov + 1e-8*np.eye(dim0), mu0)
    mahalanobis1 = mu1 @ np.linalg.solve(cov + 1e-8*np.eye(dim0), mu1)

    slope = np.linalg.solve(cov, mu1 - mu0)
    intercept = 0.5 * (mahalanobis0 - mahalanobis1)

    if not equal_prior:
        intercept += np.log(n1) - np.log(n0)

    return slope, intercept


def compute_logodds_coeffs(mean0, cov0, mean1, cov1):
    """ Compute the coeffficients of a Gaussian log-odds polynomial.
    
    Returns `a2` (matrix), `a1` (vector), `a0` (scalar).

    Use `a0 + x @ a1 + np.sum(x * (x @ a2.T), axis=-1)` to compute the
    log-odds in favor of hypothesis 1 over hypothesis 0 for each of the
    observations in the vector stack x.
    """
    Q0, Q1 = np.linalg.inv([cov0, cov1])
    _, logdetq0 = np.linalg.slogdet(Q0)
    _, logdetq1 = np.linalg.slogdet(Q1)
    mQm0 = mean0 @ Q0 @ mean0
    mQm1 = mean1 @ Q1 @ mean1
    a0 = 0.5 * (logdetq1 - logdetq0) + 0.5 * (mQm0 - mQm1)
    a1 = 1.0 * (mean1 @ Q1 - mean0 @ Q0)
    a2 = 0.5 * (Q0 - Q1)
    return a2, a1, a0


def quadratic_discriminant(x0, x1, equal_prior=False):

    n0, dim0 = x0.shape
    n1, dim1 = x1.shape
    assert dim0 == dim1
    mu0, cov0 = safe_stats(x0)
    mu1, cov1 = safe_stats(x1)
    curvature, slope, intercept = compute_logodds_coeffs(mu0, cov0, mu1, cov1)

    if not equal_prior:
        intercept += np.log(n1) - np.log(n0)

    return curvature, slope, intercept


if __name__ == "__main__":

    import numpy as np
    from scipy.stats import wishart, multivariate_normal

    dim = 3

    C0, C1 = wishart.rvs(dim, np.eye(dim), size=2)
    m0, m1 = np.random.normal(size=(2, dim))
    x = np.random.normal(size=(7, dim))

    logp1 = multivariate_normal.logpdf(x, m1, C1)
    logp0 = multivariate_normal.logpdf(x, m0, C0)
    logodds = logp1 - logp0
    print(logodds)

    Q0, Q1 = np.linalg.inv([C0, C1])
    _, logdetq0 = np.linalg.slogdet(Q0)
    _, logdetq1 = np.linalg.slogdet(Q1)
    mQm0 = m0 @ Q0 @ m0
    mQm1 = m1 @ Q1 @ m1
    a0 = 0.5 * (logdetq1 - logdetq0) + 0.5 * (mQm0 - mQm1)
    a1 = 1.0 * (m1 @ Q1 - m0 @ Q0)
    a2 = 0.5 * (Q0 - Q1)

    logodds = a0 + x @ a1 + np.sum(x * (x @ a2.T), axis=-1)
    print(logodds)

    # # ----------

    # import numpy as np

    # dim = 3
    # A = np.random.normal(size=(dim, dim))
    # u = np.random.normal(size=(1000, dim))
    # u /= np.linalg.norm(u, axis=-1, keepdims=True)
    # Ainv = np.linalg.inv(A)
    # v = u @ Ainv.T
    # # v is now on the ellipse v.T @ A.T @ A @ v

    # # norms = np.linalg.norm(v @ A.T, axis=-1)
    # # print(norms)
    # # for vt in v:
    # #     print(vt @ A.T @ A @ vt)

    # B = np.random.normal(size=(dim, dim))
    # values, vectors = np.linalg.eig(Ainv @ B @ B.T @ Ainv.T)
    # idx = np.argmax(values)
    # print(vectors[:, idx])

    # jdx = np.argmax(np.sum(v * (v @ B.T @ B), axis=1))
    # flip = np.sign(vectors[0, idx]) * np.sign(v[jdx, 0])
    # print(flip * v[jdx] / np.linalg.norm(v[jdx]))
