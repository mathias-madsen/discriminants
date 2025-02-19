import numpy as np

from methods.lda import linear_discriminant


def softplus(t, threshold=80):
    return np.where(t <= threshold, np.log(1 + np.exp(t)), t)

def sigmoid(t):
    clipt = np.clip(t, -80, +80)
    return 1 / (1 + np.exp(-clipt))

def log_sigmoid(t):
    return -softplus(-t)

def add_ones(vectors):
    ones = np.ones_like(vectors[..., :1])
    return np.concatenate([vectors, ones], axis=-1)


class LogisticOptimizer:

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.z = y[:, None] * add_ones(x)
        _, self.dim = self.z.shape

    def mean_neg_log_prob(self, w):
        """ Mean negative log-probability of the correct class label. """
        return -np.mean(log_sigmoid(self.z @ w))

    def geometric_mean_prob(self, w):
        """ Geometric mean probability of the correct class label. """
        return np.exp(-self.mean_neg_log_prob(w))

    def accuracy(self, w):
        """ Fraction of correctly classified data points. """
        logodds = add_ones(self.x) @ w
        yhats = np.sign(logodds)
        return np.isclose(yhats, self.y).mean()

    def compute_derivatives(self, w):
        """ Compute loss, loss gradient, and loss Jacobian. """
        dots = self.z @ w
        prob_correct = sigmoid(dots)  # assigned to actual state
        prob_wrong = sigmoid(-dots)  # assigned to counterfactual state
        # loss value: negative probability of what actually happened:
        f = -np.sum(log_sigmoid(dots))
        # first derivative (gradient vector)
        df = -np.sum(prob_wrong[:, None] * self.z, axis=0)
        # second derivative (Hessian matrix)
        ddf = 1e-5 * np.eye(self.dim)
        for p, q, z in zip(prob_correct, prob_wrong, self.z):
            ddf += p * q * np.outer(z, z)
        # ssm = prob_correct[:, None, None] * prob_wrong[:, None, None]
        # zzT = self.z[:, :, None] * self.z[:, None, :]
        # ddf = np.sum(ssm * zzT, axis=0)
        return f, df, ddf

    def newton_update(self, w, epsilon=1e-8):
        """ Compute current loss and updated parameter. """
        f, df, ddf = self.compute_derivatives(w)
        # mixing the Hessian with a small proportion of the identity matrix
        # causes its Eigenvalues to shift slightly towards 1., which in turn
        # causes the Newton step lengths to be biased towards a plain gradient
        # step instead of a second-derivative modulated gradient step.
        safe_hessian = (1 - epsilon)*ddf + epsilon*np.eye(self.dim)
        dw = np.linalg.solve(safe_hessian, df)
        return f, w - dw

    def newton_solve(self, min_iter=5, max_iter=30, ftol=1e-5, min_loss=1e-5, verbose=False):
        """ Find good parameters using Newton's method. """
        x0 = self.x[self.y == -1]
        x1 = self.x[self.y == +1]
        assert len(x0) + len(x1) == len(self.x)
        slope, intercept = linear_discriminant(x0, x1)
        params = np.concatenate([slope, [intercept]])
        old_loss = np.inf
        for iteration in range(max_iter):
            new_loss, params = self.newton_update(params)
            if verbose:
                print("\rIterion %s, loss: %.5f    " % (iteration, new_loss), end="")
            if new_loss < min_loss:
                break  # stop because no more improvement is necessary
            if iteration > min_iter and old_loss - new_loss < ftol:
                break  # stop because we're not improving much anymore
            old_loss = new_loss
        if verbose:
            print("")
        slope = params[:-1]
        intercept = params[-1]
        return slope, intercept


def newton_solve(x0, x1, equal_prior=False, verbose=False):
    x = np.concatenate([x0, x1], axis=0)
    if len(x.shape) == 1:
        x = x[:, None]
    y = np.array([-1]*len(x0) + [+1]*len(x1))
    assert len(x) == len(y)
    opt = LogisticOptimizer(x, y)
    slope, intercept = opt.newton_solve(verbose=verbose)
    if equal_prior:
        intercept -= np.log(len(x1)) - np.log(len(x0))
    return slope, intercept


if __name__ == "__main__":

    from matplotlib import pyplot as plt
    from scipy.stats import norm

    mu0, mu1 = -2.0, +2.0

    x0, x1 = np.random.normal(size=(2, 25))
    x0 = (x0 - x0.mean()) / x0.std() + mu0
    x1 = (x1 - x1.mean()) / x1.std() + mu1
    assert np.isclose(x0.mean(), mu0)
    assert np.isclose(x1.mean(), mu1)
    assert np.isclose(x0.var(), 1.0)
    assert np.isclose(x1.var(), 1.0)

    x = np.concatenate([x0, x1])[:, None]
    y = np.array([-1]*len(x0) + [+1]*len(x0))
    opt = LogisticOptimizer(x, y)
    slope, intercept = opt.newton_solve()

    span = np.linspace(-5, +5, 1000)
    
    print(" " * 8 + " ".join("(%.2f, 0)" % x for x in x0))
    print(" " * 8 + " ".join("(%.2f, 1)" % x for x in x1))
    print()
    print("{ 1 / (1 + exp(-(%.5f*x))) };" % (mu1 - mu0,))
    print("{ 1 / (1 + exp(-(%.5f*x + %.5f))) };" % (slope.item(), intercept.item()))
    print()

    plt.figure(figsize=(12, 5))
    plt.plot(x0, np.zeros_like(x0), "o")
    plt.plot(x1, np.ones_like(x1), "o")
    plt.plot(span, sigmoid(span * (mu1 - mu0)), alpha=0.2)
    plt.plot(span, sigmoid(span * slope + intercept), alpha=0.2)
    plt.show()


    x0 = np.random.normal(size=(20, 2), loc=(-1, 0))
    x1 = np.random.normal(size=(20, 2), loc=(+1, 0))
    x = np.concatenate([x0, x1], axis=0)
    y = np.array([-1]*len(x0) + [+1]*len(x0))
    opt = LogisticOptimizer(x, y)

    log_slope, log_intercept = opt.newton_solve()
    logisticx = lambda y: (log_intercept - log_slope[1]*y) / log_slope[0]

    lin_slope, lin_intercept = linear_discriminant(x0, x1)
    fisherx = lambda y: (lin_intercept - lin_slope[1]*y) / lin_slope[0]

    for segment in np.split(x0, 4, axis=0):
        print(" " * 8 + " ".join("(%.3f, %.3f)" % tuple(x) for x in segment))
    print()

    left, sing, right = np.linalg.svd(np.cov(x0.T, ddof=0))
    m1, m2 = np.mean(x0, axis=0)
    v1, w1, v2, w2 = np.ravel(left * np.sqrt(sing))
    text = "    \\draw[blue, line width=1] \\pgfextra{\n"
    text += "        \\pgfpathellipse{\\pgfplotspointaxisxy{%.3f}{%.3f}}\n" % (m1, m2)
    text += "        {\\pgfplotspointaxisdirectionxy{%.3f}{%.3f}}\n" % (v1, v2)
    text += "        {\\pgfplotspointaxisdirectionxy{%.3f}{%.3f}}\n" % (w1, w2)
    text += "    };\n"
    print(text)

    for segment in np.split(x1, 4, axis=0):
        print(" " * 8 + " ".join("(%.3f, %.3f)" % tuple(x) for x in segment))
    print()

    left, sing, right = np.linalg.svd(np.cov(x1.T, ddof=0))
    m1, m2 = np.mean(x1, axis=0)
    v1, w1, v2, w2 = (left * np.sqrt(sing)).flatten()
    text = "    \\draw[red, line width=1] \\pgfextra{\n"
    text += "        \\pgfpathellipse{\\pgfplotspointaxisxy{%.3f}{%.3f}}\n" % (m1, m2)
    text += "        {\\pgfplotspointaxisdirectionxy{%.3f}{%.3f}}\n" % (v1, v2)
    text += "        {\\pgfplotspointaxisdirectionxy{%.3f}{%.3f}}\n" % (w1, w2)
    text += "    };\n"
    print(text)

    points = (logisticx(-3), -3, logisticx(+3), 3)
    print(" " * 8 + "(%.3f, %.3f) (%.3f, %.3f)\n" % points)

    points = (fisherx(-3), -3, fisherx(+3), 3)
    print(" " * 8 + "(%.3f, %.3f) (%.3f, %.3f)\n" % points)

    plt.figure(figsize=(8, 6))
    plt.plot(*x0.T, "ro", alpha=0.5)
    plt.plot(*x1.T, "bo", alpha=0.5)
    plt.plot([logisticx(-3), logisticx(+3)], [-3, +3], color="orange")
    plt.plot([fisherx(-3), fisherx(+3)], [-3, +3], color="purple")
    plt.xlim(-4, +4)
    plt.ylim(-3, +3)
    plt.show()
