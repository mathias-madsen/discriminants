import numpy as np

from methods.lda import linear_discriminant


def sigmoid(t):
    return 1 / (1 + np.exp(-t))


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
        return -np.mean(np.log(sigmoid(self.z @ w)))

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
        prob_correct = sigmoid(self.z @ w)  # assigned to actual state
        prob_wrong = 1 - prob_correct  # assigned to counterfactual state
        # loss value: negative probability of what actually happened:
        f = -np.sum(np.log(prob_correct))
        # first derivative (gradient vector)
        df = -np.sum(prob_wrong[:, None] * self.z, axis=0)
        # second derivative (Hessian matrix)
        ssm = prob_correct[:, None, None] * prob_wrong[:, None, None]
        zzT = self.z[:, :, None] * self.z[:, None, :]
        ddf = np.sum(ssm * zzT, axis=0)
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

    def newton_solve(self, max_iter=30, ftol=1e-5, min_loss=1e-5):
        """ Find good parameters using Newton's method. """
        x0 = self.x[self.y == -1]
        x1 = self.x[self.y == +1]
        assert len(x0) + len(x1) == len(self.x)
        slope, intercept = linear_discriminant(x0, x1)
        params = np.concatenate([slope, [intercept]])
        old_loss = np.inf
        for _ in range(max_iter):
            new_loss, params = self.newton_update(params)
            if new_loss < min_loss:
                break  # stop because no more improvement is necessary
            if old_loss - new_loss < ftol:
                break  # stop because we're not improving much anymore
            old_loss = new_loss
        slope = params[:-1]
        intercept = params[-1]
        return slope, intercept
