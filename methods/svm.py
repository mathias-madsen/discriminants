import numpy as np
from matplotlib import pyplot as plt

from scipy.optimize import minimize, LinearConstraint

from methods.lda import linear_discriminant


def add_ones(vectors):
    ones = np.ones_like(vectors[..., :1])
    return np.concatenate([vectors, ones], axis=-1)


def solve_separable(x0, x1):

    _, dim = np.shape(x0)

    # construct the vector (1 1 ... 1 0):
    ones_zeros = np.ones(dim + 1)
    ones_zeros[-1] *= 0

    x = np.concatenate([x0, x1], axis=0)
    y = np.array([-1.0]*len(x0) + [+1.0]*len(x1))

    slope, intercept = linear_discriminant(x0, x1)
    initial = np.concatenate([slope, np.atleast_1d(intercept)])

    yz = y[:, None] * add_ones(x)
    constraint = LinearConstraint(A=yz, lb=1, ub=np.inf)

    def loss(w):
        return 0.5 * np.sum(w[:-1] ** 2)

    def dloss(w):
        return w * ones_zeros

    def ddloss(w, scalar=1.0):
        return scalar * np.diag(ones_zeros)

    result = minimize(
        fun=loss,
        jac=dloss,
        hess=ddloss,
        hessp=ddloss,
        x0=initial,
        method="trust-constr",
        constraints=constraint,
    )

    return result.x[:-1], result.x[-1]  # slope, intercept


if __name__ == "__main__":

    from methods.logistic import newton_solve

    dim = 2
    n_obs_per_class = 10
    x0 = np.random.normal(size=(n_obs_per_class, dim)) + 2.0
    x1 = np.random.normal(size=(n_obs_per_class, dim)) - 2.0
    x = np.concatenate([x0, x1], axis=0)
    y = np.array([-1.0]*len(x0) + [+1.0]*len(x1))

    a, b = solve_separable(x0, x1)

    vmax = max(abs(x.min()), abs(x.max()))
    span = np.linspace(-vmax, +vmax)

    plt.figure(figsize=(6, 6))

    plt.plot(*x0.T, "bo")
    plt.plot(*x1.T, "ro")

    a1, a2 = a
    plt.plot(span, -a1/a2*span - b/a2, lw=3, alpha=0.5, color="purple")
    plt.plot(span, -a1/a2*span - (b + 1)/a2, lw=3, alpha=0.5, color="purple", ls="dotted")
    plt.plot(span, -a1/a2*span - (b - 1)/a2, lw=3, alpha=0.5, color="purple", ls="dotted")

    (f1, f2), fb = linear_discriminant(x0, x1)
    plt.plot(span, -f1/f2*span - fb/f2, lw=3, alpha=0.5, color="orange")

    (r1, r2), rb = newton_solve(x0, x1)
    plt.plot(span, -r1/r2*span - rb/r2, lw=3, alpha=0.5, color="green")

    plt.xlim(-1.1*vmax, +1.1*vmax)
    plt.ylim(-1.1*vmax, +1.1*vmax)
    plt.tight_layout()
    plt.show()