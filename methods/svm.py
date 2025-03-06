import numpy as np
from tqdm import tqdm
from scipy.optimize import minimize, LinearConstraint

from methods.lda import linear_discriminant


def neg_y_grad(x, y, alpha):
    """
    An N-vector whose coordinates are the elementwise product of `y`
    and the gradient `g` of the loss function `0.5 @ a @ Q @ a - sum(a)`,
    that is, `Q @ a - 1`.
    """
    yx = y[:, None] * x
    gradient = yx @ (yx.T @ alpha) - np.ones_like(alpha)
    return -y * gradient


def split_indices(y, alpha, C, epsilon=1e-8):
    """ Return two boolean arrays indicating the partitioning. """
    posy = np.isclose(y, +1.0)
    negy = np.isclose(y, -1.0)
    not_C = alpha < C - epsilon
    not_0 = alpha > 0 + epsilon
    should_be_small = not_C * posy + not_0 * negy
    should_be_large = not_C * negy + not_0 * posy
    return should_be_small, should_be_large


def measure_optimality(x, y, alpha, C):
    """ A feasible point is a solution if this number is >= 0.0. """
    # see Lin 2002, section 2
    negygrad = neg_y_grad(x, y, alpha)
    small, large = split_indices(y, alpha, C)
    should_be_small = np.max(negygrad[small,])
    should_be_large = np.min(negygrad[large,])
    return should_be_large - should_be_small


def pick_good_index_set(x, y, alpha, C, num_elms):
    # see Joachims, section 3.2
    negygrad = neg_y_grad(x, y, alpha)
    indices = np.arange(len(negygrad))
    small, large = split_indices(y, alpha, C)
    order = np.argsort(negygrad)
    # sort the coordinate indices into ascending order
    # according to negative y times the gradient vector:
    sorted_indices = indices[order,]
    # make sure the boolean masks are in the same order:
    sorted_small = small[order,]
    sorted_large = large[order,]
    # then extract the two subsets of indices, the one that is
    # supposed to be small, and the one that is supposed to be big:
    minidx = sorted_indices[sorted_small]
    maxidx = sorted_indices[sorted_large * (~sorted_small)]
    # then grab the largest small and smallest big in order
    # to try to provoke a violation:
    maximin = minidx[-num_elms//2:]
    minimax = maxidx[:num_elms//2]
    while np.any(np.isin(maximin, minimax)):
        maximin = maximin[1:,]
        minimax = minimax[:-1,]
    return np.concatenate([maximin, minimax])


def solve_subproblem(x, y, C, alpha, idx):

    assert np.issubdtype(idx.dtype, np.integer)

    N, _ = x.shape
    jdx = np.array([j for j in range(N) if j not in idx], dtype=int)

    yx = y[:, None] * x

    alphaB = alpha[jdx,]
    yA = y[idx,]
    yB = y[jdx,]
    yxA = yx[idx, :]
    yxB = yx[jdx, :]

    zAzBaB = yxA @ (yxB.T @ alphaB)

    def loss(alphaA):
        return (
            + 0.5 * (alphaA @ yxA) @ (yxA.T @ alphaA)
            # + 0.5 * (alphaB @ yxB) @ (yxB.T @ alphaB)
            + alphaA @ zAzBaB
            - np.sum(alphaA)
            # - np.sum(alphaB)
            )

    def dloss(alphaA):
        return (
            + yxA @ (yxA.T @ alphaA)
            + zAzBaB
            - np.ones_like(alphaA)
            )

    # def ddloss(alphaA):
    #     return yxA @ yxA.T

    bounds = np.array([(0, C) for _ in idx])
    # initial = alpha[idx,]
    initial = np.random.uniform(*bounds.T, size=len(idx))
    rhs = -yB @ alphaB
    constraint = LinearConstraint(A=yA, lb=rhs, ub=rhs)

    result = minimize(
        loss,
        jac=dloss,
        # hess=self.hess,
        x0=initial,
        bounds=bounds,
        constraints=constraint,
        # method="trust-constr",
    )

    assert result.success

    # assert np.all(result.x >= 0)
    # assert np.all(result.x <= C)
    # assert np.isclose(result.x @ yA + alphaB @ yB, 0.0)
    # assert np.isclose(result.x @ yA, rhs)

    solution = alpha.copy()
    solution[idx] = result.x
    
    # assert np.all(solution >= 0)
    # assert np.all(solution <= C)
    # assert np.allclose(solution[idx], result.x)
    # assert np.allclose(solution[jdx], alphaB)
    # assert np.isclose(solution[idx,] @ yA + solution[jdx,] @ yB, 0.0)
    # assert np.isclose(solution @ y, 0.0)

    return solution


def objective_function(x, y, alpha):
    yx = y[:, None] * x
    return 0.5 * (alpha @ yx) @ (yx.T @ alpha) - np.sum(alpha)


def solve_svm(x0, x1, verbose=True):

    x = np.concatenate([x0, x1], axis=0)
    y = np.array([-1.0]*len(x0) + [+1.0]*len(x1))

    guess = np.zeros_like(y)

    losses = []
    optimalities = []

    for iteration in range(1000):

        # selection = np.random.choice(len(x), size=2, replace=False)
        selection = pick_good_index_set(x, y, guess, 1.0, 100)

        loss = objective_function(x, y, guess)
        optimality = measure_optimality(x, y, guess, C=1.0)
        losses.append(loss)
        optimalities.append(optimality)

        if optimality > -1e-3:
            if verbose:
                print("Achieved optimality %.5f" % optimality)
            break

        lossdrops = -np.diff(losses)
        if len(lossdrops) >= 5 and np.max(lossdrops[-5:]) < 1e-5:
            if verbose:
                print("Step %s loss drop: %.2g" % (iteration, lossdrops[-1]))
            break

        guess = solve_subproblem(x, y, 1.0, guess, selection)

        if verbose:
            n_svs = np.sum(~np.isclose(guess, 0.0) * ~np.isclose(guess, 1.0))
            n_errs = np.sum(np.isclose(guess, 1.0))
            msg = (
                "%s support vectors, %s errors, loss=%.5f, optimality=%.5f" %
                (n_svs, n_errs, loss, optimality)
                )
            print("\r%s" % msg, end=" " * 10)

    if verbose:
        print("")

    return guess


def compute_slope_and_intercept_from_multipliers(x, y, alpha):
    yx = y[:, None] * x
    idx, = np.where(~np.isclose(alpha, 0.0, atol=1e-8))
    # slope = np.sum(yx[idx,] * alpha[idx, None], axis=0)
    slope = np.sum(yx * alpha[:, None], axis=0)
    # intercept = np.mean(y - x @ slope, axis=0)
    intercept = np.mean(y[idx,] - x[idx,] @ slope, axis=0)
    return slope, intercept


if __name__ == "__main__":

    from matplotlib import pyplot as plt

    from methods.visualization import points_on_line

    dim = 2
    N1 = N2 = 500
    m0, m1 = np.random.normal(size=(2, dim), scale=3.0)
    x0 = np.random.normal(size=(N1, dim), loc=m0)
    x1 = np.random.normal(size=(N2, dim), loc=m1)
    x = np.concatenate([x0, x1], axis=0)
    y = np.array([-1.0]*len(x0) + [+1.0]*len(x1))

    alpha = solve_svm(x0, x1)
    slope, intercept = compute_slope_and_intercept_from_multipliers(x, y, alpha)

    pts_lo = points_on_line(slope, intercept - 1, width=15.0)
    pts_mid = points_on_line(slope, intercept, width=15.0)
    pts_hi = points_on_line(slope, intercept + 1, width=15.0)

    num = len(alpha)
    num_errs = np.sum(np.isclose(alpha, 1.0))
    num_svs = np.sum(~np.isclose(alpha, 0.0) * ~np.isclose(alpha, 1.0))

    figure, (left, right) = plt.subplots(figsize=(12, 8), ncols=2)
    left.plot(*x0.T, "b.", alpha=0.8)
    left.plot(*x1.T, "r.", alpha=0.8)
    mappable = right.scatter(*x.T, s=9, c=alpha)
    right.plot(*pts_lo.T, lw=3, alpha=0.5, color="purple", ls="dotted")
    right.plot(*pts_mid.T, lw=3, alpha=0.5, color="purple")
    right.plot(*pts_hi.T, lw=3, alpha=0.5, color="purple", ls="dotted")
    right.set_title(
        "%s/%s support vectors, %s/%s errors" %
        (num_svs, num, num_errs, num)
        )
    plt.colorbar(mappable)
    plt.tight_layout()
    plt.show()