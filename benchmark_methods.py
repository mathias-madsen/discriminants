import numpy as np
from time import perf_counter
from scipy.optimize import minimize

from methods.lda import linear_discriminant
from methods.logistic import LogisticOptimizer


if __name__ == "__main__":


    for iteration in range(5):

        print(f"Experiment number {iteration + 1}\n")

        dim = 300
        n1 = n0 = 1000
        mu1, mu2 = np.random.normal(size=(2, dim)) / np.sqrt(dim)
        x1 = np.random.normal(size=(n1, dim), loc=mu1)
        x0 = np.random.normal(size=(n0, dim), loc=mu2)

        start = perf_counter()
        x = np.concatenate([x1, x0])
        y = np.array([+1]*n1 + [-1]*n0)
        opt = LogisticOptimizer(x, y)

        start = perf_counter()
        slope, intercept = linear_discriminant(x0, x1)
        final_params = np.concatenate([slope, [intercept]])
        dur = perf_counter() - start
        print("Fisher linear descriminant solved in %.3fs" % (dur,))
        # print("Accuracy: %.5f" % opt.accuracy(final_params))
        # print("Mean neg log prob: %.5f" % opt.mean_neg_log_prob(final_params))
        # print("Geomean probability: %.5f" % opt.geometric_mean_prob(final_params))
        # print()

        start = perf_counter()
        slope, intercept = opt.newton_solve()
        final_params = np.concatenate([slope, [intercept]])
        dur = perf_counter() - start
        print("Newton-logistic solved in %.3fs" % (dur,))
        # print("Accuracy: %.5f" % opt.accuracy(final_params))
        # print("Mean neg log prob: %.5f" % opt.mean_neg_log_prob(final_params))
        # print("Geomean probability: %.5f" % opt.geometric_mean_prob(final_params))
        # print()

        # start = perf_counter()
        # x0 = np.random.normal(size=(dim + 1))
        # result = minimize(
        #     opt.mean_neg_log_prob,
        #     x0,
        #     jac=lambda w: opt.compute_derivatives(w)[1],
        #     hess=lambda w: opt.compute_derivatives(w)[2],
        #     # method="Newton-CG",  # Newton-CG|dogleg|trust-ncg|trust-krylov|trust-exact|trust-constr
        #     # method="trust-krylov",  # Newton-CG|dogleg|trust-ncg|trust-krylov|trust-exact|trust-constr
        #     method="dogleg",  # Newton-CG|dogleg|trust-ncg|trust-krylov|trust-exact|trust-constr
        #     )
        # slope, intercept = result.x[:-1], result.x[-1]
        # final_params = np.concatenate([slope, [intercept]])
        # dur = perf_counter() - start
        # print("Scipy-Newton-CG in %.3fs" % (dur,))
        # print("Accuracy: %.5f" % opt.accuracy(final_params))
        # print("Mean neg log prob: %.5f" % opt.mean_neg_log_prob(final_params))
        # print("Geomean probability: %.5f" % opt.geometric_mean_prob(final_params))
        # print()

        print()