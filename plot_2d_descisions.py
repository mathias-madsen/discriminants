import numpy as np
from matplotlib import pyplot as plt

from methods.lda import linear_discriminant
from methods.logistic import LogisticOptimizer, sigmoid


def safe_range(numbers, margin=0.05):
    vmin = np.min(numbers)
    vmax = np.max(numbers)
    vrange = vmax - vmin
    lower = vmin - margin * vrange
    upper = vmax + margin * vrange
    return lower, upper


if __name__ == "__main__":

    n1, n0 = np.random.randint(20, 200, size=2)
    mx1, mx0 = np.random.uniform(-3, +3, size=2)
    x1 = np.random.normal(size=(n1, 2), loc=(mx1, 0.))
    x0 = np.random.normal(size=(n0, 2), loc=(mx0, 0.))

    x = np.concatenate([x1, x0])
    y = np.array([+1]*n1 + [-1]*n0)
    opt = LogisticOptimizer(x, y)
    slope, intercept = opt.newton_solve()
    # slope, intercept = linear_discriminant(x0, x1)
    param_vector = np.concatenate([slope, [intercept]])
    acc = opt.accuracy(param_vector)
    geoprob = opt.geometric_mean_prob(param_vector)

    xmin, xmax = safe_range(x[:, 0])
    ymin, ymax = safe_range(x[:, 1])

    xspan = np.linspace(xmin, xmax, 300)
    yspan = np.linspace(ymin, ymax, 300)
    xyspan = np.stack(np.meshgrid(xspan, yspan), axis=2)
    probspos = sigmoid(xyspan @ slope + intercept)

    plt.imshow(probspos, extent=[xmin, xmax, ymin, ymax], vmin=0, vmax=1)
    plt.plot(*x0.T, ".", color="blue", label="$x_0$")
    plt.plot(*x1.T, ".", color="yellow", label="$x_1$")
    plt.legend()
    plt.title("Accuracy = %.3f; geom. mean prob. = %.3f" % (acc, geoprob))
    plt.tight_layout()
    plt.show()
