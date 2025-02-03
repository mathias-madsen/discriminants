import numpy as np
import torch
from time import perf_counter
from tqdm import tqdm

from methods.lda import linear_discriminant
from methods.mlp import BinaryClassificationDataset


DEVICE = "mps"


def sigmoid(t):
    return 1 / (1 + torch.exp(-t))


def add_ones(vectors):
    ones = torch.ones_like(vectors[..., :1])
    return torch.concat([vectors, ones], axis=-1)


def make_xy(x0, x1):
    x = np.concatenate([x0, x1])
    y = np.array([-1]*len(x0) + [+1]*len(x1))
    x = torch.tensor(x, dtype=torch.float32, device=DEVICE)
    y = torch.tensor(y, dtype=torch.float32, device=DEVICE)
    return x, y


softplus = torch.nn.Softplus()


def newton_solve(x0, x1, max_iter=100, min_loss=1e-5, ftol=1e-5):

    slope, intercept = linear_discriminant(x0, x1)
    w = np.concatenate([slope, [intercept]])
    w = torch.tensor(w, dtype=torch.float32, device=DEVICE)
    old_loss = np.inf

    x, y = make_xy(x0, x1)
    z = y[:, None] * add_ones(x)

    for iteration in tqdm(range(max_iter), leave=False):

        f = torch.mean(softplus(-z @ w))
        new_loss = float(f.cpu().detach().numpy())
        drop = old_loss - new_loss
        if drop < ftol:
            # print("Loss drop at iteration %s: %.2g" % (iteration, drop))
            break
        if new_loss < min_loss:
            # print("Loss at iteration %s: %.2g" % (iteration, new_loss))
            break

        prob_wrong = sigmoid(-z @ w)
        prob_correct = 1 - prob_wrong

        df = -torch.mean(prob_wrong[:, None] * z, axis=0)

        ssm = prob_correct[:, None, None] * prob_wrong[:, None, None]
        zzT = z[:, :, None] * z[:, None, :]
        ddf = torch.mean(ssm * zzT, axis=0)

        w -= df @ torch.linalg.inv(ddf)
        old_loss = new_loss

    return w


if __name__ == "__main__":

    dim = 300
    n1 = n0 = 1000
    mu1, mu2 = np.random.normal(size=(2, dim)) / np.sqrt(dim)

    x1 = np.random.normal(size=(n1, dim), loc=mu1)
    x0 = np.random.normal(size=(n0, dim), loc=mu2)

    start = perf_counter()
    w = newton_solve(x0, x1)
    dur = perf_counter() - start
    print("Newton-logistic solved in %.3fs\n" % (dur,))

    test_x1 = np.random.normal(size=(n1, dim), loc=mu1)
    test_x0 = np.random.normal(size=(n0, dim), loc=mu2)

    x, y = make_xy(x0, x1)
    yhats = torch.sign(add_ones(x) @ w)
    is_corrects = torch.isclose(yhats, y)
    accuracy = torch.mean(is_corrects.to(dtype=torch.float32))
    print("Training accuracy: %.5f" % accuracy)

    test_x, test_y = make_xy(test_x0, test_x1)
    yhats = torch.sign(add_ones(test_x) @ w)
    is_corrects = torch.isclose(yhats, test_y)
    accuracy = torch.mean(is_corrects.to(dtype=torch.float32))
    print("Test accuracy: %.5f" % accuracy)

