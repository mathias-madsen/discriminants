import numpy as np
import torch
from time import perf_counter
from tqdm import tqdm

from methods.lda import linear_discriminant


DEVICE = "mps"


def sigmoid(t):
    return 1 / (1 + torch.exp(-t))


def log_sigmoid(t):
    return -torch.log(1 + torch.exp(-t))


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


def newton_solve(x0, x1, min_iter=5, max_iter=100, min_loss=1e-5, ftol=1e-5, equal_prior=False, verbose=False):

    with torch.no_grad():

        slope, intercept = linear_discriminant(x0, x1, equal_prior=equal_prior)
        w = np.concatenate([slope, [intercept]])
        assert not np.any(np.isnan(w))
        w = torch.tensor(w, dtype=torch.float32, device=DEVICE)
        assert not torch.any(torch.isnan(w))
        old_loss = np.inf

        x, y = make_xy(x0, x1)
        z = y[:, None] * add_ones(x)
        length, dim = z.shape

        assert len(x) == len(y) == len(z), (x.shape, y.shape, z.shape)
        assert len(x.shape) == len(z.shape) == 2, (x.shape, z.shape)
        assert len(y.shape) == 1, (y.shape,)
        assert x.device == y.device == z.device, (x.device, y.device, z.device)
        assert x.dtype == y.dtype == z.dtype, (x.dtype, y.dtype, z.dtype)
        assert x.shape[1] == z.shape[1] - 1

        for iteration in range(max_iter):

            f = torch.mean(softplus(-z @ w))
            new_loss = float(f.cpu().detach().numpy())
            assert not np.isnan(new_loss)
            assert not np.isinf(new_loss)
            drop = old_loss - new_loss
            if verbose:
                print(f"Iteration {iteration}: loss = {new_loss}")

            if iteration > min_iter and drop < ftol:
                if verbose:
                    print("Loss drop at iteration %s: %.2g" % (iteration, drop))
                break
            if new_loss < min_loss:
                if verbose:
                    print("Loss at iteration %s: %.2g" % (iteration, new_loss))
                break

            prob_wrong = sigmoid(-z @ w)
            assert not torch.allclose(prob_wrong, torch.tensor(0.0))
            assert not torch.allclose(prob_wrong, torch.tensor(0.0))

            df = -torch.mean(prob_wrong[:, None] * z, axis=0)
            assert not torch.any(torch.isnan(df))
            assert not torch.any(torch.isinf(df))

            log_pq = log_sigmoid(-z @ w) + log_sigmoid(z @ w)
            pq = torch.exp(log_pq)
            # zzT = z[:, :, None] * z[:, None, :]
            # ddf = torch.mean(pq[:, None, None] * zzT, axis=0)
            ddf = torch.eye(dim, device=z.device)
            # epsilon = (df ** 2).mean() ** 0.5
            epsilon = 0.5  # 1e-1
            arange = epsilon + torch.arange(0, 5, device=z.device)
            for num_seen, pq_t, z_t in zip(arange, pq, z):
                ddf *= num_seen / (num_seen + 1)
                # ddf += 1 / (num_seen + 1) * pq_t * torch.outer(z_t, z_t)
                ddf += 1 / (num_seen + 1) * pq_t * (z_t[:, None] * z_t[None, :])
            assert not torch.any(torch.isnan(ddf))
            assert not torch.any(torch.isinf(ddf))
            assert ddf.shape == (dim, dim), (f, df.shape, ddf.shape)
            # assert not torch.allclose(ddf, torch.tensor(0.0))
            # import ipdb; ipdb.set_trace()
            left, sing, _ = torch.linalg.svd(ddf)
            inv_ddf = left @ torch.diag(1 / sing) @ left.T
            inv_ddf = inv_ddf.to(z.device)
            w -= ((df @ left) / (0.1 + sing)) @ left.T
            # inv_ddf = torch.linalg.inv(ddf)
            # import ipdb; ipdb.set_trace()
            # w -= df @ inv_ddf
            assert not torch.any(torch.isnan(w))

            old_loss = new_loss

        slope = w[:-1].cpu().detach().numpy()
        intercept = w[-1].cpu().detach().numpy()

    if equal_prior:
        intercept -= np.log(len(x1)) - np.log(len(x0))

    return slope, intercept


if __name__ == "__main__":

    dim = 300
    n1 = n0 = 1000
    mu1, mu2 = np.random.normal(size=(2, dim)) / np.sqrt(dim)

    x1 = np.random.normal(size=(n1, dim), loc=mu1)
    x0 = np.random.normal(size=(n0, dim), loc=mu2)

    start = perf_counter()
    slope, intercept = newton_solve(x0, x1)
    dur = perf_counter() - start
    print("Newton-logistic solved in %.3fs\n" % (dur,))

    test_x1 = np.random.normal(size=(n1, dim), loc=mu1)
    test_x0 = np.random.normal(size=(n0, dim), loc=mu2)

    good0 = x0 @ slope + intercept < 0
    good1 = x1 @ slope + intercept >= 0
    accuracy = np.mean(np.concatenate([good0, good1]))
    print("Training accuracy: %.5f" % accuracy)

    good0 = test_x0 @ slope + intercept < 0
    good1 = test_x1 @ slope + intercept >= 0
    accuracy = np.mean(np.concatenate([good0, good1]))
    print("Training accuracy: %.5f" % accuracy)
    print("Test accuracy: %.5f" % accuracy)

