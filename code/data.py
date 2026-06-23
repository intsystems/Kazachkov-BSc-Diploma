import numpy as np
import torch


# --- True density for wrapped Gaussian shift model (from reference paper) ---
def dens_gauss_shift(X, Y, shift, std, shift_prob=0.5):
    """True joint density p(x,y) for the wrapped Gaussian shift model on [0,1)."""
    o = np.ceil(4 * std)
    a = (Y - X).reshape((*X.shape, 1))
    z = np.arange(-o, o + 1, dtype=float).reshape((*[1 for _ in X.shape], -1))
    d0 = 1 / ((2 * np.pi) ** 0.5 * std) * np.sum(np.exp(-(a - z) ** 2 / (2 * std ** 2)), axis=-1)
    d1 = 1 / ((2 * np.pi) ** 0.5 * std) * np.sum(np.exp(-(a - shift - z) ** 2 / (2 * std ** 2)), axis=-1)
    return shift_prob * d1 + (1 - shift_prob) * d0

def wrapped_normal(mean, sigma, size=None):
    """
    1. Сэмплируем из нормального распределения N(mean, sigma)
    2. Переводим в [0,1)
    size == output shape
    """
    z = np.random.normal(loc=mean, scale=sigma, size=size)
    return np.mod(z, 1.0)

def sample_dataset_torus(M, sigma):
    """
    Генерация M пар (x_i, y_i) из распределения выше
    """
    x = np.random.rand(M) # U[0,1]
    coin = (np.random.rand(M) < 0.5)
    y = np.empty(M)

    idx1 = np.where(coin)[0]
    if idx1.size > 0:
        y[idx1] = wrapped_normal(x[idx1], sigma, size=idx1.size)

    idx2 = np.where(~coin)[0]
    if idx2.size > 0:
        y[idx2] = wrapped_normal(x[idx2] + 0.3, sigma, size=idx2.size)

    return x, y

def sample_batch_with_perm(M, sigma, return_perm=False):
    """
    Генерирует один батч из M точек на торе с неизвестным соответствием:

    1. Сначала генерируем истинные пары (x_i, y_i).
    2. Затем переставляем y по случайной перестановке perm.
    """
    x, y_true = sample_dataset_torus(M, sigma)
    perm = np.random.permutation(M)
    y_permuted = y_true[perm]

    if return_perm:
        # Важно: возвращаем именно perm (индексы y_true, попавшие в позиции y_permuted[k]).
        # Для соответствия m -> k нужно использовать inv_perm = np.argsort(perm).
        return x, y_permuted, perm
    else:
        return x, y_permuted

def sample_batch_tensor(batch_size: int, M: int, sigma: float, device: torch.device):
    xs, ys, perms = [], [], []
    for _ in range(batch_size):
        x, y_perm, perm = sample_batch_with_perm(M, sigma, return_perm=True)
        xs.append(x)
        ys.append(y_perm)
        perms.append(perm)
    x_t = torch.tensor(np.stack(xs), device=device, dtype=torch.float32)
    y_t = torch.tensor(np.stack(ys), device=device, dtype=torch.float32)
    return x_t, y_t, perms
