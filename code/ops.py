import torch
import torch.nn.functional as F

from implicit_sinkhorn import Sinkhorn


def circular_residual(y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
    """
    y, y_hat in R (не обязательно в [0,1)).
    Возвращает разность на торе в диапазоне [-0.5, 0.5).
    """
    return torch.remainder(y - y_hat + 0.5, 1.0) - 0.5


def gumbel_distribution_sample(shape: torch.Size, device=None, eps=1e-20) -> torch.Tensor:
    if device is None:
        device = torch.device("cpu")
    u = torch.rand(shape, device=device)
    return -torch.log(-torch.log(u + eps) + eps)


def log_sinkhorn(log_alpha: torch.Tensor, n_iters: int = 20) -> torch.Tensor:
    """
    Stabilized Sinkhorn in log-space.
    log_alpha: [..., M, M]
    Returns: log of approximately doubly-stochastic matrix.
    """
    for _ in range(n_iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)  # row norm
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)  # col norm
    return log_alpha


def sinkhorn_permutation(
    logits: torch.Tensor,
    temperature: float,
    n_iters: int = 20,
    backend: str = "implicit",
    gumbel: bool = False,
) -> torch.Tensor:
    """
    Soft permutation z from assignment logits, with switchable backend.

    logits: [..., M, M]. With `gumbel=True` adds Gumbel(0,1) noise before
    Sinkhorn (Gumbel-Sinkhorn relaxation used during training); with
    `gumbel=False` it is the deterministic projection used at inference.

    backend="unrolled": z = exp(log_sinkhorn((logits[+gumbel])/tau))
        (autograd through all iterations; the original implementation).
    backend="implicit": same forward via the implicit-Sinkhorn module
        (c = -(logits[+gumbel]), lambd_sink = tau, uniform marginals a=b=1),
        but the gradient is computed at the Sinkhorn fixed point.

    Both backends return an (approximately) doubly-stochastic [..., M, M]
    matrix with unit marginals (total mass M), so downstream loss/metrics
    are unchanged.
    """
    if gumbel:
        logits = logits + gumbel_distribution_sample(logits.shape, device=logits.device)

    if backend == "unrolled":
        return torch.exp(log_sinkhorn(logits / temperature, n_iters=n_iters))

    if backend == "implicit":
        c = -logits
        a = torch.ones(logits.shape[:-1], device=logits.device, dtype=logits.dtype)  # [..., M]
        b = torch.ones(logits.shape[:-2] + logits.shape[-1:], device=logits.device, dtype=logits.dtype)  # [..., M]
        return Sinkhorn.apply(c, a, b, n_iters, float(temperature))

    raise ValueError(f"unknown sinkhorn_backend: {backend!r} (expected 'implicit' or 'unrolled')")

# === mixture reconstruction ===
def pairwise_gaussian_mixture_nll_batched(
    y: torch.Tensor,
    component_means: torch.Tensor,
    component_logits: torch.Tensor,
    sigma2: torch.Tensor,
) -> torch.Tensor:
    """Batched version: y [B, M], component_means [B, M, K], component_logits [B, M, K] -> [B, M, M]."""
    # y: [B, M] -> [B, 1, M, 1] (expand over m and K)
    # means: [B, M, K] -> [B, M, 1, K] (expand over k)
    y_expanded = y.unsqueeze(1).unsqueeze(-1)          # [B, 1, M, 1]
    means_expanded = component_means.unsqueeze(2)      # [B, M, 1, K]
    delta = circular_residual(y_expanded, means_expanded)  # [B, M, M, K]
    component_log_prob = -0.5 * (torch.log(2.0 * torch.pi * sigma2) + (delta ** 2) / sigma2)
    mixture_log_prob = F.log_softmax(component_logits, dim=-1).unsqueeze(2)  # [B, M, 1, K]
    pair_log_prob = torch.logsumexp(mixture_log_prob + component_log_prob, dim=-1)  # [B, M, M]
    return -pair_log_prob
