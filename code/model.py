import math

import torch
import torch.nn as nn

from ops import (
    circular_residual,
    pairwise_gaussian_mixture_nll_batched,
    sinkhorn_permutation,
)


class Encoder(nn.Module):
    """
    Энкодер q_psi(sigma(m) | x_m, y_k): возвращает logits phi_{mk}.
    В этой постановке поддерживается только полный вход без batch-оси.
    """
    def __init__(self, hidden_dim=128):
        super().__init__()
        # вход: [sin(2pi*x), cos(2pi*x), sin(2pi*y), cos(2pi*y)] => 4
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x, y: [M]
        if x.dim() != 1 or y.dim() != 1:
            raise ValueError(f"Encoder expects x and y of shape [M], got x={tuple(x.shape)}, y={tuple(y.shape)}")
        if x.shape != y.shape:
            raise ValueError(f"x and y must have the same shape, got x={tuple(x.shape)}, y={tuple(y.shape)}")

        m = x.shape[0]

        two_pi = 2.0 * torch.pi
        x_feat = torch.stack([torch.sin(two_pi * x), torch.cos(two_pi * x)], dim=-1)  # [M, 2]
        y_feat = torch.stack([torch.sin(two_pi * y), torch.cos(two_pi * y)], dim=-1)  # [M, 2]

        x_exp = x_feat.unsqueeze(1).expand(m, m, 2)  # [M, M, 2]
        y_exp = y_feat.unsqueeze(0).expand(m, m, 2)  # [M, M, 2]

        pairs = torch.cat([x_exp, y_exp], dim=-1)    # [M, M, 4]
        logits = self.net(pairs).squeeze(-1)         # [M, M]
        return logits

    def forward_batched(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Batched version: x, y: [B, M] -> logits [B, M, M]."""
        B, m = x.shape
        two_pi = 2.0 * torch.pi
        x_feat = torch.stack([torch.sin(two_pi * x), torch.cos(two_pi * x)], dim=-1)  # [B, M, 2]
        y_feat = torch.stack([torch.sin(two_pi * y), torch.cos(two_pi * y)], dim=-1)  # [B, M, 2]

        x_exp = x_feat.unsqueeze(2).expand(B, m, m, 2)  # [B, M, M, 2]
        y_exp = y_feat.unsqueeze(1).expand(B, m, m, 2)  # [B, M, M, 2]

        pairs = torch.cat([x_exp, y_exp], dim=-1)       # [B, M, M, 4]
        logits = self.net(pairs).squeeze(-1)             # [B, M, M]
        return logits

class Decoder(nn.Module):
    """
    Conditional decoder p_theta(y | x, z).
    Декодер зависит только от x, а z задает перестановку.
    Поддерживается только полный вход без batch-оси.
    """
    def __init__(self, M: int, hidden_dim=128, sigma2=0.01, num_mixture_components: int = 2,
                 mixture_mean_init: str = "spread"):
        super().__init__()
        self.M = M
        self.sigma2 = sigma2
        self.num_mixture_components = max(1, int(num_mixture_components))
        self.mixture_mean_init = mixture_mean_init

        # === CHANGED: decoder predicts mixture means and logits ===
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * self.num_mixture_components)
        )

        self._init_mixture_head()

    def _init_mixture_head(self):
        """
        Symmetry-breaking init for the mixture head, so the K components start
        spread over the torus instead of nearly identical (см. лекцию по GMM:
        "do not initialize components as nearly the same Gaussians").

        Output layout follows raw_params.view(M, K, 2): на плоском bias [2K]
        индекс 2k+0 — среднее компоненты k, 2k+1 — её logit.
        """
        if self.mixture_mean_init != "spread":
            return

        last = self.net[-1]
        K = self.num_mixture_components
        with torch.no_grad():
            # means spread uniformly over the torus [0,1): k/K
            last.bias[0::2] = torch.arange(K, dtype=last.bias.dtype) / K
            # uniform mixing weights at start
            last.bias[1::2] = 0.0
            # damp the input-dependent contribution so the spread survives at init
            last.weight[0::2] *= 0.1

    def forward(
        self,
        x: torch.Tensor,
        z: torch.Tensor = None,
        permute: bool = True,
        return_mixture_params: bool = False,
    ):
        if x.dim() != 1:
            raise ValueError(f"Decoder expects x of shape [M], got {tuple(x.shape)}")
        if x.shape[0] != self.M:
            raise ValueError(f"x must have M={self.M}, got {tuple(x.shape)}")
        if z is not None:
            if z.dim() != 2:
                raise ValueError(f"Decoder expects z of shape [M, M], got {tuple(z.shape)}")
            if z.shape != (self.M, self.M):
                raise ValueError(f"z must have shape [{self.M}, {self.M}], got {tuple(z.shape)}")

        two_pi = 2.0 * torch.pi
        x_feat = torch.stack([torch.sin(two_pi * x), torch.cos(two_pi * x)], dim=-1)
        raw_params = self.net(x_feat).view(self.M, self.num_mixture_components, 2)
        component_means = raw_params[..., 0]
        component_logits = raw_params[..., 1]

        if return_mixture_params:
            return component_means, component_logits

        if self.num_mixture_components != 1:
            raise ValueError("Use return_mixture_params=True when num_mixture_components > 1")

        y_proto = component_means.squeeze(-1)
        if permute:
            y_proto = torch.matmul(z.transpose(-1, -2), y_proto.unsqueeze(-1)).squeeze(-1)
        return y_proto

    def forward_batched(
        self,
        x: torch.Tensor,
        z: torch.Tensor = None,
        permute: bool = True,
        return_mixture_params: bool = False,
    ):
        """Batched version: x: [B, M], z: [B, M, M] -> [B, M] or ([B, M, K], [B, M, K])."""
        B = x.shape[0]
        two_pi = 2.0 * torch.pi
        x_feat = torch.stack([torch.sin(two_pi * x), torch.cos(two_pi * x)], dim=-1)  # [B, M, 2]
        raw_params = self.net(x_feat).view(B, self.M, self.num_mixture_components, 2)
        component_means = raw_params[..., 0]   # [B, M, K]
        component_logits = raw_params[..., 1]  # [B, M, K]

        if return_mixture_params:
            return component_means, component_logits

        if self.num_mixture_components != 1:
            raise ValueError("Use return_mixture_params=True when num_mixture_components > 1")

        y_proto = component_means.squeeze(-1)  # [B, M]
        if permute:
            y_proto = torch.matmul(z.transpose(-1, -2), y_proto.unsqueeze(-1)).squeeze(-1)
        return y_proto

class CategoricalTorusVAE(nn.Module):
    def __init__(
        self,
        M: int,
        hidden_dim=128,
        sigma2=0.01,
        temperature=1.0,
        sinkhorn_iters: int = 20,
        num_mixture_components: int = 2,
        sinkhorn_backend: str = "implicit",
        mixture_mean_init: str = "spread",
    ):
        super().__init__()
        self.M = M
        self.num_mixture_components = max(1, int(num_mixture_components))
        self.encoder = Encoder(hidden_dim)
        self.decoder = Decoder(
            M,
            hidden_dim,
            sigma2,
            num_mixture_components=self.num_mixture_components,
            mixture_mean_init=mixture_mean_init,
        )
        self.temperature = temperature
        self.sinkhorn_iters = sinkhorn_iters
        self.sinkhorn_backend = sinkhorn_backend

    def infer_z(self, x: torch.Tensor, y: torch.Tensor, temperature=None) -> torch.Tensor:
        """
        Детерминированная оценка z без gumbel-шумов:
        z = Sinkhorn(exp(phi/tau)).
        """
        tau = temperature if temperature is not None else self.temperature
        phi = self.encoder(x, y)
        z = sinkhorn_permutation(
            phi, tau, self.sinkhorn_iters, self.sinkhorn_backend, gumbel=False
        )
        return z

    # ---------- Batched methods ----------

    def infer_z_batched(self, x: torch.Tensor, y: torch.Tensor, temperature=None) -> torch.Tensor:
        """Batched deterministic inference: x,y [B,M] -> z [B,M,M]."""
        tau = temperature if temperature is not None else self.temperature
        phi = self.encoder.forward_batched(x, y)  # [B, M, M]
        return sinkhorn_permutation(
            phi, tau, self.sinkhorn_iters, self.sinkhorn_backend, gumbel=False
        )

    def forward_batched(self, x: torch.Tensor, y: torch.Tensor, temperature=None):
        """Batched forward: x,y [B,M] -> (phi [B,M,M], decoder_output, z [B,M,M])."""
        tau = temperature if temperature is not None else self.temperature
        phi = self.encoder.forward_batched(x, y)  # [B, M, M]
        z = sinkhorn_permutation(
            phi, tau, self.sinkhorn_iters, self.sinkhorn_backend, gumbel=True
        )  # [B, M, M]

        if self.num_mixture_components == 1:
            decoder_output = self.decoder.forward_batched(x, z)
        else:
            decoder_output = self.decoder.forward_batched(x, return_mixture_params=True)
        return phi, decoder_output, z

    def loss_batched(self, x: torch.Tensor, y: torch.Tensor, temperature=None, beta: float = 1.0, return_parts: bool = False):
        """Batched loss: x,y [B,M] -> scalar (mean over batch)."""
        tau = temperature if temperature is not None else self.temperature
        phi, decoder_output, z = self.forward_batched(x, y, temperature)
        sigma2 = torch.as_tensor(self.decoder.sigma2, device=y.device, dtype=y.dtype)

        if self.num_mixture_components == 1:
            y_hat = decoder_output  # [B, M]
            delta = circular_residual(y, y_hat)  # [B, M]
            # sum over M, mean over B
            recon = (0.5 * (torch.log(2.0 * torch.pi * sigma2) + (delta ** 2) / sigma2)).sum(dim=-1).mean()
        else:
            component_means, component_logits = decoder_output  # [B, M, K] each
            pair_nll = pairwise_gaussian_mixture_nll_batched(y, component_means, component_logits, sigma2)  # [B, M, M]
            # sum over (M, M), mean over B
            recon = (z * pair_nll).sum(dim=(-1, -2)).mean()

        # KL regularizes the post-Sinkhorn doubly-stochastic latent z (the
        # distribution the decoder consumes), evaluated deterministically
        # (no Gumbel noise). Its rows are valid categorical posteriors; the
        # Gumbel-Sinkhorn permutation density is intractable, so this per-row
        # factorized form on z_det is used as a tractable surrogate.
        z_det = sinkhorn_permutation(
            phi, tau, self.sinkhorn_iters, self.sinkhorn_backend, gumbel=False
        )  # [B, M, M]
        log_q = torch.log(z_det.clamp_min(1e-12))
        # sum over (M, M), mean over B
        kl = (z_det * (log_q + math.log(self.M))).sum(dim=(-1, -2)).mean() / self.M

        total = recon + beta * kl
        if return_parts:
            return total, recon, kl
        return total
