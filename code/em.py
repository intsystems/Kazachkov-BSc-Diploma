"""EM algorithm for the global shift-mixture model.

Generative model per group (M paired points, then random permutation of y):
    x_m ~ U[0,1), c_m ~ Cat(pi_1..pi_K),
    y_{perm(m)} ~ wrapped_normal(x_m + shift_{c_m}, sigma^2).

Trainable parameters: shifts in R^K and log_pi in R^K. sigma^2 is fixed.

E-step (closed form, no backprop):
    L[m,k] = log Σ_c π_c · N_circ(y_k - x_m - shift_c; σ²)
    q      = exp(log_sinkhorn(L / τ))           # soft permutation [M,M]
    w[m,k,c] = π_c·N_circ(...) / Σ_c'...        # component posterior

M-step (closed form):
    r[m,k,c] = q[m,k] · w[m,k,c]
    S_c    = Σ_{n,m,k} r^(n)[m,k,c]
    π_c    = S_c / Σ_c S_c
    shift_c = (1/(2π)) · atan2(Σ r·sin(2π·(y_k − x_m)), Σ r·cos(2π·(y_k − x_m))) mod 1
"""

import math
from itertools import permutations

import numpy as np
import torch
import torch.nn as nn

from config import device
from data import sample_batch_with_perm
from metrics import compute_accuracy
from ops import circular_residual, log_sinkhorn


def _circular_abs_error(a: float, b: float) -> float:
    return abs(((a - b + 0.5) % 1.0) - 0.5)


def shift_parameter_errors(shifts, pi, true_shifts=(0.0, 0.3)):
    """Align estimated mixture components to true torus shifts and score errors."""
    shifts_np = np.asarray(shifts, dtype=float)
    pi_np = np.asarray(pi, dtype=float)
    true_np = np.asarray(true_shifts, dtype=float)
    if shifts_np.shape[0] != true_np.shape[0]:
        return {
            "shift_mean_abs_error": float("nan"),
            "shift_max_abs_error": float("nan"),
            "pi_l1_error": float("nan"),
        }

    best = None
    true_pi = np.full(true_np.shape[0], 1.0 / true_np.shape[0], dtype=float)
    for perm in permutations(range(shifts_np.shape[0])):
        aligned_shifts = shifts_np[list(perm)]
        errors = np.array([_circular_abs_error(est, true) for est, true in zip(aligned_shifts, true_np)])
        score = float(errors.mean())
        if best is None or score < best[0]:
            aligned_pi = pi_np[list(perm)]
            best = (score, float(errors.max()), float(np.abs(aligned_pi - true_pi).sum()))

    return {
        "shift_mean_abs_error": best[0],
        "shift_max_abs_error": best[1],
        "pi_l1_error": best[2],
    }


class ShiftMixtureModel(nn.Module):
    """Global shift-mixture decoder + closed-form E-step (no learned encoder)."""

    def __init__(self, M: int, K: int, sigma2: float, sinkhorn_iters: int = 50):
        super().__init__()
        self.M = M
        self.K = K
        self.sigma2 = float(sigma2)
        self.sinkhorn_iters = sinkhorn_iters
        # Alias kept so duck-typed callers (metrics, diagnostics) see something familiar.
        self.num_mixture_components = K

        self.register_buffer("shifts", torch.zeros(K))
        self.register_buffer("log_pi", torch.full((K,), -math.log(K)))

    def init_params(self, x=None, y=None, init_shifts=None, generator: torch.Generator = None):
        if init_shifts is not None:
            self.shifts.copy_(torch.as_tensor(init_shifts, device=self.shifts.device, dtype=self.shifts.dtype))
        elif x is not None and y is not None:
            # Sample-based GMM init: start components on the densest peaks of the
            # empirical pairwise-difference distribution (the true shift modes).
            self.shifts.copy_(self._histogram_mode_shifts(x, y, generator=generator))
        else:
            self.shifts.copy_(self._grid_shifts(generator=generator))
        self.log_pi.fill_(-math.log(self.K))

    def _grid_shifts(self, generator: torch.Generator = None) -> torch.Tensor:
        # Evenly-spaced on the torus + small jitter. Picking shifts close together
        # is the main local-optimum trap (the modes collapse onto one peak).
        base = (torch.arange(self.K, device=self.shifts.device, dtype=self.shifts.dtype) + 0.5) / self.K
        if generator is not None:
            jitter = (torch.rand(self.K, generator=generator, device=self.shifts.device) - 0.5) * 0.1 / self.K
        else:
            jitter = (torch.rand(self.K, device=self.shifts.device) - 0.5) * 0.1 / self.K
        return (base + jitter) % 1.0

    @torch.no_grad()
    def _histogram_mode_shifts(self, x: torch.Tensor, y: torch.Tensor,
                               generator: torch.Generator = None) -> torch.Tensor:
        """Pick K initial shifts from the densest peaks of the pairwise-difference
        histogram. x, y: [N, M]. Returns [K] shifts on shifts.device/dtype."""
        device, dtype = self.shifts.device, self.shifts.dtype
        sigma = max(math.sqrt(self.sigma2), 1e-3)

        # All pairwise circular diffs d = (y_k - x_m) mod 1; peaks sit at true shifts.
        diff = (y.unsqueeze(1) - x.unsqueeze(-1)).remainder(1.0).reshape(-1).to(torch.float32)
        n_bins = max(4 * self.K, int(round(1.0 / sigma)))  # ~one sigma per bin
        hist = torch.histc(diff, bins=n_bins, min=0.0, max=1.0).to(device=device, dtype=dtype)
        centers = (torch.arange(n_bins, device=device, dtype=dtype) + 0.5) / n_bins

        # Circular smoothing so a single physical peak isn't split across bins.
        kernel = torch.tensor([1.0, 2.0, 1.0], device=device, dtype=dtype)
        kernel = kernel / kernel.sum()
        pad = len(kernel) // 2
        hist_padded = torch.cat([hist[-pad:], hist, hist[:pad]])
        hist = torch.nn.functional.conv1d(
            hist_padded.view(1, 1, -1), kernel.view(1, 1, -1)
        ).view(-1)

        # Greedy non-max suppression: take the highest bin, suppress a circular
        # neighborhood of width min_sep around it, repeat.
        min_sep = max(3.0 * sigma, 1.0 / (2 * self.K))
        shifts = []
        scores = hist.clone()
        for _ in range(self.K):
            if torch.all(scores <= -float("inf")):
                break
            peak = int(torch.argmax(scores).item())
            shifts.append(centers[peak])
            circ = (centers - centers[peak] + 0.5).remainder(1.0) - 0.5  # [-0.5, 0.5)
            scores[circ.abs() < min_sep] = -float("inf")

        # Fewer than K distinct peaks survived: pad with grid positions.
        if len(shifts) < self.K:
            grid = self._grid_shifts(generator=generator)
            shifts.extend(grid[len(shifts):])

        return torch.stack(shifts[:self.K]).to(device=device, dtype=dtype)

    def log_pair_likelihood_batched(self, x: torch.Tensor, y: torch.Tensor):
        """L[B,M,M] = log Σ_c π_c·N_circ(y_k - x_m - shift_c; σ²).
        Also returns log_w[B,M,M,K] — component log-posterior given (m,k)."""
        B = x.shape[0]
        x_exp = x.view(B, self.M, 1, 1)            # [B, M, 1, 1]
        y_exp = y.view(B, 1, self.M, 1)            # [B, 1, M, 1]
        shifts_exp = self.shifts.view(1, 1, 1, self.K)
        delta = circular_residual(y_exp, x_exp + shifts_exp)  # [B, M, M, K]

        sigma2 = self.sigma2
        log_norm = -0.5 * math.log(2.0 * math.pi * sigma2)
        log_p_c = log_norm - 0.5 * (delta ** 2) / sigma2      # [B, M, M, K]
        log_pi = torch.log_softmax(self.log_pi, dim=-1).view(1, 1, 1, self.K)
        joint = log_pi + log_p_c
        L = torch.logsumexp(joint, dim=-1)                    # [B, M, M]
        log_w = joint - L.unsqueeze(-1)                       # [B, M, M, K]
        return L, log_w

    @torch.no_grad()
    def e_step_batched(self, x: torch.Tensor, y: torch.Tensor, tau: float):
        L, log_w = self.log_pair_likelihood_batched(x, y)
        log_q = log_sinkhorn(L / tau, n_iters=self.sinkhorn_iters)
        q = torch.exp(log_q)
        w = torch.exp(log_w)
        return q, w, L

    @torch.no_grad()
    def infer_z(self, x: torch.Tensor, y: torch.Tensor, temperature=None):
        tau = float(temperature) if temperature is not None else 1.0
        q, _, _ = self.e_step_batched(x.unsqueeze(0), y.unsqueeze(0), tau)
        return q.squeeze(0)

    @torch.no_grad()
    def infer_z_batched(self, x: torch.Tensor, y: torch.Tensor, temperature=None):
        tau = float(temperature) if temperature is not None else 1.0
        q, _, _ = self.e_step_batched(x, y, tau)
        return q


@torch.no_grad()
def m_step(
    x: torch.Tensor,        # [B, M]
    y: torch.Tensor,        # [B, M]
    q: torch.Tensor,        # [B, M, M]
    w: torch.Tensor,        # [B, M, M, K]
    prev_shifts: torch.Tensor,
):
    """Closed-form parameter update. Returns (new_shifts[K], new_log_pi[K])."""
    r = q.unsqueeze(-1) * w                                   # [B, M, M, K]
    S = r.sum(dim=(0, 1, 2))                                  # [K]

    log_pi_new = torch.log(torch.clamp(S, min=1e-12))
    log_pi_new = log_pi_new - torch.logsumexp(log_pi_new, dim=-1)

    two_pi = 2.0 * math.pi
    diff = y.unsqueeze(1) - x.unsqueeze(-1)                   # [B, M, M] = y_k - x_m
    sin_t = torch.sin(two_pi * diff).unsqueeze(-1)            # [B, M, M, 1]
    cos_t = torch.cos(two_pi * diff).unsqueeze(-1)
    sin_sum = (r * sin_t).sum(dim=(0, 1, 2))                  # [K]
    cos_sum = (r * cos_t).sum(dim=(0, 1, 2))

    new_shifts = torch.atan2(sin_sum, cos_sum) / two_pi
    new_shifts = new_shifts.remainder(1.0)
    # If a component lost all responsibility, keep the previous shift to avoid NaN.
    empty = S < 1e-12
    new_shifts = torch.where(empty, prev_shifts, new_shifts)
    return new_shifts, log_pi_new


def train_em(
    sigma: float = 0.05,
    M: int = 10,
    num_iterations: int = 200,
    eval_every: int = 10,
    K: int = 2,
    tau: float = 1.0,
    sinkhorn_iters: int = 50,
    val_num_batches: int = 16,
    fixed_val_batches: bool = False,
    eval_match_method: str = "argmax",
    fixed_train_data=None,
    init_shifts=None,
    seed: int = None,
    summary_writer=None,
    run_tag: str = None,
    l2_eval_every: int = None,
    l2_eval_grid: int = 100,
    true_shifts=(0.0, 0.3),
):
    """Closed-form EM trainer mirroring `train_model`'s signature where applicable.

    Requires `fixed_train_data=(x_train [N,M], y_train [N,M])` — EM runs full-batch
    over all N groups every iteration. Returns a `ShiftMixtureModel` with the same
    history attributes (.train_losses, .train_accuracies, .iteration_steps) so
    `visualize_results` / `compute_L2_density_error` can consume it.
    """
    if fixed_train_data is None:
        raise ValueError("train_em requires fixed_train_data=(x_train [N,M], y_train [N,M])")

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    x_train, y_train = fixed_train_data
    x_train = x_train.to(device)
    y_train = y_train.to(device)

    model = ShiftMixtureModel(M=M, K=K, sigma2=sigma**2, sinkhorn_iters=sinkhorn_iters).to(device)
    gen = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
    model.init_params(x=x_train, y=y_train, init_shifts=init_shifts, generator=gen)

    model.train_losses = []
    model.train_recon = []
    model.train_kl = []
    model.train_accuracies = []
    model.iteration_steps = []
    model.train_recon_per_M = []
    model.l2_errors = []
    model.l2_steps = []
    model.shift_mean_abs_errors = []
    model.shift_max_abs_errors = []
    model.pi_l1_errors = []
    model.sinkhorn_row_errors = []
    model.sinkhorn_col_errors = []

    print(
        f"Training EM Shift-Mixture | M={M}, sigma={sigma}, K={K}, "
        f"N_train={x_train.shape[0]}, sinkhorn_iters={sinkhorn_iters}, "
        f"tau={tau}, device={device}"
    )
    if summary_writer is not None and run_tag is not None:
        summary_writer.add_text("meta/run_tag", run_tag, 0)

    val_num_batches = max(1, int(val_num_batches))
    fixed_val_data = []
    if fixed_val_batches:
        for _ in range(val_num_batches):
            vx, vy, perm = sample_batch_with_perm(M, sigma, return_perm=True)
            fixed_val_data.append((
                torch.tensor(vx, device=device, dtype=torch.float32),
                torch.tensor(vy, device=device, dtype=torch.float32),
                np.argsort(perm),
            ))

    tau_cur = float(tau)  # fixed Sinkhorn temperature (no annealing)
    for iteration in range(num_iterations):
        q, w, L = model.e_step_batched(x_train, y_train, tau_cur)

        if iteration % eval_every == 0 or iteration == num_iterations - 1:
            # Reconstruction term: Σ_{m,k} q[m,k] · (−log p(y_k|x_m, θ)) averaged over batch.
            recon = float((-q * L).sum(dim=(-1, -2)).mean().item())
            recon_per_M = recon / M
            row_error = float((q.sum(dim=-1) - 1.0).abs().max().item())
            col_error = float((q.sum(dim=-2) - 1.0).abs().max().item())
            model.iteration_steps.append(iteration)
            model.train_losses.append(recon)
            model.train_recon.append(recon)
            model.train_recon_per_M.append(recon_per_M)
            model.train_kl.append(0.0)
            model.sinkhorn_row_errors.append(row_error)
            model.sinkhorn_col_errors.append(col_error)

            if fixed_val_batches:
                eval_source = fixed_val_data
            else:
                eval_source = []
                for _ in range(val_num_batches):
                    vx, vy, perm = sample_batch_with_perm(M, sigma, return_perm=True)
                    eval_source.append((
                        torch.tensor(vx, device=device, dtype=torch.float32),
                        torch.tensor(vy, device=device, dtype=torch.float32),
                        np.argsort(perm),
                    ))
            eval_accs = []
            for vx, vy, true_perm in eval_source:
                z = model.infer_z(vx, vy, temperature=tau_cur)
                eval_accs.append(compute_accuracy(true_perm, z, method=eval_match_method))
            acc = float(np.mean(eval_accs))
            model.train_accuracies.append(acc)

            pi = torch.softmax(model.log_pi, dim=-1).detach().cpu().tolist()
            shifts = model.shifts.detach().cpu().tolist()
            param_errors = shift_parameter_errors(shifts, pi, true_shifts=true_shifts)
            model.shift_mean_abs_errors.append(param_errors["shift_mean_abs_error"])
            model.shift_max_abs_errors.append(param_errors["shift_max_abs_error"])
            model.pi_l1_errors.append(param_errors["pi_l1_error"])

            l2_err = None
            should_eval_l2 = (
                l2_eval_every is not None
                and (iteration % max(1, int(l2_eval_every)) == 0 or iteration == num_iterations - 1)
            )
            if should_eval_l2:
                from metrics import compute_L2_density_error

                l2_err = float(compute_L2_density_error(model, sigma=sigma, E=l2_eval_grid))
                model.l2_steps.append(iteration)
                model.l2_errors.append(l2_err)

            if summary_writer is not None:
                summary_writer.add_scalar("em/recon", recon, iteration)
                summary_writer.add_scalar("em/recon_per_M", recon_per_M, iteration)
                summary_writer.add_scalar("em/accuracy", acc, iteration)
                summary_writer.add_scalar("em/tau", tau_cur, iteration)
                summary_writer.add_scalar("sinkhorn/max_row_error", row_error, iteration)
                summary_writer.add_scalar("sinkhorn/max_col_error", col_error, iteration)
                summary_writer.add_scalar("params/shift_mean_abs_error", param_errors["shift_mean_abs_error"], iteration)
                summary_writer.add_scalar("params/shift_max_abs_error", param_errors["shift_max_abs_error"], iteration)
                summary_writer.add_scalar("params/pi_l1_error", param_errors["pi_l1_error"], iteration)
                for idx, value in enumerate(shifts):
                    summary_writer.add_scalar(f"em/shift_{idx}", value, iteration)
                for idx, value in enumerate(pi):
                    summary_writer.add_scalar(f"em/pi_{idx}", value, iteration)
                if l2_err is not None:
                    summary_writer.add_scalar("em/L2_density_error", l2_err, iteration)
                summary_writer.flush()

            l2_msg = f" | L2 {l2_err:.4f}" if l2_err is not None else ""
            print(
                f"Iter {iteration:4d} | recon {recon:.4f} | recon/M {recon_per_M:.4f} "
                f"| tau {tau_cur:.3f} | acc {acc:.3f}{l2_msg} "
                f"| shifts [{', '.join(f'{s:.3f}' for s in shifts)}] "
                f"| pi [{', '.join(f'{p:.2f}' for p in pi)}]"
            )

        new_shifts, new_log_pi = m_step(x_train, y_train, q, w, model.shifts)
        model.shifts.copy_(new_shifts)
        model.log_pi.copy_(new_log_pi)

    print("EM training finished.")
    return model
