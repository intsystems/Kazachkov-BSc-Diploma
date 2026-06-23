import os as _os

import numpy as np
import pandas as pd
import torch

from data import dens_gauss_shift

try:
    from scipy.optimize import linear_sum_assignment as _linear_sum_assignment
except Exception:
    _linear_sum_assignment = None


def _hungarian_perm_from_scores(score_matrix: np.ndarray):
    """Опциональный conversion scores[M,M] -> permutation[M] через Hungarian."""
    if _linear_sum_assignment is None:
        return None
    row_ind, col_ind = _linear_sum_assignment(-score_matrix)  # maximize score
    pred = np.full(score_matrix.shape[0], -1, dtype=np.int64)
    pred[row_ind] = col_ind
    return pred


def compute_accuracy(true_perm, predicted_q, method: str = "argmax"):
    """
    Точность по соответствию m -> k.
    true_perm: [M] или [B,M], где true_perm[m] = индекс k в permuted-y.
    predicted_q: [M,M] или [B,M,M] (torch/np), где ось -1 это k.
    method: 'argmax' (по строкам) или 'hungarian' (если SciPy доступен).
    """
    true_perm_np = np.asarray(true_perm)

    if torch.is_tensor(predicted_q):
        score = predicted_q.detach().cpu().numpy()
    else:
        score = np.asarray(predicted_q)

    if score.ndim == 2:
        score = score[None, ...]
    if true_perm_np.ndim == 1:
        true_perm_np = true_perm_np[None, ...]

    if score.shape[0] != true_perm_np.shape[0]:
        if true_perm_np.shape[0] == 1:
            true_perm_np = np.repeat(true_perm_np, score.shape[0], axis=0)
        else:
            raise ValueError(f"Batch mismatch: true_perm {true_perm_np.shape}, predicted_q {score.shape}")

    accs = []
    for b in range(score.shape[0]):
        if method == "hungarian":
            pred = _hungarian_perm_from_scores(score[b])
            if pred is None:
                pred = np.argmax(score[b], axis=-1)
        else:
            pred = np.argmax(score[b], axis=-1)
        accs.append(np.mean(pred == true_perm_np[b]))
    return float(np.mean(accs))


def compute_L2_density_error(model, sigma, E=100):
    """L2 error between CatVAE estimated joint density and true density on E×E grid.

    The decoder models p(y|x) as a wrapped Gaussian mixture.
    Since p(x) = 1 (uniform on [0,1)), the joint density is p(x,y) = p(y|x).
    """
    x_e = np.linspace(0, 1, E, endpoint=False)
    y_e = np.linspace(0, 1, E, endpoint=False)

    # True density (same formula as reference paper)
    true_density = dens_gauss_shift(*np.meshgrid(x_e, y_e), shift=0.3, std=sigma, shift_prob=0.5)

    # CatVAE estimated density from decoder
    model.eval()
    if hasattr(model, "shifts") and hasattr(model, "log_pi"):
        # EM ShiftMixtureModel: global shifts → means[i,c] = x_e[i] + shift_c.
        shifts_np = model.shifts.detach().cpu().numpy()                         # [K]
        weights_row = torch.softmax(model.log_pi, dim=-1).detach().cpu().numpy()  # [K]
        means_np = (x_e[:, None] + shifts_np[None, :]) % 1.0                    # [E, K]
        weights = np.broadcast_to(weights_row[None, :], means_np.shape).copy()  # [E, K]
        sigma2 = model.sigma2
    else:
        device = next(model.parameters()).device
        with torch.no_grad():
            x_t = torch.tensor(x_e, device=device, dtype=torch.float32)
            two_pi = 2.0 * torch.pi
            x_feat = torch.stack([torch.sin(two_pi * x_t), torch.cos(two_pi * x_t)], dim=-1)
            raw = model.decoder.net(x_feat).view(E, model.num_mixture_components, 2)
            means_np = raw[..., 0].cpu().numpy()   # [E, K]
            weights = torch.softmax(raw[..., 1], dim=-1).cpu().numpy()  # [E, K]
        sigma2 = model.decoder.sigma2
    std_val = np.sqrt(sigma2)
    o = int(np.ceil(4 * std_val))
    offsets = np.arange(-o, o + 1, dtype=float)

    # est_density[j, i] = p(y_e[j] | x_e[i])
    est_density = np.zeros((E, E))
    for k in range(model.num_mixture_components):
        # delta[j, i] = y_e[j] - mu_k(x_e[i])
        delta = y_e[:, None] - means_np[None, :, k]  # [E_y, E_x]
        # wrapped normal: sum over periodic copies
        log_terms = -(delta[:, :, None] - offsets[None, None, :]) ** 2 / (2 * sigma2)
        wrapped = np.exp(log_terms).sum(axis=-1) / (np.sqrt(2 * np.pi * sigma2))
        est_density += weights[None, :, k] * wrapped

    return np.linalg.norm(true_density - est_density) / E


def _load_reference_csv(csv_name):
    """Load reference paper CSV data from BatchTransferOperator/ directory."""
    try:
        base = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    except NameError:
        base = _os.getcwd()
    path = _os.path.join(base, "BatchTransferOperator", csv_name)
    if not _os.path.isfile(path):
        # If running from code/ subdirectory, go one level up
        alt = _os.path.join(_os.path.dirname(base), "BatchTransferOperator", csv_name)
        if _os.path.isfile(alt):
            path = alt
        else:
            raise FileNotFoundError(
                f"Cannot find {csv_name}. Looked at:\n  {path}\n  {alt}\n"
                f"Run from the repo root or set working directory accordingly."
            )
    return pd.read_csv(path)
