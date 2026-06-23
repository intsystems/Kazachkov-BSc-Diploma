from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import device
from data import sample_batch_with_perm, sample_dataset_torus
from metrics import _hungarian_perm_from_scores, compute_accuracy


# ==================== ФУНКЦИИ ДЛЯ ВАЛИДАЦИИ ====================

def visualize_results(model, sigma, num_samples=1000, num_pred=1000, save_path=None):
    """Визуализация результатов после обучения"""
    model.eval()

    # 1. Генерируем тестовые данные (исходное распределение)
    x_test, y_test = sample_dataset_torus(num_samples, sigma)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0,0].hist2d(x_test, y_test, bins=50, range=[[0,1],[0,1]], cmap='viridis')
    axes[0,0].set_title(f'True Distribution (sigma={sigma})')
    axes[0,0].set_xlabel('x')
    axes[0,0].set_ylabel('y')

    # === CHANGED: sample from decoder mixture ===
    with torch.no_grad():
        x_batch = torch.tensor(x_test[:num_pred], device=device, dtype=torch.float32)
        M = model.M
        std = float(np.sqrt(model.decoder.sigma2))

        y_pred_chunks = []
        for i in range(0, x_batch.shape[0], M):
            x_sub = x_batch[i:i + M]
            n = x_sub.shape[0]
            if n < M:
                pad = M - n
                x_sub = torch.cat([x_sub, x_sub[:pad]], dim=0)

            if model.decoder.num_mixture_components == 1:
                z_dummy = torch.eye(M, device=x_sub.device, dtype=x_sub.dtype)
                mean_sub = model.decoder(x_sub, z_dummy, permute=False)
            else:
                component_means, component_logits = model.decoder(x_sub, return_mixture_params=True)
                component_probs = torch.softmax(component_logits, dim=-1)
                component_idx = torch.multinomial(component_probs, num_samples=1).squeeze(-1)
                mean_sub = component_means.gather(-1, component_idx.unsqueeze(-1)).squeeze(-1)

            y_sub = mean_sub[:n] + std * torch.randn_like(mean_sub[:n])
            y_pred_chunks.append(torch.remainder(y_sub, 1.0))

        y_pred_noisy = torch.cat(y_pred_chunks, dim=0).cpu().numpy()

    axes[0, 1].hist2d(x_test[:num_pred], y_pred_noisy, bins=50, range=[[0, 1], [0, 1]], cmap="viridis")
    axes[0, 1].set_title("Model Samples: decoder mixture + noise (wrapped)")
    axes[0, 1].set_xlabel("x")
    axes[0, 1].set_ylabel("predicted y")

    # 3. График accuracy от итерации (вместо гистограммы)
    if hasattr(model, 'train_accuracies') and len(model.train_accuracies) > 0:
        if hasattr(model, "iteration_steps") and len(model.iteration_steps) == len(model.train_accuracies):
            iter_steps = model.iteration_steps
        else:
            iter_steps = list(range(len(model.train_accuracies)))

        axes[1, 0].plot(iter_steps, model.train_accuracies, "b-", linewidth=1.5, label="Accuracy")
        axes[1, 0].set_xlabel("Iteration")
        axes[1, 0].set_ylabel("Permutation Accuracy")
        axes[1, 0].set_title("Accuracy vs Iteration")
        axes[1, 0].set_ylim(-0.05, 1.05)
        axes[1, 0].grid(True, alpha=0.3)

        rand_acc = 1.0 / model.M
        axes[1, 0].axhline(rand_acc, color="gray", linestyle="--", label=f"Random (1/M={rand_acc:.2f})")
        axes[1, 0].legend()
    else:
        axes[1, 0].text(0.5, 0.5, "No accuracy data", ha="center", va="center", transform=axes[1, 0].transAxes)
        axes[1, 0].set_title("Accuracy vs Iteration")

    # 4) Loss vs Iteration
    if hasattr(model, "train_losses") and len(model.train_losses) > 0:
        if hasattr(model, "iteration_steps") and len(model.iteration_steps) == len(model.train_losses):
            iter_steps = model.iteration_steps
        else:
            iter_steps = list(range(len(model.train_losses)))

        axes[1, 1].plot(iter_steps, model.train_losses, "r-", linewidth=1.5, alpha=0.8, label="total (NELBO)")
        if hasattr(model, "train_recon") and len(model.train_recon) == len(iter_steps):
            axes[1, 1].plot(iter_steps, model.train_recon, "g-", linewidth=1, alpha=0.7, label="recon")
        if hasattr(model, "train_kl") and len(model.train_kl) == len(iter_steps):
            axes[1, 1].plot(iter_steps, model.train_kl, "b-", linewidth=1, alpha=0.7, label="KL")
        axes[1, 1].set_xlabel("Iteration")
        axes[1, 1].set_ylabel("Loss")
        axes[1, 1].set_title("Training Loss")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend(fontsize=9)
    else:
        axes[1, 1].text(0.5, 0.5, "No loss data", ha="center", va="center", transform=axes[1, 1].transAxes)
        axes[1, 1].set_title("Training Loss")

    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)

def visualize_assignment_recovery(model, sigma, M=None, num_groups=500,
                                  eval_temperature=0.2, match_method="argmax",
                                  save_path=None):
    """Визуализация восстановления сопоставления (распутывание перестановки).

    Слева: пары по хранимому (перемешанному) индексу (x_m, y_perm[m]) —
    бесструктурное облако. Справа: восстановленные пары (x_m, y_perm[argmax z_m]),
    зелёным верные назначения, красным ошибочные. Модель-агностично через infer_z,
    работает и для CategoricalTorusVAE (SGD), и для ShiftMixtureModel (EM)."""
    model.eval()
    if M is None:
        M = model.M

    x_shuf, y_shuf = [], []        # пары по хранимому индексу
    x_rec, y_rec, correct = [], [], []  # восстановленные пары + верность
    for _ in range(num_groups):
        x, y_perm, perm = sample_batch_with_perm(M, sigma, return_perm=True)
        inv_perm = np.argsort(perm)  # gt[m] = k

        with torch.no_grad():
            x_t = torch.tensor(x, device=device, dtype=torch.float32)
            y_t = torch.tensor(y_perm, device=device, dtype=torch.float32)
            z = model.infer_z(x_t, y_t, temperature=eval_temperature)  # [M, M]
        z_np = z.detach().cpu().numpy()

        if match_method == "hungarian":
            pred = _hungarian_perm_from_scores(z_np)
            if pred is None:
                pred = np.argmax(z_np, axis=-1)
        else:
            pred = np.argmax(z_np, axis=-1)

        x_shuf.append(x); y_shuf.append(y_perm)
        x_rec.append(x); y_rec.append(y_perm[pred])
        correct.append(pred == inv_perm)

    x_shuf = np.concatenate(x_shuf); y_shuf = np.concatenate(y_shuf)
    x_rec = np.concatenate(x_rec); y_rec = np.concatenate(y_rec)
    correct = np.concatenate(correct)
    acc = float(np.mean(correct))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    axes[0].scatter(x_shuf, y_shuf, s=6, alpha=0.3, color="tab:gray")
    axes[0].set_title("Shuffled input (paired by stored index)")

    axes[1].scatter(x_rec[correct], y_rec[correct], s=6, alpha=0.4,
                    color="tab:green", label="correct")
    axes[1].scatter(x_rec[~correct], y_rec[~correct], s=6, alpha=0.4,
                    color="tab:red", label="wrong")
    axes[1].set_title(f"Recovered pairing (accuracy={acc:.1%})")
    axes[1].legend(loc="upper left", markerscale=2)

    for i, ax in enumerate(axes):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("x")
        if i == 0:
            ax.set_ylabel("y")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


## Функции для экспериментов
def evaluate_accuracy(model, sigma=0.05, num_trials=50, M_test=None, eval_temperature=0.2, match_method: str = "argmax"):
    """
    Оценивает точность предсказания перестановки на новых данных.

    Args:
        model: обученная CategoricalTorusVAE (с фикс. M)
        sigma: уровень шума
        num_trials: число тестовых батчей
        M_test: если None — используем model.M, иначе — проверяем совместимость
    """
    model.eval()

    if M_test is None:
        M_test = model.M
    else:
        assert M_test == model.M, f"Model trained with M={model.M}, but M_test={M_test} requested!"

    accuracies = []
    for _ in range(num_trials):
        x, y_perm, perm = sample_batch_with_perm(M_test, sigma, return_perm=True)
        inv_perm = np.argsort(perm)

        with torch.no_grad():
            x_tensor = torch.tensor(x, device=device, dtype=torch.float32)
            y_tensor = torch.tensor(y_perm, device=device, dtype=torch.float32)

            z = model.infer_z(x_tensor, y_tensor, temperature=eval_temperature)  # [M,M], без Gumbel
            acc = compute_accuracy(inv_perm, z, method=match_method)  # ожидаем gt[m]=k

            # _, _, z = model(x_tensor, y_tensor)  # z, а не q
            # acc = compute_accuracy(true_perm, z)  # z вместо q
        accuracies.append(float(acc))
    return np.mean(accuracies)
