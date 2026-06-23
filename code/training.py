import numpy as np
import torch

from config import device
from data import sample_batch_tensor, sample_batch_with_perm
from metrics import compute_accuracy
from model import CategoricalTorusVAE


# ==================== ОБУЧЕНИЕ ====================
def train_model(
    sigma=0.05,
    M=10,
    num_iterations=2500,
    lr=1e-3,
    eval_every=50,
    batch_size=64,
    tau=1.0,
    kl_warmup_steps=1000,
    grad_clip=1.0,
    val_num_batches: int = 1,
    fixed_val_batches: bool = True,
    eval_match_method: str = "argmax",
    num_mixture_components: int = 2,
    sinkhorn_backend: str = "implicit",
    sinkhorn_iters: int = 20,
    fixed_train_data=None,
    mixture_mean_init: str = "spread",
):
    # === CHANGED: explicit mixture hyperparameter ===
    model = CategoricalTorusVAE(
        M=M,
        hidden_dim=128,
        sigma2=float(sigma**2),
        temperature=tau,
        num_mixture_components=num_mixture_components,
        sinkhorn_backend=sinkhorn_backend,
        sinkhorn_iters=sinkhorn_iters,
        mixture_mean_init=mixture_mean_init,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train_losses = []
    model.train_recon = []
    model.train_kl = []
    model.train_accuracies = []
    model.iteration_steps = []

    batch_size = max(1, int(batch_size))
    print(
        f"Training Categorical Torus VAE | M={M}, sigma={sigma}, "
        f"num_mixture_components={num_mixture_components}, "
        f"sinkhorn_backend={sinkhorn_backend}, sinkhorn_iters={sinkhorn_iters}, "
        f"problems_per_iter={batch_size}, device={device}"
    )

    val_num_batches = max(1, int(val_num_batches))
    fixed_val_data = []
    if fixed_val_batches:
        for _ in range(val_num_batches):
            val_x_np, val_y_np, perm = sample_batch_with_perm(M, sigma, return_perm=True)
            fixed_val_data.append(
                (
                    torch.tensor(val_x_np, device=device, dtype=torch.float32),
                    torch.tensor(val_y_np, device=device, dtype=torch.float32),
                    np.argsort(perm),
                )
            )

    for iteration in range(num_iterations):
        model.train()

        # tau is a fixed Sinkhorn temperature (set on the model at construction);
        # only the KL weight beta is annealed.
        beta = min(1.0, iteration / max(1, kl_warmup_steps))

        if fixed_train_data is not None:
            x_train, y_train = fixed_train_data
            N_train = x_train.shape[0]
            if batch_size >= N_train:
                x_batch, y_batch = x_train, y_train
            else:
                idx = torch.randint(0, N_train, (batch_size,), device=device)
                x_batch, y_batch = x_train[idx], y_train[idx]
        else:
            x_batch, y_batch, _ = sample_batch_tensor(batch_size, M, sigma, device)
        total, recon, kl = model.loss_batched(x_batch, y_batch, beta=beta, return_parts=True)

        optimizer.zero_grad()
        total.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if iteration % eval_every == 0:
            model.iteration_steps.append(iteration)
            model.train_losses.append(float(total.item()))
            model.train_recon.append(float(recon.item()))
            model.train_kl.append(float(kl.item()))

            model.eval()
            with torch.no_grad():
                eval_accs = []
                if fixed_val_batches:
                    eval_source = fixed_val_data
                else:
                    eval_source = []
                    for _ in range(val_num_batches):
                        val_x_np, val_y_np, perm = sample_batch_with_perm(M, sigma, return_perm=True)
                        eval_source.append(
                            (
                                torch.tensor(val_x_np, device=device, dtype=torch.float32),
                                torch.tensor(val_y_np, device=device, dtype=torch.float32),
                                np.argsort(perm),
                            )
                        )

                for val_x, val_y, val_true_perm in eval_source:
                    # Детерминированно: infer_z без Gumbel
                    z = model.infer_z(val_x, val_y, temperature=tau)  # [M,M]
                    eval_accs.append(compute_accuracy(val_true_perm, z, method=eval_match_method))
                acc = float(np.mean(eval_accs))

            model.train_accuracies.append(float(acc))

            print(
                f"Iter {iteration:4d} | total {total.item():.4f} | recon {recon.item():.4f} | kl {kl.item():.4f} "
                f"| beta {beta:.2f} | tau {tau:.3f} | acc {acc:.3f}"
            )

    print("Training finished.")
    return model
