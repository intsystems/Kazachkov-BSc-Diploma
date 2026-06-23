import csv
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import device
from data import sample_batch_tensor
from diagnostics import evaluate_accuracy, visualize_assignment_recovery, visualize_results
from em import shift_parameter_errors, train_em
from metrics import _load_reference_csv, compute_L2_density_error
from training import train_model


# ==================== shared experiment grids / hyperparameters ====================
# Kept at module scope so the parallel worker functions and the driver loops read
# the exact same values (no duplicated magic numbers across the two code paths).
SIGMAS = [0.01, 0.025, 0.05]
NUM_MIXTURE_COMPONENTS = 2
NUM_SEEDS = 5

# varying M (Data_7.csv grid, capped at 512: encoder builds an explicit
# [B, M, M, 4] tensor, so M in the thousands is infeasible).
M_VALUES = [1, 2, 3, 4, 6, 8, 10, 13, 17, 22, 29, 38, 49, 64, 83, 107, 139,
            181, 234, 304, 395, 512]
N_TRAIN_M = 20
# Cap on groups * M*M activations per SGD step: keeps the fixed pool at N=20
# while shrinking groups-per-step for large M (M<=64 -> full batch of 20).
PAIR_BUDGET = 200_000

# varying N (Data_5.csv grid, M=20, std=0.05).
M_FIXED_N = 20
TARGET_N_VALUES = [1, 2, 3, 4, 5, 7, 8, 10, 12, 15, 19, 23, 28, 34, 41, 50,
                   56, 63, 72, 81, 91, 103, 117, 132, 149, 168, 178, 190,
                   214, 219, 270, 331, 407]

# Iterations are fixed (numerics-preserving): the parallelism only changes how the
# independent (sigma, size, seed) runs are *scheduled*, never the training itself.
SGD_NUM_ITERATIONS = 5000

# EM convergence diagnostics. These defaults intentionally focus on the M range
# where varying_M_em.png starts to degrade.
EM_ITERS_M_VALUES = [64, 83, 107, 139, 181]
EM_ITERS_NUM_ITERATIONS = [200, 500, 1000, 2000]
EM_ITERS_SINKHORN_VALUES = [50, 100, 200]


# ==================== GPU pool plumbing ====================
def _gpu_pool_init(counter):
    """Pool worker initializer: pin this process to one CUDA card, round-robin.

    config.device is an index-less torch.device("cuda"), so every .to(device) /
    torch.tensor(..., device=device) lands on the process's *current* CUDA device.
    Assigning each worker a distinct current device here is therefore enough to
    spread runs across all visible cards — with no change to config/model code.

    torch.cuda.device_count() already honors CUDA_VISIBLE_DEVICES, which is what
    scripts/remote_run.sh's GPU=<id> sets: GPU pinned -> one card -> every worker
    uses it; GPU unset -> workers fan out across all physical cards.
    """
    n = torch.cuda.device_count()
    if n <= 0:
        return
    with counter.get_lock():
        ordinal = counter.value
        counter.value += 1
    torch.cuda.set_device(ordinal % n)


def _run_grid(worker_fn, work_items, max_workers):
    """Yield (item, result) as each independent work item completes.

    max_workers<=1 runs inline in the main process, in submission order — the exact
    pre-parallel code path, used as a numerics-equality baseline and CPU fallback.
    Otherwise a spawn-based process pool runs items concurrently on the visible
    GPU(s); completion order is arbitrary but the per-item results are identical.
    """
    if max_workers <= 1:
        for item in work_items:
            yield item, worker_fn(*item)
        return

    ctx = multiprocessing.get_context("spawn")
    counter = ctx.Value("i", 0)
    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=ctx,
        initializer=_gpu_pool_init,
        initargs=(counter,),
    ) as ex:
        futures = {ex.submit(worker_fn, *item): item for item in work_items}
        for fut in as_completed(futures):
            yield futures[fut], fut.result()


def _max_workers():
    return max(1, int(os.environ.get("MAX_WORKERS", "8")))


def _parse_env_list(name, default, cast):
    raw = os.environ.get(name)
    if not raw:
        return list(default)
    return [cast(item.strip()) for item in raw.split(",") if item.strip()]


def _plot_varying_M(results, sigmas, ref_df, save_path):
    """Render the 'varying M' figure from a (possibly partial) results dict.

    Safe to call repeatedly during a run: it plots only the M values whose seed
    list is already filled, so the curve grows left-to-right as points complete.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {0.01: 'tab:blue', 0.025: 'tab:orange', 0.05: 'tab:green'}

    # CatVAE curves (solid, median + IQR band over seeds)
    for sigma in sigmas:
        Ms = [m for m in sorted(results[sigma]) if results[sigma][m]]
        if not Ms:
            continue
        med = [np.median(results[sigma][m]) for m in Ms]
        q25 = [np.quantile(results[sigma][m], 0.25) for m in Ms]
        q75 = [np.quantile(results[sigma][m], 0.75) for m in Ms]
        ax.plot(Ms, med, 'o-', color=colors[sigma],
                label=rf'CatVAE $\sigma$={sigma}', linewidth=2, markersize=7)
        ax.fill_between(Ms, q25, q75, color=colors[sigma], alpha=0.15)

    # Reference curves (dashed, median + IQR band)
    for std_val in sigmas:
        sub = ref_df[ref_df["std"] == std_val]
        if sub.empty:
            continue
        grouped = sub.groupby("M")["L2"]
        M_ref = sorted(grouped.groups.keys())
        medians = [grouped.get_group(m).median() for m in M_ref]
        q25 = [grouped.get_group(m).quantile(0.25) for m in M_ref]
        q75 = [grouped.get_group(m).quantile(0.75) for m in M_ref]
        ax.plot(M_ref, medians, 's--', color=colors[std_val],
                label=rf'Ref $\sigma$={std_val}', linewidth=1.5, markersize=5, alpha=0.7)
        ax.fill_between(M_ref, q25, q75, color=colors[std_val], alpha=0.1)

    ax.set_xscale('log')
    ax.set_xlabel('Problem Size M (Number of Pairs)', fontsize=12)
    ax.set_ylabel('L2 Density Error', fontsize=12)
    ax.set_title('Effect of M on Density Estimation (CatVAE vs Reference)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    Path("figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_varying_N(results, sigmas, ref_df, M, save_path):
    """Render the 'varying N' figure from a (possibly partial) results dict.

    Safe to call repeatedly during a run: it plots only the N values whose seed
    list is already filled, so the curve grows left-to-right as points complete.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {0.01: 'tab:blue', 0.025: 'tab:orange', 0.05: 'tab:green'}

    # CatVAE curves (solid, median + IQR band over seeds)
    for sigma in sigmas:
        Ns = [n for n in sorted(results[sigma]) if results[sigma][n]]
        if not Ns:
            continue
        med = [np.median(results[sigma][n]) for n in Ns]
        q25 = [np.quantile(results[sigma][n], 0.25) for n in Ns]
        q75 = [np.quantile(results[sigma][n], 0.75) for n in Ns]
        ax.plot(Ns, med, 'o-', color=colors[sigma],
                label=rf'CatVAE $\sigma$={sigma}', linewidth=2, markersize=7)
        ax.fill_between(Ns, q25, q75, color=colors[sigma], alpha=0.15)

    # Reference curves (dashed, median + IQR band)
    # Reference varies N with fixed std=0.05 and different epsi values
    ref_epsi_colors = {0.001: 'tab:red', 0.0025: 'tab:purple', 0.01: 'tab:brown'}
    for epsi in [0.001, 0.0025, 0.01]:
        sub = ref_df[np.isclose(ref_df["epsi"], epsi)]
        if sub.empty:
            continue
        grouped = sub.groupby("N")["L2"]
        N_ref = sorted(grouped.groups.keys())
        medians = [grouped.get_group(n).median() for n in N_ref]
        q25 = [grouped.get_group(n).quantile(0.25) for n in N_ref]
        q75 = [grouped.get_group(n).quantile(0.75) for n in N_ref]
        ax.plot(N_ref, medians, 's--', color=ref_epsi_colors[epsi],
                label=rf'Ref $\varepsilon$={epsi}', linewidth=1.5, markersize=5, alpha=0.7)
        ax.fill_between(N_ref, q25, q75, color=ref_epsi_colors[epsi], alpha=0.1)

    ax.set_xscale('log')
    ax.set_xlabel('N (number of groups)', fontsize=12)
    ax.set_ylabel('L2 Density Error', fontsize=12)
    ax.set_title(rf'Effect of Data Amount on Density Estimation (M = {M})', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    Path("figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def experiment_varying_sigma(algorithm: str = "sgd", mixture_mean_init: str = "spread", tau: float = 1.0):
    """Эксперимент с разными значениями sigma (шума в данных)"""
    sigmas = SIGMAS
    M = 2  # фиксируем M для сравнения
    num_mixture_components = NUM_MIXTURE_COMPONENTS
    N_TRAIN = 64  # for EM we need a fixed pool of groups
    init_tag = f"_{mixture_mean_init}" if algorithm == "sgd" else ""
    tau_tag = f"_tau{tau}" if tau != 1.0 else ""

    for sigma in sigmas:
        print(f"\n{'='*50}")
        print(f"Training [{algorithm}] with sigma = {sigma}, M = {M}")
        print(f"{'='*50}")

        if algorithm == "em":
            x_train, y_train, _ = sample_batch_tensor(N_TRAIN, M, sigma, device)
            model = train_em(
                sigma=sigma,
                M=M,
                num_iterations=200,
                eval_every=10,
                K=num_mixture_components,
                tau=tau,
                sinkhorn_iters=50,
                val_num_batches=16,
                fixed_val_batches=False,
                fixed_train_data=(x_train, y_train),
            )
        else:
            model = train_model(
                sigma=sigma,
                M=M,
                num_iterations=2500,
                batch_size=64,
                eval_every=50,
                lr=1e-3,
                tau=tau,
                val_num_batches=16,
                fixed_val_batches=False,
                num_mixture_components=num_mixture_components,
                mixture_mean_init=mixture_mean_init,
            )

        if algorithm == "sgd":
            visualize_results(model, sigma=sigma, save_path=f"figures/sigma_{sigma}_{algorithm}{init_tag}{tau_tag}.png")
            visualize_assignment_recovery(
                model, sigma=sigma,
                save_path=f"figures/sigma_{sigma}_{algorithm}{init_tag}{tau_tag}_recovery.png")

        test_acc = evaluate_accuracy(model, sigma=sigma, num_trials=100)
        print(f" → Test Accuracy: {test_acc:.3f}")


def _run_one_M(sigma, M, seed, mixture_mean_init, tau=None):
    """Train one CatVAE for (sigma, M, seed) via SGD and return its L2 error.

    Top-level (picklable) so the spawn-based process pool can run many of these
    concurrently. The body is identical to the former inner loop of
    experiment_varying_M — same seed, same train_model args — so the result is
    numerically unchanged regardless of how it is scheduled.
    """
    print(f"\n{'='*40}")
    print(f"[sgd] sigma={sigma}, M={M}, seed={seed} (N={N_TRAIN_M}, iters={SGD_NUM_ITERATIONS})")
    print(f"{'='*40}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    x_train, y_train, _ = sample_batch_tensor(N_TRAIN_M, M, sigma, device)
    batch_size = max(1, min(N_TRAIN_M, PAIR_BUDGET // (M * M)))
    model = train_model(
        sigma=sigma,
        M=M,
        num_iterations=SGD_NUM_ITERATIONS,
        batch_size=batch_size,
        fixed_train_data=(x_train, y_train),
        tau=tau,
        sinkhorn_iters=50,
        eval_every=500,
        lr=1e-3,
        val_num_batches=16,
        fixed_val_batches=False,
        num_mixture_components=NUM_MIXTURE_COMPONENTS,
        mixture_mean_init=mixture_mean_init,
    )
    l2_err = compute_L2_density_error(model, sigma=sigma, E=100)
    print(f"  → [sgd] sigma={sigma}, M={M}, seed={seed} L2 error: {l2_err:.4f}")
    return l2_err


def _run_one_M_em(sigma, M, seed, tau=None):
    """EM counterpart of _run_one_M (kept for the sequential EM path)."""
    print(f"\n{'='*40}")
    print(f"[em] sigma={sigma}, M={M}, seed={seed} (N={N_TRAIN_M}, iters=200)")
    print(f"{'='*40}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    x_train, y_train, _ = sample_batch_tensor(N_TRAIN_M, M, sigma, device)
    model = train_em(
        sigma=sigma,
        M=M,
        num_iterations=200,
        eval_every=50,
        K=NUM_MIXTURE_COMPONENTS,
        tau=tau,
        sinkhorn_iters=50,
        val_num_batches=16,
        fixed_val_batches=False,
        fixed_train_data=(x_train, y_train),
        seed=seed,
    )
    l2_err = compute_L2_density_error(model, sigma=sigma, E=100)
    print(f"  → [em] sigma={sigma}, M={M}, seed={seed} L2 error: {l2_err:.4f}")
    return l2_err


def experiment_varying_M(algorithm: str = "sgd", mixture_mean_init: str = "spread", tau: float = 1.0):
    """Эксперимент с разными размерами задачи M (число пар).
    Аналог Section 6.1.3 (Varying M) / Fig 5 из статьи.
    Параметры выровнены с референсом: sigmas={0.01, 0.025, 0.05}, N=20, epsi=0.01.
    Обучаемся на фиксированных N=20 группах (как авторы), повторяем NUM_SEEDS раз
    для оценки разброса (медиана + IQR).

    SGD-прогоны независимы по (sigma, M, seed) и считаются параллельно через пул
    процессов (см. _run_grid); численные результаты не меняются."""
    sigmas = SIGMAS
    init_tag = f"_{mixture_mean_init}" if algorithm == "sgd" else ""
    tau_tag = f"_tau{tau}" if tau != 1.0 else ""

    # results[sigma][M] = list of L2 errors over NUM_SEEDS
    results = {sigma: {M: [] for M in M_VALUES} for sigma in sigmas}

    # Load reference data once (Data_7.csv: varies M, std ∈ {0.01, 0.025, 0.05}).
    ref_df = _load_reference_csv("Data_7.csv")
    save_path = f"figures/varying_M_{algorithm}{init_tag}{tau_tag}.png"

    if algorithm == "em":
        # EM is the short path (200 iters); keep it sequential.
        for sigma in sigmas:
            for M in M_VALUES:
                for seed in range(NUM_SEEDS):
                    results[sigma][M].append(_run_one_M_em(sigma, M, seed, tau=tau))
                _plot_varying_M(results, sigmas, ref_df, save_path)
        return results

    # SGD: fan the independent grid out across the visible GPU(s).
    work_items = [(sigma, M, seed, mixture_mean_init, tau)
                  for sigma in sigmas for M in M_VALUES for seed in range(NUM_SEEDS)]
    completed = {sigma: {M: 0 for M in M_VALUES} for sigma in sigmas}
    for (sigma, M, _seed, _, _), l2_err in _run_grid(_run_one_M, work_items, _max_workers()):
        results[sigma][M].append(l2_err)
        completed[sigma][M] += 1
        # Redraw whenever a point (this M, all seeds) is complete.
        if completed[sigma][M] == NUM_SEEDS:
            _plot_varying_M(results, sigmas, ref_df, save_path)
    _plot_varying_M(results, sigmas, ref_df, save_path)
    return results


def _run_one_N(sigma, target_N, seed, mixture_mean_init, tau=None):
    """Train one CatVAE for (sigma, N, seed) via SGD and return its L2 error.

    Top-level/picklable; body identical to the former inner loop of
    experiment_varying_N (same seed, same train_model args)."""
    actual_N = max(target_N, 1)
    print(f"\n{'='*40}")
    print(f"[sgd] sigma={sigma}, N={actual_N}, seed={seed}")
    print(f"{'='*40}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Fixed pool of exactly N groups (mirrors EM / reference EMML): N is the pool
    # size, independent of batch_size / num_iterations.
    x_train, y_train, _ = sample_batch_tensor(actual_N, M_FIXED_N, sigma, device)
    model = train_model(
        sigma=sigma,
        M=M_FIXED_N,
        num_iterations=SGD_NUM_ITERATIONS,
        batch_size=min(actual_N, 64),
        fixed_train_data=(x_train, y_train),
        tau=tau,
        sinkhorn_iters=50,
        eval_every=500,
        lr=1e-3,
        val_num_batches=16,
        fixed_val_batches=False,
        num_mixture_components=NUM_MIXTURE_COMPONENTS,
        mixture_mean_init=mixture_mean_init,
    )
    l2_err = compute_L2_density_error(model, sigma=sigma, E=100)
    print(f"  → [sgd] sigma={sigma}, N={actual_N}, seed={seed} L2 error: {l2_err:.4f}")
    return actual_N, l2_err


def _run_one_N_em(sigma, target_N, seed, tau=None):
    """EM counterpart of _run_one_N (kept for the sequential EM path)."""
    actual_N = max(target_N, 1)
    print(f"\n{'='*40}")
    print(f"[em] sigma={sigma}, N={actual_N}, seed={seed}")
    print(f"{'='*40}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    x_train, y_train, _ = sample_batch_tensor(actual_N, M_FIXED_N, sigma, device)
    model = train_em(
        sigma=sigma,
        M=M_FIXED_N,
        num_iterations=200,
        eval_every=50,
        K=NUM_MIXTURE_COMPONENTS,
        tau=tau,
        sinkhorn_iters=50,
        val_num_batches=16,
        fixed_val_batches=False,
        fixed_train_data=(x_train, y_train),
        seed=seed,
    )
    l2_err = compute_L2_density_error(model, sigma=sigma, E=100)
    print(f"  → [em] sigma={sigma}, N={actual_N}, seed={seed} L2 error: {l2_err:.4f}")
    return actual_N, l2_err


def experiment_varying_N(algorithm: str = "sgd", mixture_mean_init: str = "spread", tau: float = 1.0):
    """Эксперимент: влияние объёма обучающих данных на качество.
    Аналог Section 6.1.1 (Varying N) / Fig 3 из статьи.
    Параметры выровнены с референсом: M=20, std=0.05.
    N = число независимых групп (батчей).

    SGD-прогоны независимы по (sigma, N, seed) и считаются параллельно через пул
    процессов (см. _run_grid); численные результаты не меняются."""
    M = M_FIXED_N
    sigmas = SIGMAS
    init_tag = f"_{mixture_mean_init}" if algorithm == "sgd" else ""
    tau_tag = f"_tau{tau}" if tau != 1.0 else ""

    results = {sigma: {} for sigma in sigmas}

    # Load reference data once (Data_5.csv: varies N, epsi ∈ {0.001, 0.0025, 0.01},
    # std=0.05, M=20).
    ref_df = _load_reference_csv("Data_5.csv")
    save_path = f"figures/varying_N_{algorithm}{init_tag}{tau_tag}.png"

    if algorithm == "em":
        # EM is the short path (200 iters); keep it sequential.
        for sigma in sigmas:
            for target_N in TARGET_N_VALUES:
                for seed in range(NUM_SEEDS):
                    actual_N, l2_err = _run_one_N_em(sigma, target_N, seed, tau=tau)
                    results[sigma].setdefault(actual_N, []).append(l2_err)
                _plot_varying_N(results, sigmas, ref_df, M, save_path)
        return results

    # SGD: fan the independent grid out across the visible GPU(s).
    work_items = [(sigma, target_N, seed, mixture_mean_init, tau)
                  for sigma in sigmas for target_N in TARGET_N_VALUES for seed in range(NUM_SEEDS)]
    completed = {sigma: {max(n, 1): 0 for n in TARGET_N_VALUES} for sigma in sigmas}
    for (sigma, _target_N, _seed, _, _), (actual_N, l2_err) in _run_grid(_run_one_N, work_items, _max_workers()):
        results[sigma].setdefault(actual_N, []).append(l2_err)
        completed[sigma][actual_N] += 1
        # Redraw whenever a point (this N, all seeds) is complete.
        if completed[sigma][actual_N] == NUM_SEEDS:
            _plot_varying_N(results, sigmas, ref_df, M, save_path)
    _plot_varying_N(results, sigmas, ref_df, M, save_path)
    return results


def _append_csv_row(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _run_one_em_iters(
    *,
    phase,
    sigma,
    M,
    seed,
    num_iterations,
    sinkhorn_iters,
    log_root,
    l2_eval_every,
    l2_eval_grid,
    eval_every,
    val_num_batches,
):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise RuntimeError(
            "TensorBoard is not installed in the active Python environment. "
            "Rebuild the remote container with: "
            "BUILD=1 scripts/remote_run.sh 'python -c \"from torch.utils.tensorboard import SummaryWriter\"'"
        ) from exc

    print(f"\n{'='*60}")
    print(
        f"[em_iters:{phase}] sigma={sigma}, M={M}, seed={seed}, "
        f"iters={num_iterations}, sinkhorn_iters={sinkhorn_iters}"
    )
    print(f"{'='*60}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    x_train, y_train, _ = sample_batch_tensor(N_TRAIN_M, M, sigma, device)

    run_name = (
        f"{phase}/sigma_{sigma}/M_{M}/seed_{seed}/"
        f"iters_{num_iterations}/sinkhorn_{sinkhorn_iters}"
    )
    writer = SummaryWriter(log_dir=str(log_root / run_name))
    try:
        writer.add_text(
            "meta/config",
            (
                f"phase={phase}, sigma={sigma}, M={M}, seed={seed}, "
                f"num_iterations={num_iterations}, "
                f"sinkhorn_iters={sinkhorn_iters}, N_train={N_TRAIN_M}"
            ),
            0,
        )
        model = train_em(
            sigma=sigma,
            M=M,
            num_iterations=num_iterations,
            eval_every=eval_every,
            K=NUM_MIXTURE_COMPONENTS,
            sinkhorn_iters=sinkhorn_iters,
            val_num_batches=val_num_batches,
            fixed_val_batches=True,
            fixed_train_data=(x_train, y_train),
            seed=seed,
            summary_writer=writer,
            run_tag=run_name,
            l2_eval_every=l2_eval_every,
            l2_eval_grid=l2_eval_grid,
        )
    finally:
        writer.close()

    final_l2 = (
        float(model.l2_errors[-1])
        if getattr(model, "l2_errors", None)
        else float(compute_L2_density_error(model, sigma=sigma, E=l2_eval_grid))
    )
    final_pi = torch.softmax(model.log_pi, dim=-1).detach().cpu().numpy()
    final_shifts = model.shifts.detach().cpu().numpy()
    param_errors = shift_parameter_errors(final_shifts, final_pi)

    row = {
        "phase": phase,
        "sigma": sigma,
        "M": M,
        "seed": seed,
        "N_train": N_TRAIN_M,
        "num_iterations": num_iterations,
        "sinkhorn_iters": sinkhorn_iters,
        "final_L2": final_l2,
        "final_recon": float(model.train_recon[-1]) if model.train_recon else float("nan"),
        "final_recon_per_M": float(model.train_recon_per_M[-1]) if model.train_recon_per_M else float("nan"),
        "final_accuracy": float(model.train_accuracies[-1]) if model.train_accuracies else float("nan"),
        "shift_mean_abs_error": param_errors["shift_mean_abs_error"],
        "shift_max_abs_error": param_errors["shift_max_abs_error"],
        "pi_l1_error": param_errors["pi_l1_error"],
        "sinkhorn_max_row_error": float(model.sinkhorn_row_errors[-1]) if model.sinkhorn_row_errors else float("nan"),
        "sinkhorn_max_col_error": float(model.sinkhorn_col_errors[-1]) if model.sinkhorn_col_errors else float("nan"),
        "shift_0": float(final_shifts[0]) if len(final_shifts) > 0 else float("nan"),
        "shift_1": float(final_shifts[1]) if len(final_shifts) > 1 else float("nan"),
        "pi_0": float(final_pi[0]) if len(final_pi) > 0 else float("nan"),
        "pi_1": float(final_pi[1]) if len(final_pi) > 1 else float("nan"),
        "tensorboard_run": str(log_root / run_name),
    }
    print(
        f"  -> [em_iters:{phase}] sigma={sigma}, M={M}, seed={seed}, "
        f"iters={num_iterations}, sinkhorn={sinkhorn_iters}, "
        f"L2={final_l2:.4f}, shift_err={param_errors['shift_mean_abs_error']:.4f}, "
        f"pi_l1={param_errors['pi_l1_error']:.4f}"
    )
    return row


def experiment_em_iters(algorithm: str = None, mixture_mean_init: str = None):
    """EM convergence diagnostics with TensorBoard logging.

    Environment overrides for quick probes:
      EM_DIAG_M=64,83
      EM_DIAG_SIGMAS=0.01
      EM_DIAG_SEEDS=0
      EM_DIAG_NUM_ITERATIONS=200,500
      EM_DIAG_SINKHORN=50,100
      EM_DIAG_L2_EVAL_EVERY=50
    """
    del algorithm, mixture_mean_init

    sigmas = _parse_env_list("EM_DIAG_SIGMAS", SIGMAS, float)
    m_values = _parse_env_list("EM_DIAG_M", EM_ITERS_M_VALUES, int)
    seeds = _parse_env_list("EM_DIAG_SEEDS", range(NUM_SEEDS), int)
    num_iteration_values = _parse_env_list("EM_DIAG_NUM_ITERATIONS", EM_ITERS_NUM_ITERATIONS, int)
    sinkhorn_values = _parse_env_list("EM_DIAG_SINKHORN", EM_ITERS_SINKHORN_VALUES, int)
    l2_eval_every = int(os.environ.get("EM_DIAG_L2_EVAL_EVERY", "50"))
    l2_eval_grid = int(os.environ.get("EM_DIAG_L2_GRID", "100"))
    eval_every = int(os.environ.get("EM_DIAG_EVAL_EVERY", "50"))
    val_num_batches = int(os.environ.get("EM_DIAG_VAL_BATCHES", "16"))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_root = Path(os.environ.get("TB_LOGDIR", f"runs/em_iters/{timestamp}"))
    summary_path = Path(os.environ.get("EM_DIAG_SUMMARY_CSV", "figures/em_iters_summary.csv"))
    if summary_path.exists():
        summary_path.unlink()

    print(f"TensorBoard logdir: {log_root}")
    print(f"CSV summary: {summary_path}")

    rows = []
    for sigma in sigmas:
        for M in m_values:
            for seed in seeds:
                for num_iterations in num_iteration_values:
                    row = _run_one_em_iters(
                        phase="iteration_sweep",
                        sigma=sigma,
                        M=M,
                        seed=seed,
                        num_iterations=num_iterations,
                        sinkhorn_iters=50,
                        log_root=log_root,
                        l2_eval_every=l2_eval_every,
                        l2_eval_grid=l2_eval_grid,
                        eval_every=eval_every,
                        val_num_batches=val_num_batches,
                    )
                    rows.append(row)
                    _append_csv_row(summary_path, row)

    sinkhorn_num_iterations = int(os.environ.get("EM_DIAG_SINKHORN_NUM_ITERATIONS", "200"))
    for sigma in sigmas:
        for M in m_values:
            for seed in seeds:
                for sinkhorn_iters in sinkhorn_values:
                    row = _run_one_em_iters(
                        phase="sinkhorn_sweep",
                        sigma=sigma,
                        M=M,
                        seed=seed,
                        num_iterations=sinkhorn_num_iterations,
                        sinkhorn_iters=sinkhorn_iters,
                        log_root=log_root,
                        l2_eval_every=l2_eval_every,
                        l2_eval_grid=l2_eval_grid,
                        eval_every=eval_every,
                        val_num_batches=val_num_batches,
                    )
                    rows.append(row)
                    _append_csv_row(summary_path, row)

    print(f"EM diagnostics finished. Wrote {len(rows)} rows to {summary_path}")
    print(f"TensorBoard logdir: {log_root}")
    return rows


if __name__ == "__main__":
    algorithm = os.environ.get("ALGO", "sgd").lower()
    exp = os.environ.get("EXP", "sigma").lower()
    mixture_mean_init = os.environ.get("MIXTURE_INIT", "spread").lower()
    # Fixed Sinkhorn temperature (no annealing). Defaults to 1.0; override via TAU.
    tau = float(os.environ["TAU"]) if os.environ.get("TAU") else 1.0

    runners = {
        "sigma": experiment_varying_sigma,
        "m":     experiment_varying_M,
        "n":     experiment_varying_N,
        "em_iters": experiment_em_iters,
    }
    if exp == "all":
        for name in ["sigma", "m", "n"]:
            print(f"\n{'#'*60}\n# Running experiment_varying_{name} [{algorithm}, init={mixture_mean_init}, tau={tau}]\n{'#'*60}")
            runners[name](algorithm=algorithm, mixture_mean_init=mixture_mean_init, tau=tau)
    elif exp in runners:
        if exp == "em_iters":
            runners[exp](algorithm=algorithm, mixture_mean_init=mixture_mean_init)
        else:
            runners[exp](algorithm=algorithm, mixture_mean_init=mixture_mean_init, tau=tau)
    else:
        raise ValueError(f"Unknown EXP={exp!r}. Expected one of: sigma, M, N, em_iters, all.")
