"""
Implicit Sinkhorn differentiation (CVPR 2022, Eisenberger et al.).

Vendored from implicit-sinkhorn/sinkhorn/sinkhorn.py with two compatibility
fixes for the current stack (torch >= 2.x, flat-module project run from code/):

  1. Dropped `from utils.tools import device`; the backward zero-pad now uses
     the input tensor's own device/dtype (correct and dependency-free).
  2. Deprecated `torch.solve(t, K)` -> `torch.linalg.solve(K, t)`
     (new API solves `A X = B`; no tuple return to unpack).

The forward is mathematically identical to the original. The point of this
module is the custom `backward`: it computes the gradient at the Sinkhorn
fixed point by solving a linear system, instead of unrolling the iterations
through autograd (O(M^2) memory instead of O(num_sink * M^2)).
"""

import torch


class Sinkhorn(torch.autograd.Function):
    """
    A Sinkhorn layer with a custom backward based on implicit differentiation.

    :param c: input cost matrix, size [*, m, n] (* = arbitrary batch dims)
    :param a: first input marginal, size [*, m]
    :param b: second input marginal, size [*, n]
    :param num_sink: number of Sinkhorn iterations
    :param lambd_sink: entropy regularization weight
    :return: optimized soft permutation matrix
    """

    @staticmethod
    def forward(ctx, c, a, b, num_sink, lambd_sink):
        log_p = -c / lambd_sink
        log_a = torch.log(a).unsqueeze(dim=-1)
        log_b = torch.log(b).unsqueeze(dim=-2)
        for _ in range(num_sink):
            log_p -= (torch.logsumexp(log_p, dim=-2, keepdim=True) - log_b)
            log_p -= (torch.logsumexp(log_p, dim=-1, keepdim=True) - log_a)
        p = torch.exp(log_p)

        ctx.save_for_backward(p, torch.sum(p, dim=-1), torch.sum(p, dim=-2))
        ctx.lambd_sink = lambd_sink
        return p

    @staticmethod
    def backward(ctx, grad_p):
        p, a, b = ctx.saved_tensors

        m, n = p.shape[-2:]
        batch_shape = list(p.shape[:-2])

        # incoming grad may be a broadcast/expanded view (shared memory);
        # the in-place ops below require a private, contiguous buffer.
        grad_p = grad_p.clone()
        grad_p *= -1 / ctx.lambd_sink * p
        K = torch.cat((torch.cat((torch.diag_embed(a), p), dim=-1),
                       torch.cat((p.transpose(-2, -1), torch.diag_embed(b)), dim=-1)), dim=-2)[..., :-1, :-1]
        t = torch.cat((grad_p.sum(dim=-1), grad_p[..., :, :-1].sum(dim=-2)), dim=-1).unsqueeze(-1)
        # Early in training (and for small M) the implicit system K can be
        # singular when Sinkhorn produces a near-deterministic p. A tiny
        # ridge keeps the solve well-posed; the bias is negligible since K
        # entries are O(1). (Mitigation planned for the singular-K risk.)
        eye = torch.eye(K.shape[-1], device=K.device, dtype=K.dtype)
        K = K + 1e-6 * eye
        grad_ab = torch.linalg.solve(K, t)
        grad_a = grad_ab[..., :m, :]
        grad_b = torch.cat((grad_ab[..., m:, :],
                            torch.zeros(batch_shape + [1, 1], device=grad_p.device, dtype=grad_p.dtype)), dim=-2)
        U = grad_a + grad_b.transpose(-2, -1)
        grad_p -= p * U
        grad_a = -ctx.lambd_sink * grad_a.squeeze(dim=-1)
        grad_b = -ctx.lambd_sink * grad_b.squeeze(dim=-1)
        return grad_p, grad_a, grad_b, None, None, None
