"""
Muon optimizer — orthogonalized gradient descent for 2D weight matrices.
=========================================================================

What Muon is
------------
Muon (MomentUm Orthogonalized by Newton-schulz) replaces the per-element
gradient direction with an *orthogonalized* version of the Nesterov-momentum
buffer.  After accumulating momentum it runs a few Newton-Schulz iterations
on the momentum matrix to push it toward an orthogonal matrix (one with
orthonormal rows or columns), then takes the update step.

Why does that help?  In a transformer, dense weight matrices receive
gradients with highly anisotropic singular value spectra — a few directions
are huge and the rest are tiny, so SGD/Adam either underupdate the small
directions or overshoot the large ones.  Orthogonalizing the update equalizes
all directions, giving a much more well-conditioned step that empirically
converges faster on 2D layers (attention QKVO projections, FFN matrices).
The trade-off: the operation is only meaningful for ndim ≥ 2 tensors, so
1D parameters (norms, biases, embedding tables) must go to AdamW instead.

How this implementation works
-----------------------------
1. ``buf = momentum * buf + grad``  (in-place momentum accumulation).
2. The Nesterov look-ahead direction ``grad + momentum * buf`` is reshaped to
   2D as (out_features, rest) and run through ``newton_schulz`` for a few
   iterations.  The iteration converges on the matrix's polar factor (the
   closest orthogonal matrix) using only matrix-multiplies — no SVDs, all
   autograd-free, all on the device.
3. The orthogonalized update is scaled by ``sqrt(max(1, fan_out/fan_in))`` so
   the *applied* step is shape-consistent across matrices of different aspect
   ratios — a single ``lr`` is then meaningful for every layer — and
   subtracted from ``p``.

The ``MUON_NS_ITERATIONS`` constant in ``config.py`` controls how many NS
iterations we run per step; 5 is the standard count for the quintic
coefficients below (the polynomial is tuned to converge in ~5).

Routing
-------
Pass only ``ndim >= 2`` parameters to Muon.  The training loop routes
embeddings, RMSNorm scales, and any other 1D parameters to AdamW.
"""

import torch

# === Newton-Schulz coefficients ===
# The iteration
#     X_{t+1} = a·X + b·(X Xᵀ X) + c·(X Xᵀ)² X
# is a *quintic* polynomial (degree 5 in X).  These coefficients are the ones
# from Keller Jordan's Muon: the polynomial's scalar map f(s) = a·s + b·s³ +
# c·s⁵ has a steep slope at s = 0 (f'(0) = a ≈ 3.44), so even singular values
# very close to 0 are pulled toward 1 within ~5 iterations.  A plain cubic
# (1.5s − 0.5s³, f'(0) = 1.5) needs ~30 iterations under the same Frobenius
# pre-normalization — far too slow to actually orthogonalize at a small
# iteration count.
NS_COEFFS = (3.4445, -4.7750, 2.0315)


def newton_schulz(G: torch.Tensor, n_iter: int = 5) -> torch.Tensor:
    """
    Approximately orthogonalize ``G`` via the quintic Newton-Schulz iteration.

    Pushes ``G`` toward the closest matrix with orthonormal rows/columns (its
    polar factor).  This avoids the cost of a full SVD while reaching a good
    orthogonalization in a handful of matmuls.

    Iteration: ``X_{t+1} = a·X + b·(X Xᵀ X) + c·(X Xᵀ)² X``.

    Args:
        G: (m, n) gradient matrix.  Must be 2D.
        n_iter: Number of iterations (5 is standard for the quintic coeffs).

    Returns:
        X: (m, n) approximately orthogonalized matrix (singular values ≈ 1).
    """
    assert G.ndim == 2, f"Expected 2D tensor, got shape {G.shape}"
    a, b, c = NS_COEFFS

    # NS iterates run in fp32 regardless of the parameter dtype — bf16 NS is
    # unstable around the convergent region (the quintic term overflows fast).
    X = G.to(torch.float32)
    # Pre-normalize so every singular value of X starts in (0, 1], which sits
    # inside the convergent region for these coefficients.
    X = X / (X.norm() + 1e-8)

    # Work with the shorter axis on the rows so ``X @ X.T`` is the smaller
    # Gram matrix; transpose back at the end.  Mathematically identical either
    # way — this is purely a memory/flops optimization for rectangular grads.
    transpose = X.shape[0] > X.shape[1]
    if transpose:
        X = X.T
    for _ in range(n_iter):
        A = X @ X.T                # (min_dim, min_dim) Gram matrix
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transpose:
        X = X.T

    return X


class Muon(torch.optim.Optimizer):
    """
    Muon optimizer for ``ndim >= 2`` weight matrices.

    Workflow per step:
      1. ``buf = momentum * buf + grad``  (in-place).
      2. ``update = newton_schulz(grad + momentum * buf)``  (Nesterov
         look-ahead direction, orthogonalized).
      3. Scale by ``sqrt(max(1, fan_out/fan_in))`` so ``lr`` is interpretable
         independent of matrix aspect ratio.
      4. ``p -= lr * update``.

    Decoupled weight decay is applied *before* the gradient step (multiplying
    ``p`` by ``1 - lr*wd``) — the AdamW convention.  Pass it through ``params``
    config like any other optimizer.

    All parameters passed to this optimizer must have ``ndim >= 2``.  Route 1D
    parameters (RMSNorm scales, biases, embeddings) to AdamW.

    Args:
        params: Iterable of ``ndim >= 2`` parameters.
        lr: Learning rate.
        momentum: Momentum coefficient.
        ns_iterations: Number of Newton-Schulz iterations per step.
        weight_decay: Decoupled weight decay coefficient.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        ns_iterations: int = 5,
        weight_decay: float = 0.0,
    ) -> None:
        defaults = dict(lr=lr, momentum=momentum, ns_iterations=ns_iterations,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None) -> None:  # type: ignore[override]
        """
        Perform a single optimization step.

        For each parameter with a gradient:
          1. Flatten the gradient to 2D ``(out_dim, rest)``.
          2. Update the per-param momentum buffer in-place.
          3. Orthogonalize the Nesterov look-ahead direction via Newton-Schulz.
          4. Scale by the aspect-ratio factor and apply the update.

        Returns ``None`` (matches ``torch.optim.Optimizer.step``'s behaviour
        when no closure is provided).
        """
        if closure is not None:
            with torch.enable_grad():
                closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            ns_iter = group['ns_iterations']
            wd = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                # Hard guard — Muon is *only* defined for ndim >= 2.  This is
                # easier to debug than letting the reshape silently produce
                # wrong outputs for 1D params.
                assert grad.ndim >= 2, (
                    f"Muon requires ndim >= 2, got shape {p.shape}. "
                    "Route 1D parameters to AdamW."
                )

                # Decoupled weight decay (AdamW convention) — applied *before*
                # the gradient step so it acts as a pure shrinkage and doesn't
                # interact with the orthogonalized update direction.
                if wd > 0:
                    p.mul_(1.0 - lr * wd)

                orig_shape = p.shape
                # Flatten any high-rank tensor to (out, in_total).  Newton-
                # Schulz only knows how to orthogonalize 2D matrices, so we
                # collapse all "input" dims into the second axis.
                grad_2d = grad.reshape(grad.shape[0], -1)

                # Lazy momentum-buffer init on first step.
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(grad_2d)

                buf = state['momentum_buffer']
                # ``buf = momentum*buf + grad`` in place.
                buf.mul_(momentum).add_(grad_2d)
                # Nesterov look-ahead: orthogonalize ``grad + momentum*buf``
                # rather than ``buf`` itself, so the update anticipates the
                # next momentum step (matches reference Muon).
                update_input = grad_2d.add(buf, alpha=momentum)

                # Push the look-ahead direction toward an orthogonal matrix.
                update_2d = newton_schulz(update_input, n_iter=ns_iter)

                # Aspect-ratio scaling: Newton-Schulz already drives the
                # singular values to ≈1, so the update's spectral norm is ≈1
                # regardless of shape.  The sqrt(max(1, fan_out/fan_in)) factor
                # keeps the *applied* update magnitude comparable across
                # matrices of different aspect ratios, so a single ``lr`` is
                # meaningful for every layer (Keller Jordan's Muon convention).
                m_dim, n_dim = grad_2d.shape
                # For square params the factor is exactly 1.0; skip the
                # multiply (``x * 1.0 == x`` in fp32, so this is bit-identical
                # and preserves any NaN/Inf) to avoid a wasted tensor copy.
                scale = max(1.0, m_dim / n_dim) ** 0.5
                if scale != 1.0:
                    update_2d = update_2d * scale

                # Cast back to the parameter's dtype (typically fp32 on the
                # master copy — AMP keeps fp32 master weights).  The reshape
                # restores any high-rank tensor structure we collapsed earlier.
                update = update_2d.to(p.dtype).reshape(orig_shape)
                p.add_(update, alpha=-lr)
