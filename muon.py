"""Muon: Nesterov momentum orthogonalized by Newton-Schulz.

ndim >= 2 only — 1D parameters (norms, biases, embeddings) go to AdamW.
``config.MUON_NS_ITERATIONS`` sets the iteration count.
"""

import torch

# Keller Jordan's quintic X <- a·X + b·(X Xᵀ)X + c·(X Xᵀ)²X. f'(0) = 3.44 pulls
# near-zero singular values to 1 in ~5 iterations; a cubic (f'(0) = 1.5) needs ~30.
NS_COEFFS = (3.4445, -4.7750, 2.0315)


def newton_schulz(G: torch.Tensor, n_iter: int = 5) -> torch.Tensor:
    """(m, n) -> its polar factor, singular values ≈ 1. Matmuls only, no SVD."""
    assert G.ndim == 2, f"Expected 2D tensor, got shape {G.shape}"
    a, b, c = NS_COEFFS

    # fp32 whatever p.dtype is: bf16 overflows in the quintic term.
    X = G.to(torch.float32)
    # Every singular value into (0, 1] — the convergent region for these coefficients.
    X = X / (X.norm() + 1e-8)

    # Gram on the shorter axis: same result, fewer flops.
    transpose = X.shape[0] > X.shape[1]
    if transpose:
        X = X.T
    for _ in range(n_iter):
        A = X @ X.T                # (min, min) Gram
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transpose:
        X = X.T

    return X


class Muon(torch.optim.Optimizer):
    """Muon for ``ndim >= 2`` matrices; route 1D parameters to AdamW.

    Weight decay is decoupled and applied BEFORE the gradient step (AdamW convention).
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
                assert grad.ndim >= 2, (
                    f"Muon requires ndim >= 2, got shape {p.shape}. "
                    "Route 1D parameters to AdamW."
                )

                if wd > 0:
                    p.mul_(1.0 - lr * wd)

                orig_shape = p.shape
                grad_2d = grad.reshape(grad.shape[0], -1)

                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(grad_2d)

                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(grad_2d)
                nesterov_dir = grad_2d.add(buf, alpha=momentum)

                update_2d = newton_schulz(nesterov_dir, n_iter=ns_iter)

                # Newton-Schulz leaves the spectral norm at ≈1 whatever the shape, so
                # sqrt(max(1, fan_out/fan_in)) is what makes one ``lr`` mean the same
                # applied step across aspect ratios.
                m_dim, n_dim = grad_2d.shape
                # Exactly 1.0 when square; skipping is bit-identical and preserves NaN/Inf.
                scale = max(1.0, m_dim / n_dim) ** 0.5
                if scale != 1.0:
                    update_2d = update_2d * scale

                update = update_2d.to(p.dtype).reshape(orig_shape)
                p.add_(update, alpha=-lr)
