"""DILATE — shape + time distortion loss (Le Guen & Thome, NeurIPS 2019), risk space.

shape is soft-DTW DIVERGENCE; tdi is ``<A, Ω>`` via a directional derivative, not by
materialising ``A``. Non-finite median/cost propagates, not asserted away (train.py's guard).
"""

import torch

from config import DILATE_ALPHA, DILATE_GAMMA

# TDI finite-difference step; the fallback keeps a config predating the constant loadable.
try:
    from config import DILATE_TDI_FD_EPS
except ImportError:
    DILATE_TDI_FD_EPS = 0.05

# Triton carries the DP on CUDA fp32 only; absent, the eager reference is complete alone.
try:
    import triton
    import triton.language as tl
    _HAVE_TRITON = True
except ImportError:                                            # pragma: no cover
    _HAVE_TRITON = False

# One warp holds a whole anti-diagonal, bounding H; above it the reference runs instead.
_TRITON_MAX_H = 1024


def _pairwise_sq_cost(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """(B, H) x (B, H) -> (B, H, H) with ``C[b,i,j] = (x[b,i] - y[b,j])**2``, risk space."""
    assert x.dim() == 2 and y.dim() == 2, "series must be (B, H)"
    assert x.shape == y.shape, "x and y must share shape (B, H)"
    diff = x.unsqueeze(2) - y.unsqueeze(1)  # (B, H, H)
    return diff * diff


def _softmin(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, gamma: float) -> torch.Tensor:
    """``-γ·logsumexp(-{a,b,c}/γ)`` elementwise over three (B, K) predecessors.

    Max-subtraction holds the exponentials in fp32 range at the single-cell risk-space
    cost peak 99.6416; overflow-free down to ``gamma = 1e-3``.
    """
    stacked = torch.stack((a, b, c), dim=0) / -gamma           # (3, B, K)
    z, _ = stacked.max(dim=0, keepdim=True)                     # (1, B, K)
    out = -gamma * (z.squeeze(0) + (stacked - z).exp().sum(dim=0).log())
    return out


def _use_triton(cost: torch.Tensor) -> bool:
    """Whether the Triton DP may carry this call — fp32 CUDA and nothing else.

    ``H > 0`` is correctness, not an optimisation: a ``H = 0`` launch asks for a zero-width
    block and fails to compile. Eager and Triton must agree (test_softdtw_triton_matches_reference).
    """
    return (_HAVE_TRITON and cost.is_cuda and cost.dtype == torch.float32
            and 0 < cost.shape[1] <= _TRITON_MAX_H)


if _HAVE_TRITON:
    @triton.jit
    def _softdtw_forward_kernel(cost_ptr, r_ptr, gamma, H, BLOCK: tl.constexpr):
        """One program per batch row; the whole anti-diagonal in one warp, sweep = one launch.

        ``R`` reads/writes through L2 (``.cg``), not L1; ``num_warps=1`` keeps the diagonal
        inside one warp and ``tl.debug_barrier`` orders the accesses across iterations.
        """
        b = tl.program_id(0)
        HR = H + 1
        cost_base = b * H * H
        r_base = b * HR * HR
        offs = tl.arange(0, BLOCK)
        neg_gamma = -gamma
        for k in range(2, 2 * H + 1):
            i = tl.maximum(1, k - H) + offs
            live = i <= tl.minimum(H, k - 1)
            j = k - i
            here = r_base + i * HR + j
            up = tl.load(r_ptr + here - HR, mask=live, other=0.0, cache_modifier=".cg")
            diag = tl.load(r_ptr + here - HR - 1, mask=live, other=0.0, cache_modifier=".cg")
            left = tl.load(r_ptr + here - 1, mask=live, other=0.0, cache_modifier=".cg")
            c = tl.load(cost_ptr + cost_base + (i - 1) * H + (j - 1), mask=live, other=0.0)
            # +inf boundary divides to -inf, exp 0; every reachable cell has a finite predecessor.
            sa = up / neg_gamma
            sb = diag / neg_gamma
            sc = left / neg_gamma
            z = tl.maximum(sa, tl.maximum(sb, sc))
            s = tl.exp(sa - z) + tl.exp(sb - z) + tl.exp(sc - z)
            tl.store(r_ptr + here, c + neg_gamma * (z + tl.log(s)),
                     mask=live, cache_modifier=".cg")
            tl.debug_barrier()

    @triton.jit
    def _softdtw_backward_kernel(cost_ptr, r_ptr, e_ptr, gamma, H, BLOCK: tl.constexpr):
        """Reverse sweep for the alignment soft-assignment ``E``.

        Off-grid ``Rp`` reads -inf and off-grid ``D`` reads 0 via masked loads, matching the
        reference's padding rings — so this pass allocates only ``E``, not three padded tables.
        """
        b = tl.program_id(0)
        HR = H + 1
        HE = H + 2
        cost_base = b * H * H
        r_base = b * HR * HR
        e_base = b * HE * HE
        offs = tl.arange(0, BLOCK)
        NEG = float("-inf")
        # Rp[H+1, H+1] — the terminal cell the recursion is seeded from.
        r_terminal = tl.load(r_ptr + r_base + H * HR + H)
        for k in range(2 * H, 1, -1):
            i = tl.maximum(1, k - H) + offs
            live = i <= tl.minimum(H, k - 1)
            j = k - i
            r_here = tl.load(r_ptr + r_base + i * HR + j, mask=live, other=0.0)

            up_live = live & (i + 1 <= H)
            r_up = tl.load(r_ptr + r_base + (i + 1) * HR + j, mask=up_live, other=NEG)
            d_up = tl.load(cost_ptr + cost_base + i * H + (j - 1), mask=up_live, other=0.0)
            a = tl.exp((r_up - r_here - d_up) / gamma)

            dg_live = live & (i + 1 <= H) & (j + 1 <= H)
            r_dg = tl.load(r_ptr + r_base + (i + 1) * HR + (j + 1), mask=dg_live, other=NEG)
            d_dg = tl.load(cost_ptr + cost_base + i * H + j, mask=dg_live, other=0.0)
            r_dg = tl.where(live & (i == H) & (j == H), r_terminal, r_dg)
            bb = tl.exp((r_dg - r_here - d_dg) / gamma)

            lf_live = live & (j + 1 <= H)
            r_lf = tl.load(r_ptr + r_base + i * HR + (j + 1), mask=lf_live, other=NEG)
            d_lf = tl.load(cost_ptr + cost_base + (i - 1) * H + j, mask=lf_live, other=0.0)
            cc = tl.exp((r_lf - r_here - d_lf) / gamma)

            here = e_base + i * HE + j
            e_up = tl.load(e_ptr + here + HE, mask=live, other=0.0, cache_modifier=".cg")
            e_dg = tl.load(e_ptr + here + HE + 1, mask=live, other=0.0, cache_modifier=".cg")
            e_lf = tl.load(e_ptr + here + 1, mask=live, other=0.0, cache_modifier=".cg")
            tl.store(e_ptr + here, e_up * a + e_dg * bb + e_lf * cc,
                     mask=live, cache_modifier=".cg")
            tl.debug_barrier()


class SoftDTWBatch(torch.autograd.Function):
    """Batched soft-DTW value + gradient, vectorised anti-diagonal DP.

    ``R[i,j] = C[i,j] + softmin_γ(R[i-1,j], R[i-1,j-1], R[i,j-1])``, value is ``R[H,H]``.
    Backward pushes dual recursion ``E`` onto the cost: ``∂loss/∂C = grad·E``, no autograd graph.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        cost: torch.Tensor,
        gamma: float,
    ) -> torch.Tensor:
        """cost ``(B, H, H)`` non-negative, gamma > 0 -> ``(B,)`` soft-DTW value."""
        assert cost.dim() == 3, "cost must be (B, H, H)"
        B, H, W = cost.shape
        assert H == W, "soft-DTW expects a square (H, H) cost matrix"
        dev, dt = cost.device, cost.dtype

        # Padded by one axis: R[:,0,:]/R[:,:,0] are +inf boundary (no off-grid path); R[:,0,0]=0.
        R = torch.full((B, H + 1, H + 1), float("inf"), device=dev, dtype=dt)
        R[:, 0, 0] = 0.0

        if _use_triton(cost):
            cost = cost.contiguous()
            _softdtw_forward_kernel[(B,)](
                cost, R, float(gamma), H,
                BLOCK=triton.next_power_of_2(H), num_warps=1)
        else:
            for k in range(2, 2 * H + 1):
                i_lo = max(1, k - H)
                i_hi = min(H, k - 1)
                if i_lo > i_hi:
                    continue
                i = torch.arange(i_lo, i_hi + 1, device=dev)       # (K,)
                j = k - i                                          # (K,)
                r_up = R[:, i - 1, j]                              # (B, K)
                r_diag = R[:, i - 1, j - 1]                        # (B, K)
                r_left = R[:, i, j - 1]                            # (B, K)
                c = cost[:, i - 1, j - 1]                          # (B, K)
                R[:, i, j] = c + _softmin(r_up, r_diag, r_left, gamma)

        # Non-finite value not asserted away: the isfinite/EMA-restore guard catches it downstream.
        value = R[:, H, H].clone()                             # (B,)

        ctx.save_for_backward(cost, R)
        ctx.gamma = gamma  # type: ignore[attr-defined]
        ctx.H = H          # type: ignore[attr-defined]
        return value

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        grad_value: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        """grad_value ``(B,)`` -> ``(grad_cost (B, H, H), None)``; None for ``gamma``."""
        cost, R = ctx.saved_tensors  # type: ignore[attr-defined]
        gamma: float = ctx.gamma     # type: ignore[attr-defined]
        H: int = ctx.H               # type: ignore[attr-defined]
        B = cost.shape[0]
        dev, dt = cost.device, cost.dtype

        E = torch.zeros((B, H + 2, H + 2), device=dev, dtype=dt)
        E[:, H + 1, H + 1] = 1.0

        if _use_triton(cost):
            # D and Rp below are pure padding, masked-load defaults; kernel allocates E alone.
            _softdtw_backward_kernel[(B,)](
                cost, R, E, float(gamma), H,
                BLOCK=triton.next_power_of_2(H), num_warps=1)
        else:
            # Padded so cost-cell indexing inside the E-recursion is uniform.
            D = torch.zeros((B, H + 2, H + 2), device=dev, dtype=dt)
            D[:, 1:H + 1, 1:H + 1] = cost

            # Outer ring -inf so an off-grid neighbour never wins; terminal cell seeds recursion.
            Rp = torch.full((B, H + 2, H + 2), -float("inf"), device=dev, dtype=dt)
            Rp[:, 0:H + 1, 0:H + 1] = R
            Rp[:, H + 1, H + 1] = R[:, H, H]

            # Each cell's soft-assignment sums its three successors' assignments times a/b/c.
            for k in range(2 * H, 1, -1):
                i_lo = max(1, k - H)
                i_hi = min(H, k - 1)
                if i_lo > i_hi:
                    continue
                i = torch.arange(i_lo, i_hi + 1, device=dev)       # (K,)
                j = k - i                                          # (K,)

                # up
                a = ((Rp[:, i + 1, j] - Rp[:, i, j] - D[:, i + 1, j]) / gamma).exp()
                # diag
                b = ((Rp[:, i + 1, j + 1] - Rp[:, i, j] - D[:, i + 1, j + 1]) / gamma).exp()
                # left
                c = ((Rp[:, i, j + 1] - Rp[:, i, j] - D[:, i, j + 1]) / gamma).exp()

                E[:, i, j] = (
                    E[:, i + 1, j] * a
                    + E[:, i + 1, j + 1] * b
                    + E[:, i, j + 1] * c
                )

        e = E[:, 1:H + 1, 1:H + 1]                             # (B, H, H)
        # Non-finite E flows into the gradient; train.py's guard catches the bad step.
        grad_cost = grad_value.view(B, 1, 1) * e
        return grad_cost, None


def _omega_distance(H: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """(H, H) ``Ω[i, j] = ((i - j) / H)**2`` — normalised squared deviation from the diagonal."""
    idx = torch.arange(H, device=device, dtype=dtype)
    d = (idx.unsqueeze(1) - idx.unsqueeze(0)) / H
    return d * d


def dilate_loss(
    m: torch.Tensor,
    y_risk: torch.Tensor,
    alpha: float = DILATE_ALPHA,
    gamma: float = DILATE_GAMMA,
    tdi_fd_eps: float = DILATE_TDI_FD_EPS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``L = α·shape + (1-α)·tdi`` in risk space. m, y_risk ``(B, H)``; returns three scalars.

    NOT scale-free in ``H``: shape grows with the horizon, so ``alpha`` weights a different
    mixture at each ``H`` — combine calls at different ``H`` weighted and logged per bucket.
    """
    assert m.dim() == 2 and y_risk.dim() == 2, "m and y_risk must be (B, H)"
    assert m.shape == y_risk.shape, "m and y_risk must share shape (B, H)"
    # (0, H) means NaN with no exception; NaN < best_val_loss is False, so no checkpoint ever saves.
    assert m.shape[0] > 0, (
        "dilate_loss got an EMPTY batch (0 rows): the mean over the batch axis "
        "would be NaN. Skip empty span-length buckets in the caller."
    )
    assert m.shape[1] > 0, "dilate_loss got a zero-length horizon (B, 0)"
    assert 0.0 <= alpha <= 1.0, "alpha must be in [0, 1]"
    assert gamma > 0.0, "gamma must be positive"
    assert tdi_fd_eps > 0.0, "tdi_fd_eps must be positive"
    # PROMOTE fp16/bf16 only, never downcast: .float() would truncate fp64 and break gradcheck.
    if m.dtype in (torch.float16, torch.bfloat16):
        m = m.float()
    if y_risk.dtype in (torch.float16, torch.bfloat16):
        y_risk = y_risk.float()
    B, H = m.shape

    cost_my = _pairwise_sq_cost(m, y_risk)
    sdtw_my = SoftDTWBatch.apply(cost_my, gamma)               # (B,)

    cost_mm = _pairwise_sq_cost(m, m)
    sdtw_mm = SoftDTWBatch.apply(cost_mm, gamma)               # (B,)

    # A forecast constant, so detached.
    with torch.no_grad():
        cost_yy = _pairwise_sq_cost(y_risk, y_risk)
        sdtw_yy = SoftDTWBatch.apply(cost_yy, gamma)           # (B,)

    shape_per = sdtw_my - 0.5 * sdtw_mm - 0.5 * sdtw_yy        # (B,)
    shape = shape_per.mean()

    # One extra forward approximates <A,Ω> to O(eps); autograd carries the exact TDI gradient.
    omega = _omega_distance(H, m.device, m.dtype)             # (H, H)
    sdtw_my_eps = SoftDTWBatch.apply(
        cost_my + tdi_fd_eps * omega.unsqueeze(0), gamma)      # (B,)
    tdi_per = (sdtw_my_eps - sdtw_my) / tdi_fd_eps             # (B,)
    tdi = tdi_per.mean()

    loss = alpha * shape + (1.0 - alpha) * tdi
    return loss, shape, tdi
