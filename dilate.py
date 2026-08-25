"""DILATE — shape + time distortion loss (Le Guen & Thome, NeurIPS 2019), risk space.

shape is the soft-DTW DIVERGENCE ``sDTW(m,y) - ½sDTW(m,m) - ½sDTW(y,y)`` (Blondel et al.
2021): plain soft-DTW is not zero at ``m == y``, since the soft-min still pays an entropic
price. time is TDI ``= <A, Ω>``, ``A = ∂sDTW/∂C``, evaluated as the directional derivative
of the soft-DTW value along ``Ω`` by one extra forward rather than by materialising ``A``.

The DP is swept along anti-diagonals — ``2H-1`` of them, every cell on one independent —
with the whole batch and the whole diagonal vectorised, never a per-sample Python loop.
Both sweeps are dispatch-bound, so the measured cost tracks ``H`` and ignores the batch
size; on CUDA fp32 a Triton kernel sinks the diagonal loop into one launch, and the eager
loop is the definition of record everywhere else (CPU, fp64, no Triton). The two agree to
fp32 rounding — ``tests/test_dilate.py``.

Cost is the squared difference in risk space; one cell peaks at
``(f(BG_CLAMP_MAX) - f(BG_CLAMP_MIN))**2 = 99.6416``. A non-finite median or cost is NOT
asserted away — it propagates into the returned loss so train.py's isfinite / EMA-restore
guard handles it.
"""

import torch

from config import DILATE_ALPHA, DILATE_GAMMA

# TDI finite-difference step; the fallback keeps a config predating the constant loadable.
try:
    from config import DILATE_TDI_FD_EPS
except ImportError:
    DILATE_TDI_FD_EPS = 0.05

# Triton carries the DP on CUDA fp32 only; absence is not an error, the eager
# reference below is complete on its own.
try:
    import triton
    import triton.language as tl
    _HAVE_TRITON = True
except ImportError:                                            # pragma: no cover
    _HAVE_TRITON = False

# One program holds a whole anti-diagonal in one warp, so H is bounded by the largest
# block Triton maps to it. Above this the reference runs; no caller comes close
# (H = span length x PATCH_SIZE).
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

    The eager reference is the definition of record and the only path the fp64
    ``gradcheck`` exercises; both must agree (``test_softdtw_triton_matches_reference``).
    ``H > 0`` is correctness, not an optimisation: the reference returns ``R[0, 0] = 0``
    while a ``H = 0`` launch asks for a zero-width block and fails to compile.
    """
    return (_HAVE_TRITON and cost.is_cuda and cost.dtype == torch.float32
            and 0 < cost.shape[1] <= _TRITON_MAX_H)


if _HAVE_TRITON:
    @triton.jit
    def _softdtw_forward_kernel(cost_ptr, r_ptr, gamma, H, BLOCK: tl.constexpr):
        """One program per batch row; the whole anti-diagonal in one warp, sweep = one launch.

        ``R`` is read and written through L2 (``.cg``), not L1, so the previous diagonal a
        later iteration reads is the one this iteration stored; ``num_warps=1`` at the call
        site keeps the diagonal inside one warp and ``tl.debug_barrier`` orders the accesses.
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
            # The +inf boundary divides to -inf, exp 0; every reachable cell has a
            # finite predecessor, so z is never -inf.
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

        The reference's ``D``/``Rp`` padding is a masked load here — off-grid ``Rp`` reads
        -inf and off-grid ``D`` reads 0, which is what those rings held — so this pass
        allocates ``E`` alone where the reference allocates three tables of that size.
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

    ``R[i, j] = C[i, j] + softmin_γ(R[i-1, j], R[i-1, j-1], R[i, j-1])`` over an
    ``(H+1, H+1)`` table per sample; the value is ``R[H, H]``. Backward runs the dual
    recursion for the alignment soft-assignment ``E`` and pushes it onto the cost:
    ``∂loss/∂C = grad · E``. Neither pass builds an autograd graph over the ``2H-1`` steps.
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

        # Padded by one on each axis: R[:, 0, :] and R[:, :, 0] are the +inf boundary
        # (no path enters from off-grid), R[:, 0, 0] = 0 is the origin.
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

        # A non-finite value is deliberately NOT asserted away: it propagates into the
        # loss so train.py's isfinite / _maybe_restore_from_ema guard handles it.
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
            # D and Rp below are pure padding; the kernel reads their rings as masked-load
            # defaults, so it allocates E alone.
            _softdtw_backward_kernel[(B,)](
                cost, R, E, float(gamma), H,
                BLOCK=triton.next_power_of_2(H), num_warps=1)
        else:
            # Padded so cost-cell indexing inside the E-recursion is uniform.
            D = torch.zeros((B, H + 2, H + 2), device=dev, dtype=dt)
            D[:, 1:H + 1, 1:H + 1] = cost

            # Outer ring -inf so an off-grid neighbour never wins the soft-assignment;
            # the terminal cell seeds the recursion.
            Rp = torch.full((B, H + 2, H + 2), -float("inf"), device=dev, dtype=dt)
            Rp[:, 0:H + 1, 0:H + 1] = R
            Rp[:, H + 1, H + 1] = R[:, H, H]

            # Each cell's soft-assignment is the sum of its three successors' assignments
            # times the local soft-min derivatives a/b/c.
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

    shape is the soft-DTW DIVERGENCE, so it and its gradient vanish at ``m == y``; the
    ``y``-self term is a forecast constant and is detached. tdi is ``<A, Ω>`` as a one-sided
    finite difference of the soft-DTW value along ``Ω``, never by materialising ``A``.

    NOT scale-free in ``H``: shape grows with the horizon while the normalised ``Ω`` does
    not, so ``alpha`` weights a different mixture at each ``H`` — a caller combining calls
    at different ``H`` must weight and log them per bucket.

    gamma is a softness knob (smaller ⇒ harder min), overflow-free in fp32 to 1e-3;
    soft-DTW is 1-homogeneous in ``(cost, gamma)``. Smaller ``tdi_fd_eps`` trades ``O(ε)``
    bias for fp headroom in the value difference.
    """
    assert m.dim() == 2 and y_risk.dim() == 2, "m and y_risk must be (B, H)"
    assert m.shape == y_risk.shape, "m and y_risk must share shape (B, H)"
    # (0, H) would mean to NaN with no exception, and that NaN passes ``val_total <
    # best_val_loss`` (False for NaN against inf), ending the run with no best checkpoint.
    assert m.shape[0] > 0, (
        "dilate_loss got an EMPTY batch (0 rows): the mean over the batch axis "
        "would be NaN. Skip empty span-length buckets in the caller."
    )
    assert m.shape[1] > 0, "dilate_loss got a zero-length horizon (B, 0)"
    assert 0.0 <= alpha <= 1.0, "alpha must be in [0, 1]"
    assert gamma > 0.0, "gamma must be positive"
    assert tdi_fd_eps > 0.0, "tdi_fd_eps must be positive"
    # PROMOTE fp16/bf16 only, never downcast: a hard ``.float()`` truncates an fp64 input
    # and breaks fp64 ``gradcheck``, whose numerator would be dominated by fp32 rounding.
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

    # One extra forward, reusing sdtw_my, approximates <A, Ω> to O(tdi_fd_eps); autograd
    # then carries the exact TDI gradient through SoftDTWBatch's first-order backward.
    omega = _omega_distance(H, m.device, m.dtype)             # (H, H)
    sdtw_my_eps = SoftDTWBatch.apply(
        cost_my + tdi_fd_eps * omega.unsqueeze(0), gamma)      # (B,)
    tdi_per = (sdtw_my_eps - sdtw_my) / tdi_fd_eps             # (B,)
    tdi = tdi_per.mean()

    loss = alpha * shape + (1.0 - alpha) * tdi
    return loss, shape, tdi
