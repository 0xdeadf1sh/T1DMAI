"""
DILATE loss — DIstortion Loss with shApe and TimE (Le Guen & Thome, NeurIPS 2019).
====================================================================================

The headline forecast in the risk-space redesign is the per-step ``median``
trajectory (risk space, see ``risk_loss.py``).  Pinball loss anchors its
*level* pointwise; DILATE supplies the complementary *warp-invariant* shape
objective so a forecast that predicts the right excursion a step early/late is
not punished as if it predicted the wrong excursion entirely.

DILATE has two halves:

* **shape** — a soft-DTW alignment cost between forecast and target.  We use the
  *divergence* form ``sDTW(m, y) - ½ sDTW(m, m) - ½ sDTW(y, y)`` (Blondel et al.
  2021): plain soft-DTW is not zero at ``m == y`` because the soft-min over
  alignment paths still pays an entropic price, so the self-terms recentre it to
  a proper divergence (``shape ≈ 0`` and ``∂shape ≈ 0`` at ``m == y``).  The
  ``y``-self term is a constant in the forecast and is detached.
* **time** (TDI, Temporal Distortion Index) — the expected deviation of the
  soft-DTW optimal alignment path from the diagonal, penalising temporal
  misalignment that the shape term alone is indifferent to.  With the
  off-diagonal distance ``Ω[i,j] = ((i-j)/H)**2``, TDI ``= <A, Ω>`` where
  ``A = ∂sDTW/∂C`` is the soft alignment.  Because ``A`` is the cost-gradient of
  the soft-DTW value, ``<A, Ω>`` is exactly the *directional derivative* of that
  value along ``Ω`` in cost space, so we evaluate it by a one-sided finite
  difference ``[sDTW(C + ε·Ω) - sDTW(C)] / ε`` — one extra ``SoftDTWBatch``
  forward, reusing the already-computed ``sDTW(C)``.  Ordinary autograd then
  differentiates the surrogate through ``SoftDTWBatch``'s exact first-order
  backward, so the median gradient is the exact TDI gradient to ``O(ε)`` — at a
  fraction of the cost of materialising the alignment matrix and
  second-order-differentiating through it.

The shape divergence runs three batched soft-DTW dynamic programs (the cross
term and two self terms) and TDI one more.  The recursion has a sequential
dependency along anti-diagonals, but every cell *on* a given anti-diagonal is
independent, so we sweep diagonals (``2·H - 1`` of them) with the whole batch
and the whole diagonal vectorised — never a per-sample Python loop.  The caller
(``risk_loss.py``) buckets masked spans by length and issues one call per bucket,
so the batch axis is that bucket's span count and the horizon ``H`` is its span
length times ``PATCH_SIZE`` — neither is fixed across calls, and an empty bucket
must not be dispatched at all (see :func:`dilate_loss`).
``SoftDTWBatch``'s forward and backward are hand-written so
the DP runs once and the gradient is the standard soft-DTW backward recursion
(no autograd graph over the ``2·H-1`` diagonal steps).

Both sweeps are bound by dispatch, not arithmetic: a diagonal is a handful of
elementwise ops on a few hundred values, and there are ``2·H-1`` of them, so the
measured cost tracks ``H`` and ignores the batch size.  On CUDA fp32 the loop
therefore runs as a Triton kernel — one program per batch row, the diagonals
swept inside it, the whole sweep one launch.  The eager loop stays as the
definition of the recursion and runs everywhere else (CPU, fp64, no Triton);
the two agree to fp32 rounding, which ``tests/test_dilate.py`` asserts.

Everything is fp32; the soft-min is max-subtraction-stabilised.  Cost is the
squared difference in risk space — a single cell peaks at
``(f(BG_CLAMP_MAX) - f(BG_CLAMP_MIN))**2 = 99.6416`` (with the default
``BG_CLAMP_MAX = 400`` / ``BG_CLAMP_MIN = 10``), and the max-subtracted softmin
is overflow-free in fp32 down to
``gamma = 1e-3``.  ``gamma`` is purely a softness knob (smaller ⇒ harder min);
soft-DTW is 1-homogeneous in ``(cost, gamma)``, so scaling both by the same
factor scales the value by it and leaves the alignment unchanged.  A non-finite
median/cost is NOT asserted away here — it is allowed to propagate into the
returned loss so train.py's isfinite / EMA-restore resilience guard handles it.
"""

import torch

from config import DILATE_ALPHA, DILATE_GAMMA

# Finite-difference step for the TDI directional-derivative surrogate. Imported
# defensively so a config predating the constant still loads, falling back to
# the same default.
try:
    from config import DILATE_TDI_FD_EPS
except ImportError:
    DILATE_TDI_FD_EPS = 0.05

# Triton carries the DP on CUDA fp32; the eager reference below carries every
# other case.  Absence is not an error — the reference is complete on its own.
try:
    import triton
    import triton.language as tl
    _HAVE_TRITON = True
except ImportError:                                            # pragma: no cover
    _HAVE_TRITON = False

# One Triton program holds a whole anti-diagonal in a single warp, so H is
# bounded by the largest block Triton will map to it.  Above this the reference
# runs; no caller comes close (H = span length x PATCH_SIZE).
_TRITON_MAX_H = 1024


def _pairwise_sq_cost(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Squared-difference cost matrix between two batched 1-D series (risk space).

    Args:
        x: (B, H) risk-space series.
        y: (B, H) risk-space series.

    Returns:
        (B, H, H) cost where ``C[b, i, j] = (x[b, i] - y[b, j])**2``.
    """
    assert x.dim() == 2 and y.dim() == 2, "series must be (B, H)"
    assert x.shape == y.shape, "x and y must share shape (B, H)"
    diff = x.unsqueeze(2) - y.unsqueeze(1)  # (B, H, H)
    return diff * diff


def _softmin(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    Stabilised soft-minimum ``-γ·logsumexp(-{a,b,c}/γ)`` over three predecessors.

    The max-subtraction (here a *min*-subtraction on the pre-divided terms)
    keeps the exponentials in fp32 range even when the risk-space costs peak
    near a single-cell maximum of
    ``(f(BG_CLAMP_MAX) - f(BG_CLAMP_MIN))**2 = 99.6416``; this softmin
    is overflow-free in fp32 down to ``gamma = 1e-3``.

    Args:
        a, b, c: (B, K) predecessor accumulated costs on one anti-diagonal.
        gamma:   soft-min temperature (> 0).

    Returns:
        (B, K) soft-minimum of the three inputs, element-wise.
    """
    stacked = torch.stack((a, b, c), dim=0) / -gamma           # (3, B, K)
    z, _ = stacked.max(dim=0, keepdim=True)                     # (1, B, K)
    out = -gamma * (z.squeeze(0) + (stacked - z).exp().sum(dim=0).log())
    return out


def _use_triton(cost: torch.Tensor) -> bool:
    """Whether the Triton DP may carry this call.

    The reference DP below is the definition of record and the only path the
    fp64 ``gradcheck`` in ``tests/test_dilate.py`` exercises; Triton is an
    fp32 CUDA accelerator for it and nothing else.  Both must agree — see
    ``test_softdtw_triton_matches_reference``.

    ``H > 0`` is part of that agreement, not an optimisation: the reference
    walks an empty diagonal range and returns ``R[0, 0] = 0``, while a Triton
    launch at ``H = 0`` asks for a zero-width block and fails to compile.
    ``dilate_loss`` rejects a zero-length horizon before either sees it, so this
    only governs a direct ``SoftDTWBatch.apply``.
    """
    return (_HAVE_TRITON and cost.is_cuda and cost.dtype == torch.float32
            and 0 < cost.shape[1] <= _TRITON_MAX_H)


if _HAVE_TRITON:
    @triton.jit
    def _softdtw_forward_kernel(cost_ptr, r_ptr, gamma, H, BLOCK: tl.constexpr):
        """One program per batch row; the whole anti-diagonal in one warp.

        The eager DP pays ~16 ATen dispatches per anti-diagonal and 2H-1
        diagonals per sweep, so it is bound by Python and launch overhead, not
        arithmetic — the measured cost tracks H and ignores the batch size
        entirely.  Sinking the diagonal loop INTO the kernel is what removes
        that: the sweep becomes one launch.

        ``R`` is read and written through L2 (``.cg``) rather than L1 so the
        previous diagonal a later iteration reads is the one this iteration
        stored; ``num_warps=1`` at the call site keeps the whole diagonal
        inside a single warp, and ``tl.debug_barrier`` orders the accesses.
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
            # The +inf boundary divides to -inf, whose exponential is 0; every
            # reachable cell has a finite predecessor, so z is never -inf.
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

        The reference pads ``cost`` into ``D`` and ``R`` into ``Rp`` to make the
        neighbour reads uniform; here the padding is a masked load instead — an
        off-grid ``Rp`` reads -inf and an off-grid ``D`` reads 0, which is what
        those rings held.  So this pass allocates ``E`` alone where the
        reference allocates three tables of that size.
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
    """
    Batched soft-DTW value + gradient via vectorised anti-diagonal DP.

    The standard soft-DTW recursion fills an ``(H+1, H+1)`` cost-to-go table
    ``R`` per sample::

        R[i, j] = C[i, j] + softmin_γ(R[i-1, j], R[i-1, j-1], R[i, j-1])

    Cells on a fixed anti-diagonal ``i + j = k`` depend only on diagonals
    ``k-1`` and ``k-2``, and are mutually independent, so we advance ``k`` from
    ``2`` to ``2H`` with the entire batch and the whole diagonal as a single
    vectorised ``_softmin``.  The soft-DTW value is ``R[H, H]``.

    Backward runs the dual recursion for the alignment soft-assignment ``E``
    (``∂loss/∂R``) and pushes it onto the cost matrix: ``∂loss/∂C = grad · E``.
    Both passes are diagonal-vectorised; neither builds an autograd graph over
    the ``2H-1`` steps.

    Each sweep runs one of two ways, over the same recursion and to the same
    ``R`` and ``E``: a Triton kernel on CUDA fp32 (:func:`_use_triton`), the
    eager loop everywhere else — CPU, fp64, or no Triton installed.  The eager
    loop is the definition, and the one the fp64 ``gradcheck`` reads.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        cost: torch.Tensor,
        gamma: float,
    ) -> torch.Tensor:
        """
        Args:
            cost:  (B, H, H) non-negative pairwise cost matrix.
            gamma: soft-min temperature (> 0).

        Returns:
            (B,) soft-DTW value per batch element.
        """
        assert cost.dim() == 3, "cost must be (B, H, H)"
        B, H, W = cost.shape
        assert H == W, "soft-DTW expects a square (H, H) cost matrix"
        dev, dt = cost.device, cost.dtype

        # R is padded by one on each axis: R[:, 0, :] and R[:, :, 0] are the
        # +inf boundary (no path enters from off-grid), R[:, 0, 0] = 0 origin.
        R = torch.full((B, H + 1, H + 1), float("inf"), device=dev, dtype=dt)
        R[:, 0, 0] = 0.0

        if _use_triton(cost):
            cost = cost.contiguous()
            _softdtw_forward_kernel[(B,)](
                cost, R, float(gamma), H,
                BLOCK=triton.next_power_of_2(H), num_warps=1)
        else:
            # Sweep anti-diagonals k = i + j, i,j in [1, H].  For each k gather
            # the cells (i, j) with i + j == k, read their three predecessors as
            # flat vectors, soft-min them, and add the (i-1, j-1) cost cell.
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

        # A non-finite value (NaN/Inf cost) is deliberately NOT asserted away:
        # it propagates into the returned loss so train.py's isfinite /
        # _maybe_restore_from_ema resilience guard handles it instead of crashing.
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
        """
        Args:
            grad_value: (B,) upstream gradient w.r.t. the per-sample value.

        Returns:
            (grad_cost (B, H, H), None) — None for the non-tensor ``gamma``.
        """
        cost, R = ctx.saved_tensors  # type: ignore[attr-defined]
        gamma: float = ctx.gamma     # type: ignore[attr-defined]
        H: int = ctx.H               # type: ignore[attr-defined]
        B = cost.shape[0]
        dev, dt = cost.device, cost.dtype

        E = torch.zeros((B, H + 2, H + 2), device=dev, dtype=dt)
        E[:, H + 1, H + 1] = 1.0

        if _use_triton(cost):
            # D and Rp below are pure padding — the kernel reads their rings as
            # masked-load defaults instead, so it allocates E alone.
            _softdtw_backward_kernel[(B,)](
                cost, R, E, float(gamma), H,
                BLOCK=triton.next_power_of_2(H), num_warps=1)
        else:
            # Pad the cost so cost-cell indexing inside the E-recursion is
            # uniform; the dual table E is (H+2) on each axis with the terminal
            # seed E[H+1, H+1] handled via the boundary below.
            D = torch.zeros((B, H + 2, H + 2), device=dev, dtype=dt)
            D[:, 1:H + 1, 1:H + 1] = cost

            # R re-padded to (H+2): outer ring -inf so off-grid neighbours never
            # win the soft-assignment; terminal cell seeds the recursion.
            Rp = torch.full((B, H + 2, H + 2), -float("inf"), device=dev, dtype=dt)
            Rp[:, 0:H + 1, 0:H + 1] = R
            Rp[:, H + 1, H + 1] = R[:, H, H]

            # Reverse anti-diagonal sweep k = i + j, i,j in [H, 1].  Each cell's
            # soft-assignment is the sum of its three successors' assignments
            # times the local soft-min derivatives a/b/c.
            for k in range(2 * H, 1, -1):
                i_lo = max(1, k - H)
                i_hi = min(H, k - 1)
                if i_lo > i_hi:
                    continue
                i = torch.arange(i_lo, i_hi + 1, device=dev)       # (K,)
                j = k - i                                          # (K,)

                # a: (i+1, j) came from (i, j) via the "up" predecessor.
                a = ((Rp[:, i + 1, j] - Rp[:, i, j] - D[:, i + 1, j]) / gamma).exp()
                # b: (i+1, j+1) came from (i, j) via the "diag" predecessor.
                b = ((Rp[:, i + 1, j + 1] - Rp[:, i, j] - D[:, i + 1, j + 1]) / gamma).exp()
                # c: (i, j+1) came from (i, j) via the "left" predecessor.
                c = ((Rp[:, i, j + 1] - Rp[:, i, j] - D[:, i, j + 1]) / gamma).exp()

                E[:, i, j] = (
                    E[:, i + 1, j] * a
                    + E[:, i + 1, j + 1] * b
                    + E[:, i, j + 1] * c
                )

        e = E[:, 1:H + 1, 1:H + 1]                             # (B, H, H)
        # Non-finite E (from a non-finite forward) is allowed to flow into the
        # gradient — the train.py resilience guard catches the bad step.
        grad_cost = grad_value.view(B, 1, 1) * e
        return grad_cost, None


def _omega_distance(H: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Off-diagonal squared-distance penalty matrix for the TDI term.

    Args:
        H: horizon length.

    Returns:
        (H, H) where ``Ω[i, j] = ((i - j) / H)**2`` — the normalised squared
        deviation of an alignment cell from the diagonal.
    """
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
    """
    DILATE loss in risk space: shape (divergence soft-DTW) + time (TDI).

    The shape term is the soft-DTW DIVERGENCE
    ``sDTW(m, y) - ½ sDTW(m, m) - ½ sDTW(y, y)`` so that ``shape ≈ 0`` and its
    gradient vanishes when ``m == y``; the ``y``-self term is a constant in the
    forecast and is detached.  The time term ``<A, Ω>`` (the soft alignment
    ``A = ∂sDTW/∂C`` against the off-diagonal distance ``Ω``) is evaluated as the
    directional derivative of the soft-DTW value along ``Ω`` via a one-sided
    finite difference ``[sDTW(C + ε·Ω) - sDTW(C)] / ε`` — one extra
    ``SoftDTWBatch`` forward, reusing ``sDTW(C)`` — never by materialising the
    alignment matrix.  Cost is the squared difference in risk space.

    The loss is NOT scale-free in ``H``: the shape term grows with the horizon
    while the normalised TDI (``Ω[i,j] = ((i-j)/H)**2``) does not track it, so
    ``alpha`` weights a different mixture at each ``H``.  A caller comparing or
    combining calls at different ``H`` — the span-length buckets of the masked-BG
    objective — must weight and log them per bucket rather than treat the scalars
    as commensurate.

    Args:
        m:      (B, H) forecast median trajectory in RISK space (requires grad).
                ``B > 0``: the reductions are means over the batch axis, so an
                empty bucket would yield NaN and is rejected, not skipped here.
        y_risk: (B, H) f-transformed target trajectory in RISK space.
        alpha:  shape/time mix in ``[0, 1]`` (``L = α·shape + (1-α)·tdi``).
        gamma:  soft-min softness knob (> 0; smaller ⇒ harder min).  The
                max-subtracted softmin is overflow-free in fp32 to ``1e-3``;
                soft-DTW is 1-homogeneous in ``(cost, gamma)``.
        tdi_fd_eps: finite-difference step for the TDI directional derivative
                (> 0; smaller ⇒ less ``O(ε)`` bias, less fp headroom in the value
                difference).

    Returns:
        (loss, shape, tdi) — all scalar (mean over batch); ``loss`` is the
        DILATE-combined term, ``shape`` and ``tdi`` returned for logging.
    """
    assert m.dim() == 2 and y_risk.dim() == 2, "m and y_risk must be (B, H)"
    assert m.shape == y_risk.shape, "m and y_risk must share shape (B, H)"
    # An EMPTY batch is rejected loudly. Both reductions below are ``.mean()``
    # over the batch axis, so a (0, H) input would return (nan, nan, nan) with no
    # exception — and that NaN reaches the training/validation totals, where a
    # ``val_total < best_val_loss`` comparison is False for NaN against an
    # initial float('inf') and the run ends with no best checkpoint. A caller
    # bucketing spans by length MUST skip its empty buckets (it also pays the
    # full 2H-1 anti-diagonal sweep for them otherwise); this assert is what
    # stops a missed skip from becoming a silent NaN.
    assert m.shape[0] > 0, (
        "dilate_loss got an EMPTY batch (0 rows): the mean over the batch axis "
        "would be NaN. Skip empty span-length buckets in the caller."
    )
    assert m.shape[1] > 0, "dilate_loss got a zero-length horizon (B, 0)"
    assert 0.0 <= alpha <= 1.0, "alpha must be in [0, 1]"
    assert gamma > 0.0, "gamma must be positive"
    assert tdi_fd_eps > 0.0, "tdi_fd_eps must be positive"
    # fp32-native: the production caller (risk_loss.py) hands fp32, so the loss
    # runs in fp32.  Only *promote* a lower-precision
    # input (fp16 / bf16) up to fp32 — never DOWNCAST.  A hard ``.float()`` would
    # silently truncate an fp64 input, which both loses precision the caller may
    # have deliberately supplied and breaks fp64 ``torch.autograd.gradcheck``
    # (the finite-difference numerator would be dominated by fp32 rounding).
    if m.dtype in (torch.float16, torch.bfloat16):
        m = m.float()
    if y_risk.dtype in (torch.float16, torch.bfloat16):
        y_risk = y_risk.float()
    B, H = m.shape

    # --- shape: divergence soft-DTW (cross term minus the two self terms) ---
    cost_my = _pairwise_sq_cost(m, y_risk)
    sdtw_my = SoftDTWBatch.apply(cost_my, gamma)               # (B,)

    cost_mm = _pairwise_sq_cost(m, m)
    sdtw_mm = SoftDTWBatch.apply(cost_mm, gamma)               # (B,)

    # y-self term is a detached constant in the forecast.
    with torch.no_grad():
        cost_yy = _pairwise_sq_cost(y_risk, y_risk)
        sdtw_yy = SoftDTWBatch.apply(cost_yy, gamma)           # (B,)

    shape_per = sdtw_my - 0.5 * sdtw_mm - 0.5 * sdtw_yy        # (B,)
    shape = shape_per.mean()

    # --- time: TDI = <A, Ω> via the directional-derivative finite difference.
    # A = ∂sDTW(cost_my)/∂C is the soft alignment, so <A, Ω> is the directional
    # derivative of the soft-DTW value along Ω; one extra SoftDTWBatch forward
    # (reusing sdtw_my) approximates it to O(tdi_fd_eps), and autograd carries the
    # exact TDI gradient through SoftDTWBatch's first-order backward.
    omega = _omega_distance(H, m.device, m.dtype)             # (H, H)
    sdtw_my_eps = SoftDTWBatch.apply(
        cost_my + tdi_fd_eps * omega.unsqueeze(0), gamma)      # (B,)
    tdi_per = (sdtw_my_eps - sdtw_my) / tdi_fd_eps             # (B,)
    tdi = tdi_per.mean()

    loss = alpha * shape + (1.0 - alpha) * tdi
    return loss, shape, tdi
