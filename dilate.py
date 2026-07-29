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
and the whole diagonal vectorised — never a per-sample Python loop (the batch
axis is ``BATCH_SIZE`` and the horizon ``H = PREDICTION_PATCHES * PATCH_SIZE``).
``SoftDTWBatch``'s forward and backward are hand-written so
the DP runs once and the gradient is the standard soft-DTW backward recursion
(no autograd graph over the ``2·H-1`` diagonal steps).

Everything is fp32; the soft-min is max-subtraction-stabilised.  Cost is the
squared difference in risk space — a single cell peaks at
``(f(BG_CLAMP_MAX) - f(BG_CLAMP_MIN))**2 = 40.0`` (with the default
``BG_CLAMP_MAX = 400`` / ``BG_CLAMP_MIN = 40``), and the max-subtracted softmin
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
    ``(f(BG_CLAMP_MAX) - f(BG_CLAMP_MIN))**2 = 40.0``; this softmin
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

        # Sweep anti-diagonals k = i + j, i,j in [1, H].  For each k gather the
        # cells (i, j) with i + j == k, read their three predecessors as flat
        # vectors, soft-min them, and add the (i-1, j-1) cost cell.
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

        # Pad the cost so cost-cell indexing inside the E-recursion is uniform;
        # the dual table E is (H+2) on each axis with the terminal seed E[H+1,
        # H+1] handled via the boundary below.
        D = torch.zeros((B, H + 2, H + 2), device=dev, dtype=dt)
        D[:, 1:H + 1, 1:H + 1] = cost

        # R re-padded to (H+2): outer ring -inf so off-grid neighbours never
        # win the soft-assignment; terminal cell seeds the recursion.
        Rp = torch.full((B, H + 2, H + 2), -float("inf"), device=dev, dtype=dt)
        Rp[:, 0:H + 1, 0:H + 1] = R
        Rp[:, H + 1, H + 1] = R[:, H, H]

        E = torch.zeros((B, H + 2, H + 2), device=dev, dtype=dt)
        E[:, H + 1, H + 1] = 1.0

        # Reverse anti-diagonal sweep k = i + j, i,j in [H, 1].  Each cell's
        # soft-assignment is the sum of its three successors' assignments times
        # the local soft-min derivatives a/b/c.
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

    Args:
        m:      (B, H) forecast median trajectory in RISK space (requires grad).
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
