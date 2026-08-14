"""
T1DMAI Risk-space BG loss — pinball quantile loss + DILATE, Kendall-Gal weighted.
=================================================================================

The redesign forecasts BG in **Kovatchev risk space** (space (c) of the frozen
contract): the model emits an ascending bundle of quantiles per future timestep,
and the target glucose trajectory is risk-transformed exactly once here, at the
top of :func:`risk_total_loss`.

The supervised set is the **masked patches** of a window, gathered into the
head's ``M`` fixed slots (``MAX_MASKED_PATCHES``); a sample with fewer masked
patches pads the surplus and the padded slots gather patch 0, so both terms below
take a ``(B, M)`` ``valid`` mask and neither may reduce over the slot axis by
shape alone.  A trailing forecast is one case of this: it arrives as a single
dense right-edge span, and the ``valid``/``mask_idx`` defaults reproduce the
pre-masking loss exactly.

Two complementary terms supervise the forecast:

* **L_Q — pinball (quantile) loss.** The pinball / check loss over all seven
  quantile levels, including τ=0.5.  τ=0.5 is kept deliberately as a *pointwise*
  level anchor: it pins the median to the target value at each step, complementing
  DILATE's warp-invariant shape objective (which is insensitive to a constant
  level offset).  The pinball loss is computed in risk space against ``y_risk``
  and averaged **per masked patch** — over the valid slots only, denominator
  included.

* **L_D — DILATE on the median.** A shape (divergence soft-DTW) + temporal
  (TDI) loss (Le Guen & Thome, NeurIPS 2019) on the median trajectory only,
  evaluated **once per masked span**: spans are bucketed by length ``L``, each
  bucket stacked to ``(n_b, L*S)`` patch-major for one ``dilate_loss`` call, and
  the per-bucket scalars combined by a span-count-weighted mean.  An empty bucket
  is never dispatched — ``dilate_loss`` would return NaN from a mean over zero
  rows.  DILATE is **recomputed at validation** (at ``VAL_BATCH_SIZE``) — the
  soft-DTW dynamic program runs in both phases.

The two losses are combined by **learned Kendall-Gal homoscedastic uncertainty
weighting** (Kendall, Gal & Cipolla, CVPR 2018): each term carries a learned
log-σ, and the combine ``L_KG = ½·exp(−2·log_σ_Q)·L_Q + log_σ_Q +
½·exp(−2·log_σ_D)·L_D + log_σ_D`` lets the optimizer trade the two objectives
adaptively rather than at a fixed ratio.  The two log-σ live on a small
:class:`KendallGalWeighting` module (:mod:`train.py` keeps it off the weight EMA
and in its own AdamW group), and the numerically-safe ``exp(−2·log_σ)`` form
avoids the ``1/σ²`` division of the naive parameterization.

Everything here is fp32-native (no autocast, no bf16).  ``kovatchev_f_target`` is
the *only* (b)→(c) bridge on the target path and is applied exactly once.
"""

from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

import config
from config import D_BALANCED_LOSS
from dilate import dilate_loss
from utils import kovatchev_f_target


class KendallGalWeighting(nn.Module):
    """Learned homoscedastic-uncertainty weighting of the pinball and DILATE terms.

    Holds the two Kendall-Gal log-variance parameters (Kendall, Gal & Cipolla,
    CVPR 2018), one per loss term, as scalar :class:`nn.Parameter` s.  The module
    is kept deliberately tiny (two 0-d parameters) so :mod:`train.py` can give it
    its own AdamW group (weight_decay 0, never Muon) and exclude it from the
    weight EMA by simply never passing it to :class:`ModelEMA`.

    Attributes:
        log_sigma_Q: scalar log-σ for the pinball term ``L_Q``.
        log_sigma_D: scalar log-σ for the DILATE term ``L_D``.
    """

    _CLAMP_LO: float = -7.0
    _CLAMP_HI: float = 7.0

    def __init__(self) -> None:
        super().__init__()
        init = float(config.KENDALL_LOGVAR_INIT)
        self.log_sigma_Q = nn.Parameter(torch.zeros(()) + init)
        self.log_sigma_D = nn.Parameter(torch.zeros(()) + init)

    def clamped(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the two log-σ clamped to ``[_CLAMP_LO, _CLAMP_HI]``.

        Returns:
            ``(log_sigma_Q, log_sigma_D)``, each a clamped 0-d tensor.
        """
        lo, hi = self._CLAMP_LO, self._CLAMP_HI
        return (self.log_sigma_Q.clamp(lo, hi), self.log_sigma_D.clamp(lo, hi))


# Per-(levels, dtype, device) cache of the τ tensor: rebuilding it every call is
# a host→device copy on the hot path.  The levels are a fixed constant tuple, so
# the cached tensor is bit-identical to a fresh ``torch.as_tensor`` each call.
_TAU_CACHE: Dict[Tuple, torch.Tensor] = {}


@lru_cache(maxsize=1)
def _N_D_GROUPS() -> int:
    from d_balance import N_D_GROUPS

    return int(N_D_GROUPS)


@lru_cache(maxsize=8)
def _d_weight_table(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """``d``-indexed lookup of the per-group weight, index 0 unused (padding).

    Built once per (device, dtype).  Index by the (B, M) ``d`` tensor directly;
    every ``d`` at or above ``N_D_GROUPS`` reads the pooled tail weight, so the
    table is short and no ``d`` can fall off it.
    """
    from d_balance import N_D_GROUPS, d_weights

    w = d_weights()
    # Slot 0 is the padded-slot value; ``valid`` zeroes those anyway, so its
    # magnitude is irrelevant and 1.0 keeps the table free of surprises.
    table = [1.0] + [w[min(d, N_D_GROUPS) - 1] for d in range(1, N_D_GROUPS + 1)]
    return torch.tensor(table, dtype=dtype, device=device)


def pinball_loss(
    q_tau: torch.Tensor,
    y_risk: torch.Tensor,
    levels: Tuple[float, ...],
    valid: Optional[torch.Tensor] = None,
    d: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pinball (quantile / check) loss over all quantile levels, in risk space.

    For each quantile level ``τ`` and prediction ``b = q_τ`` against target
    ``a = y_risk``, the per-element check loss is
    ``ρ_τ(a, b) = (a - b) · (τ - 1[a < b])``.  The result is the mean over the
    supervised slots (slot × step) and over the quantile levels, reduced to a
    scalar.

    τ=0.5 is **retained** (a pointwise level anchor that complements DILATE's
    warp-invariant median shape loss).

    **The reduction is per MASKED PATCH, not per head slot.**  The head emits a
    fixed ``M`` slots (``MAX_MASKED_PATCHES``) and a sample with fewer masked
    patches pads the surplus; a padded slot gathers patch 0, so its ρ is a real
    number against a real target and must be *removed from the denominator*, not
    merely zeroed in the numerator.  Zeroing the numerator alone still divides by
    ``B·M·S·Q`` and rescales ``L_Q`` against ``L_D`` by the padded fraction — a
    level that ``log_sigma_Q`` absorbs, silently moving the converged Q:D balance
    while every per-sample relative weight, and so every loss curve, looks
    unchanged.

    Args:
        q_tau: predicted quantiles in risk space, ``(B, M, PATCH_SIZE,
            N_QUANTILES)``, ascending τ.
        y_risk: risk-space target glucose, ``(B, M, PATCH_SIZE)``.
        levels: the quantile levels ``τ``, length ``N_QUANTILES``, ascending.
        valid: ``(B, M)`` bool — True where the slot holds a real masked patch,
            False for a padded slot.  ``None`` (the default) means every slot is
            real and takes the plain ``rho.mean()``, which is what a caller
            holding a dense right-edge span wants and is bit-identical to the
            pre-masking reduction.

    Returns:
        Scalar pinball loss, mean over ``(valid slots, S, τ)``.
    """
    assert q_tau.dim() == 4, f"q_tau must be (B,M,S,Q), got {tuple(q_tau.shape)}"
    assert y_risk.dim() == 3, f"y_risk must be (B,M,S), got {tuple(y_risk.shape)}"
    assert q_tau.shape[:3] == y_risk.shape, (
        f"q_tau {tuple(q_tau.shape)} and y_risk {tuple(y_risk.shape)} "
        "must share (B,M,S)"
    )
    assert q_tau.shape[-1] == len(levels), (
        f"q_tau has {q_tau.shape[-1]} quantiles but {len(levels)} levels given"
    )
    # The four asserts above pass for ANY M and catch every malformed rank/width,
    # but none of them can see ``valid`` — a (B,) or transposed mask would
    # broadcast silently and reweight the loss.
    assert valid is None or tuple(valid.shape) == tuple(y_risk.shape[:2]), (
        f"valid {None if valid is None else tuple(valid.shape)} must be "
        f"(B,M) = {tuple(y_risk.shape[:2])}"
    )

    tau_key = (levels, q_tau.dtype, q_tau.device)
    tau = _TAU_CACHE.get(tau_key)
    if tau is None:
        tau = torch.as_tensor(levels, dtype=q_tau.dtype, device=q_tau.device)  # (Q,)
        _TAU_CACHE[tau_key] = tau
    a = y_risk.unsqueeze(-1)  # (B,M,S,1)  broadcast over τ
    b = q_tau  # (B,M,S,Q)
    diff = a - b  # (B,M,S,Q)
    rho = diff * (tau - (diff < 0).to(q_tau.dtype))  # (a-b)*(τ - 1[a<b])
    if valid is None:
        return rho.mean()
    w = valid.to(rho.dtype)  # (B,M)
    if d is not None and D_BALANCED_LOSS:
        # Uniform PLACEMENT is not uniform DIFFICULTY. Under the sampler d = 1
        # carries 47% of masked patches and d >= 4 carries 7.7%, while the
        # deployed forecast needs one patch at each of d = 1..4. Reweighting
        # here — and only here — gives each d group equal mass without touching
        # the sampler or its placement.
        #
        # The table is normalised so E[w] == 1 under the sampler, which keeps
        # this loss on the same scale as the unweighted form. A rescale would be
        # absorbed by log_sigma_Q as a level and quietly shift the converged Q:D
        # balance, with nothing in the loss curves looking wrong.
        # Clamp the INDEX, not just the table: d reaches the span ceiling on an
        # edge-touching span (8 at MASK_SPAN_LENGTHS[-1] = 8), well past the
        # pooled tail group, and an unclamped gather is an out-of-bounds read —
        # a device-side assert on CUDA, silent garbage on CPU.
        idx = d.clamp(min=0, max=_N_D_GROUPS())
        w = w * _d_weight_table(rho.device, rho.dtype)[idx]
    # Denominator is the WEIGHT MASS · S · Q — clamped at 1 so an all-padded
    # batch returns an exact 0.0 (numerator is 0 too) instead of 0/0. It tracks
    # the numerator's weights rather than counting slots; counting slots here
    # would rescale the loss by the mean weight.
    denom = (w.sum() * float(rho.shape[2] * len(levels))).clamp_min(1.0)
    return (rho * w[:, :, None, None]).sum() / denom


# Set of ``(p, s, device)`` for which the patch-major time-monotonicity sentinel
# has already been verified.  The probe depends only on the shape and device, and
# ``torch.sort`` + ``torch.equal`` force a host sync; running it once per unique
# key removes that sync from the hot path while preserving the assertion semantics.
_PATCH_MAJOR_PROBE_VERIFIED: set = set()


def _to_patch_major(x: torch.Tensor) -> torch.Tensor:
    """Reshape an ``(N, P, S)`` trajectory to ``(N, P*S)`` patch-major/step-minor.

    The flatten is C-contiguous so that patch ``p`` step ``s`` lands at flat index
    ``p*S + s`` — i.e. time runs monotonically along the flat axis (patch-major,
    step-minor).  This ordering is load-bearing for DILATE's temporal alignment;
    a P/S transpose would silently scramble the time axis.

    Args:
        x: ``(N, P, PATCH_SIZE)`` — ``N`` rows of ``P`` consecutive patches.  On
            the bucketed DILATE path ``N`` is a bucket's span count and ``P`` its
            span length ``L``, not the batch size and ``PREDICTION_PATCHES``.

    Returns:
        ``(N, P * PATCH_SIZE)``, time-monotone along axis 1.
    """
    assert x.dim() == 3, f"expected (N,P,S), got {tuple(x.shape)}"
    b, p, s = x.shape
    flat = x.reshape(b, p * s)
    # Time-monotonicity sentinel: a constructed arange must remain sorted after
    # the reshape, guarding the patch-major flatten order against a P/S swap.  It
    # depends only on ``(p, s, device)``, so verify each unique key once — the
    # ``torch.sort``/``torch.equal`` otherwise force a host sync twice per step.
    probe_key = (p, s, x.device)
    if probe_key not in _PATCH_MAJOR_PROBE_VERIFIED:
        probe = (
            torch.arange(p, device=x.device).view(p, 1) * s
            + torch.arange(s, device=x.device).view(1, s)
        ).reshape(p * s)
        assert torch.equal(probe, torch.sort(probe).values), (
            "patch-major flatten is not time-monotone — P/S order is wrong"
        )
        _PATCH_MAJOR_PROBE_VERIFIED.add(probe_key)
    return flat


def _span_buckets(
    mask_idx: Optional[torch.Tensor],
    valid: Optional[torch.Tensor],
    b: int,
    m: int,
) -> Dict[int, Tuple[List[int], List[int]]]:
    """Group the head's masked slots into contiguous spans, bucketed by length.

    Two valid slots belong to the same span iff they are adjacent in the slot
    axis **and** their patch indices differ by exactly one.  That test is only
    equivalent to "same span" because the mask sampler guarantees a mandatory
    visible patch between neighbouring spans — two spans that abutted would be
    indistinguishable from one longer span here, and would then share a DILATE
    bucket, an anchor and a median basis they do not in fact share.

    The grouping runs on the host: bucket membership is data-dependent Python
    control flow (an empty bucket must not be dispatched at all), so exactly one
    device→host transfer of the two ``(B, M)`` index tensors is unavoidable.  It
    is a single small sync per call, not one per span.

    Args:
        mask_idx: ``(B, M)`` int64 patch index per head slot, ascending over the
            valid slots of a row.  ``None`` means slot ``j`` is patch ``j``, i.e.
            one dense span per row — the legacy right-edge forecast.
        valid: ``(B, M)`` bool, True for a real masked patch.  ``None`` means
            every slot is real.
        b: batch size.
        m: head slot count.

    Returns:
        ``{L: (rows, starts)}`` — for each span length ``L`` present, the row
        index and the first slot index of every span of that length.  Lengths
        with no spans are **absent**, never present-and-empty.
    """
    v = (torch.ones(b, m, dtype=torch.bool) if valid is None
         else valid.detach().to("cpu", torch.bool))
    idx = (torch.arange(m, dtype=torch.int64).expand(b, m) if mask_idx is None
           else mask_idx.detach().to("cpu", torch.int64))

    # A slot continues the span to its left iff that neighbour is valid and sits
    # exactly one patch earlier; everything else opens a new span.
    cont = torch.zeros_like(v)
    if m > 1:
        # Slot order carries the span structure: two adjacent valid slots out of
        # ascending patch order would be split into two spans, or (on a repeat)
        # merged, with no shape error anywhere downstream.
        adjacent = v[:, 1:] & v[:, :-1]
        assert not bool((adjacent & (idx[:, 1:] <= idx[:, :-1])).any()), (
            "mask_idx must be strictly ascending over a row's valid slots"
        )
        cont[:, 1:] = adjacent & (idx[:, 1:] == idx[:, :-1] + 1)
    starts = v & ~cont                                          # (B, M)

    # Span id within the row (1-based over starts), keyed per row so a single
    # scatter_add counts every span's slots at once.
    gid = starts.to(torch.int64).cumsum(dim=1)                  # (B, M)
    keyed = torch.arange(b, dtype=torch.int64).unsqueeze(1) * (m + 1) + gid
    counts = torch.zeros(b * (m + 1), dtype=torch.int64)
    counts.scatter_add_(0, keyed[v], torch.ones(int(v.sum()), dtype=torch.int64))
    lengths = counts[keyed][starts]                             # (n_spans,)
    where = starts.nonzero(as_tuple=False)                      # (n_spans, 2)

    buckets: Dict[int, Tuple[List[int], List[int]]] = {}
    for (row, slot), length in zip(where.tolist(), lengths.tolist()):
        rows_l, starts_l = buckets.setdefault(int(length), ([], []))
        rows_l.append(int(row))
        starts_l.append(int(slot))
    return buckets


def risk_total_loss(
    q_tau: torch.Tensor,
    median: torch.Tensor,
    true_bg_mgdl: torch.Tensor,
    weighting: KendallGalWeighting,
    valid: Optional[torch.Tensor] = None,
    mask_idx: Optional[torch.Tensor] = None,
    d: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Total risk-space BG loss: pinball + DILATE, Kendall-Gal weighted.

    The Kovatchev risk transform is applied to the target **exactly once** here
    (``y_risk = kovatchev_f_target(true_bg_mgdl)``), then shared by both terms.

    The two terms are combined by learned Kendall-Gal homoscedastic-uncertainty
    weighting in its numerically-safe form::

        log_σ_Q, log_σ_D = weighting.clamped()
        L = ½·exp(−2·log_σ_Q)·L_Q + log_σ_Q + ½·exp(−2·log_σ_D)·L_D + log_σ_D

    ``log_σ_Q`` / ``log_σ_D`` are the two learned scalars on ``weighting``
    (:class:`KendallGalWeighting`), clamped to ``[-7, 7]``.

    DILATE is computed on the **median only** (its shape/temporal objective), in
    risk space, **once per masked span** rather than once per sample.  A span is
    a maximal run of adjacent masked patches; spans are bucketed by length ``L``
    and each bucket is stacked to ``(n_b, L*S)`` patch-major for a single
    :func:`dilate.dilate_loss` call.  Two rules govern that:

    * **An empty bucket is never dispatched.**  ``dilate_loss`` reduces over the
      batch axis with ``.mean()``, so a ``(0, H)`` input returns ``(nan, nan,
      nan)`` with no exception and no shape assert — and a fixed protocol can
      leave a bucket empty in *every* batch (a right-edge forecast of
      ``PREDICTION_PATCHES`` never populates ``L < PREDICTION_PATCHES``).  That
      NaN would flow through the running totals and past ``val_total <
      best_val_loss``, which is False for NaN against ``float('inf')``, ending
      the run with no best checkpoint.  A span-count-weighted mean does not
      rescue it: ``0.0 * nan = nan``.  Dispatching the empty bucket anyway also
      pays the full ``2H-1`` anti-diagonal sweep for nothing.
    * **Buckets combine by a span-count-weighted mean of the per-bucket
      scalars** — never by concatenation and never unweighted.  DILATE is not
      scale-free in ``H = L·S``: the shape term grows with ``H`` while the
      normalised TDI (``Ω[i,j] = ((i-j)/H)²``) does not track it, so ``alpha``
      weights a different mixture in each bucket.  ``log_sigma_D`` is learned and
      absorbs that mixture silently, so the effective Q:D ratio now moves with
      ``MASK_SPAN_LENGTHS`` even though both log-σ *parameters* are pinned.  The
      per-bucket ``loss_D_L{L}`` and the span-length histogram ``n_spans_L{L}``
      are therefore logged beside the combined value: two runs are comparable
      only at an equal span-length mixture.

    Args:
        q_tau: predicted quantiles in risk space, ``(B, M, PATCH_SIZE,
            N_QUANTILES)``, ascending τ.  ``M`` is the head's slot count
            (``MAX_MASKED_PATCHES``), or ``PREDICTION_PATCHES`` on a caller that
            still hands a dense right-edge span.
        median: predicted median in risk space (== ``q_tau[..., 3]``),
            ``(B, M, PATCH_SIZE)``.
        true_bg_mgdl: target glucose in **mg/dL** (space (b), raw mg/dL),
            ``(B, M, PATCH_SIZE)``.
        weighting: the :class:`KendallGalWeighting` module holding the two learned
            log-σ parameters combined here.
        valid: ``(B, M)`` bool — True where the slot holds a real masked patch.
            ``None`` (the default) means every slot is real, which is exactly the
            dense right-edge case and reproduces the pre-masking loss.
        mask_idx: ``(B, M)`` int64 patch index per slot, ascending over a row's
            valid slots.  ``None`` means slot ``j`` is patch ``j`` — one dense
            span per row, one bucket, one ``dilate_loss`` call.

    Returns:
        ``(total, components)`` where ``total`` is the scalar Kendall-Gal combined
        loss and ``components`` is a dict of detached-for-logging scalars:
        ``{loss_Q, loss_D, loss_D_shape, loss_D_tdi, log_sigma_Q, log_sigma_D}``
        plus, per span length ``L``, ``loss_D_L{L}`` and the span count
        ``n_spans_L{L}``, plus ``n_masked_mean`` and ``n_spans_mean`` (the mean
        masked-patch and span counts per sample).  The counters are host scalars.
    """
    assert q_tau.dim() == 4, f"q_tau must be (B,M,S,Q), got {tuple(q_tau.shape)}"
    assert median.dim() == 3, f"median must be (B,M,S), got {tuple(median.shape)}"
    assert true_bg_mgdl.shape == median.shape, (
        f"true_bg_mgdl {tuple(true_bg_mgdl.shape)} must match median "
        f"{tuple(median.shape)}"
    )
    assert q_tau.shape[:3] == median.shape, (
        f"q_tau {tuple(q_tau.shape)} and median {tuple(median.shape)} "
        "must share (B,M,S)"
    )
    # Neither assert above can see ``valid`` or ``mask_idx``: both pass for any M
    # and a wrongly-shaped mask would broadcast rather than raise.
    assert valid is None or tuple(valid.shape) == tuple(median.shape[:2]), (
        f"valid {None if valid is None else tuple(valid.shape)} must be "
        f"(B,M) = {tuple(median.shape[:2])}"
    )
    assert mask_idx is None or tuple(mask_idx.shape) == tuple(median.shape[:2]), (
        f"mask_idx {None if mask_idx is None else tuple(mask_idx.shape)} must be "
        f"(B,M) = {tuple(median.shape[:2])}"
    )

    b_size, n_slots, n_steps = median.shape

    # (b)->(c) target bridge — applied EXACTLY ONCE, shared by pinball + DILATE.
    y_risk = kovatchev_f_target(true_bg_mgdl)  # (B,M,S) risk space

    # --- L_Q: pinball over all quantile levels (incl. τ=0.5), per masked patch ---
    loss_Q = pinball_loss(q_tau, y_risk, config.QUANTILE_LEVELS, valid=valid, d=d)

    # --- L_D: DILATE on the median only, one call per span-length bucket ---
    buckets = _span_buckets(mask_idx, valid, b_size, n_slots)
    dev = median.device
    per_bucket: Dict[int, torch.Tensor] = {}
    num_loss = num_shape = num_tdi = None
    n_spans_total = 0
    n_masked_total = 0
    for length in sorted(buckets):
        rows_l, starts_l = buckets[length]
        n_b = len(rows_l)
        if n_b == 0:
            # An empty bucket is NEVER dispatched: dilate_loss means over the
            # batch axis and would return NaN for it. _span_buckets does not
            # emit one, so this only fires if that ever changes.
            continue
        rows = torch.as_tensor(rows_l, dtype=torch.long, device=dev)      # (n_b,)
        starts = torch.as_tensor(starts_l, dtype=torch.long, device=dev)  # (n_b,)
        slots = starts.unsqueeze(1) + torch.arange(length, device=dev)    # (n_b, L)
        # Gathering only the span's slots is also what makes a padded slot's
        # gradient EXACTLY zero on this term: it is never read, so no grad path
        # to it exists at all (as opposed to one multiplied by zero).
        m_b = _to_patch_major(median[rows.unsqueeze(1), slots])           # (n_b, L*S)
        y_b = _to_patch_major(y_risk[rows.unsqueeze(1), slots])
        l_b, s_b, t_b = dilate_loss(
            m_b,
            y_b,
            alpha=config.DILATE_ALPHA,
            gamma=config.DILATE_GAMMA,
        )
        w = float(n_b)
        num_loss = l_b * w if num_loss is None else num_loss + l_b * w
        num_shape = s_b * w if num_shape is None else num_shape + s_b * w
        num_tdi = t_b * w if num_tdi is None else num_tdi + t_b * w
        per_bucket[length] = l_b.detach()
        n_spans_total += n_b
        n_masked_total += n_b * length

    if num_loss is None:
        # No populated bucket at all (every slot padded). Emit an exact zero on
        # the loss dtype/device rather than a 0/0 NaN.
        loss_D = median.new_zeros(())
        loss_D_shape = median.new_zeros(())
        loss_D_tdi = median.new_zeros(())
    else:
        denom = float(n_spans_total)
        loss_D = num_loss / denom
        loss_D_shape = num_shape / denom
        loss_D_tdi = num_tdi / denom

    # --- Kendall-Gal combine (numerically-safe exp(−2·log_σ) form) ---
    log_sigma_Q, log_sigma_D = weighting.clamped()
    total = (0.5 * torch.exp(-2.0 * log_sigma_Q) * loss_Q + log_sigma_Q
             + 0.5 * torch.exp(-2.0 * log_sigma_D) * loss_D + log_sigma_D)

    components: Dict[str, torch.Tensor] = {
        "loss_Q": loss_Q.detach(),
        "loss_D": loss_D.detach(),
        "loss_D_shape": loss_D_shape.detach(),
        "loss_D_tdi": loss_D_tdi.detach(),
        "log_sigma_Q": log_sigma_Q.detach(),
        "log_sigma_D": log_sigma_D.detach(),
    }
    # Per-bucket DILATE and the span-length histogram. The key set is the union
    # of the configured span lengths and the realised ones, so a bucket that is
    # empty this batch still reports a column (0 spans) instead of dropping out
    # of the log.
    zero = median.new_zeros(())
    for length in sorted(set(config.MASK_SPAN_LENGTHS) | set(buckets)):
        components[f"loss_D_L{length}"] = per_bucket.get(length, zero)
        components[f"n_spans_L{length}"] = torch.tensor(
            float(len(buckets.get(length, ((), ()))[0])))
    components["n_masked_mean"] = torch.tensor(float(n_masked_total) / max(b_size, 1))
    components["n_spans_mean"] = torch.tensor(float(n_spans_total) / max(b_size, 1))
    return total, components
