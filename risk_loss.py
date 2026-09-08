"""Risk-space BG loss: pinball + (per-span DILATE mixed with MSE), Kendall-Gal weighted.

Supervised set = a window's MASKED patches in the head's ``M`` slots; padded slots
gather patch 0, so every term takes the ``(B, M)`` ``valid`` flag. ``kovatchev_f_target``
is the sole (b)->(c) target bridge, run once at the top of :func:`risk_total_loss`."""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

import config
from dilate import dilate_loss
from utils import kovatchev_f_target


class KendallGalWeighting(nn.Module):
    """The two 0-d Kendall-Gal log-σ (Kendall, Gal & Cipolla, CVPR 2018).

    Off ``model`` so :class:`ModelEMA` structurally never sees them; their own AdamW
    group, weight_decay 0, never Muon — a log-variance must not decay toward 0.
    """

    _CLAMP_LO: float = -7.0
    _CLAMP_HI: float = 7.0

    def __init__(self) -> None:
        super().__init__()
        init = float(config.KENDALL_LOGVAR_INIT)
        self.log_sigma_Q = nn.Parameter(torch.zeros(()) + init)
        self.log_sigma_D = nn.Parameter(torch.zeros(()) + init)

    def clamped(self) -> Tuple[torch.Tensor, torch.Tensor]:
        lo, hi = self._CLAMP_LO, self._CLAMP_HI
        return (self.log_sigma_Q.clamp(lo, hi), self.log_sigma_D.clamp(lo, hi))


# τ rebuild per call = host->device copy; cached tensor is bit-identical to a fresh one.
_TAU_CACHE: Dict[Tuple, torch.Tensor] = {}


def pinball_loss(
    q_tau: torch.Tensor,
    y_risk: torch.Tensor,
    levels: Tuple[float, ...],
    valid: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``ρ_τ(a,b) = (a-b)·(τ - 1[a<b])`` over all τ, risk space, mean over (valid slot, step, τ).

    q_tau ``(B,M,S,Q)`` ascending τ; y_risk ``(B,M,S)``; valid ``(B,M)``, ``None`` = dense
    right-edge case. A padded slot must leave the DENOMINATOR too — zeroing only the
    numerator still divides by ``B·M·S·Q``, silently rescaling L_Q against L_D."""
    assert q_tau.dim() == 4, f"q_tau must be (B,M,S,Q), got {tuple(q_tau.shape)}"
    assert y_risk.dim() == 3, f"y_risk must be (B,M,S), got {tuple(y_risk.shape)}"
    assert q_tau.shape[:3] == y_risk.shape, (
        f"q_tau {tuple(q_tau.shape)} and y_risk {tuple(y_risk.shape)} "
        "must share (B,M,S)"
    )
    assert q_tau.shape[-1] == len(levels), (
        f"q_tau has {q_tau.shape[-1]} quantiles but {len(levels)} levels given"
    )
    # No assert above sees ``valid``: a (B,) or transposed mask broadcasts and reweights.
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
    rho = diff * (tau - (diff < 0).to(q_tau.dtype))
    if valid is None:
        return rho.mean()
    w = valid.to(rho.dtype)  # (B,M)
    # Denom = weight mass·S·Q (not slot count); clamped at 1 so all-padded returns exact 0.0.
    denom = (w.sum() * float(rho.shape[2] * len(levels))).clamp_min(1.0)
    return (rho * w[:, :, None, None]).sum() / denom


def mse_loss(
    median: torch.Tensor,
    y_risk: torch.Tensor,
    valid: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``mean (median − y_risk)²`` over every valid (slot, step), risk space.

    median/y_risk ``(B,M,S)``; valid ``(B,M)``, ``None`` = every slot real. Same
    denominator rule as :func:`pinball_loss`: weight mass·S, clamped at 1, so
    all-padded returns exact 0.0."""
    assert median.dim() == 3, f"median must be (B,M,S), got {tuple(median.shape)}"
    assert median.shape == y_risk.shape, (
        f"median {tuple(median.shape)} and y_risk {tuple(y_risk.shape)} must match"
    )
    assert valid is None or tuple(valid.shape) == tuple(median.shape[:2]), (
        f"valid {None if valid is None else tuple(valid.shape)} must be "
        f"(B,M) = {tuple(median.shape[:2])}"
    )
    sq = (median - y_risk) ** 2  # (B,M,S)
    if valid is None:
        return sq.mean()
    w = valid.to(sq.dtype)  # (B,M)
    denom = (w.sum() * float(sq.shape[2])).clamp_min(1.0)
    return (sq * w[:, :, None]).sum() / denom


# (p,s,device) keys whose monotonicity sentinel ran; sort+equal force a host sync, cached once.
_PATCH_MAJOR_PROBE_VERIFIED: set = set()


def _to_patch_major(x: torch.Tensor) -> torch.Tensor:
    """(N, P, S) -> (N, P*S) patch-major: patch ``p`` step ``s`` lands at ``p*S + s``.

    Time runs monotonically along axis 1, which DILATE's alignment depends on; a P/S
    transpose scrambles time with no shape error. On the bucketed path ``N`` is a
    bucket's span count, ``P`` its span length ``L``."""
    assert x.dim() == 3, f"expected (N,P,S), got {tuple(x.shape)}"
    b, p, s = x.shape
    flat = x.reshape(b, p * s)
    # A constructed arange must stay sorted through the reshape — the P/S-swap guard.
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
    """Contiguous masked spans bucketed by length: ``{L: (rows, starts)}``, one entry per span.

    A length with no span is ABSENT, never present-and-empty. ``None`` = one dense
    right-edge span per row. "Same span" = adjacent slots AND patch indices one
    apart (separator-guaranteed); host-side, one D2H copy of the (B,M) tensors per call."""
    v = (torch.ones(b, m, dtype=torch.bool) if valid is None
         else valid.detach().to("cpu", torch.bool))
    idx = (torch.arange(m, dtype=torch.int64).expand(b, m) if mask_idx is None
           else mask_idx.detach().to("cpu", torch.int64))

    cont = torch.zeros_like(v)
    if m > 1:
        # Out-of-order valid slots split or merge spans with no shape error downstream.
        adjacent = v[:, 1:] & v[:, :-1]
        assert not bool((adjacent & (idx[:, 1:] <= idx[:, :-1])).any()), (
            "mask_idx must be strictly ascending over a row's valid slots"
        )
        cont[:, 1:] = adjacent & (idx[:, 1:] == idx[:, :-1] + 1)
    starts = v & ~cont                                          # (B, M)

    # Span id within the row, keyed per row so one scatter_add counts every span at once.
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
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Pinball + per-span DILATE/MSE, Kendall-Gal weighted; target f-transformed once, here.

    DILATE runs on the MEDIAN once per span (not per sample), bucketed by length ``L``,
    patch-major. An empty bucket is NEVER dispatched — ``(0,H)`` means to NaN, killing
    checkpoint selection. Buckets combine by a SPAN-COUNT-WEIGHTED mean, never concatenated."""
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
    # No assert above sees ``valid`` / ``mask_idx``: a wrong shape broadcasts, never raises.
    assert valid is None or tuple(valid.shape) == tuple(median.shape[:2]), (
        f"valid {None if valid is None else tuple(valid.shape)} must be "
        f"(B,M) = {tuple(median.shape[:2])}"
    )
    assert mask_idx is None or tuple(mask_idx.shape) == tuple(median.shape[:2]), (
        f"mask_idx {None if mask_idx is None else tuple(mask_idx.shape)} must be "
        f"(B,M) = {tuple(median.shape[:2])}"
    )

    b_size, n_slots, n_steps = median.shape
    alpha = float(config.MSE_ALPHA)
    assert 0.0 <= alpha <= 1.0, f"MSE_ALPHA must be in [0, 1], got {alpha}"

    # (b)->(c) target bridge, EXACTLY ONCE, shared by every term.
    y_risk = kovatchev_f_target(true_bg_mgdl)  # (B,M,S) risk space

    loss_Q = pinball_loss(q_tau, y_risk, config.QUANTILE_LEVELS, valid=valid)

    buckets = _span_buckets(mask_idx, valid, b_size, n_slots)
    n_spans_total = sum(len(rows_l) for rows_l, _ in buckets.values())
    n_masked_total = sum(len(rows_l) * length for length, (rows_l, _) in buckets.items())
    dev = median.device
    per_bucket: Dict[int, torch.Tensor] = {}
    num_loss = num_shape = num_tdi = None
    for length in (sorted(buckets) if alpha < 1.0 else ()):
        rows_l, starts_l = buckets[length]
        n_b = len(rows_l)
        if n_b == 0:
            # _span_buckets emits no empty bucket; this fires only if that changes.
            continue
        rows = torch.as_tensor(rows_l, dtype=torch.long, device=dev)      # (n_b,)
        starts = torch.as_tensor(starts_l, dtype=torch.long, device=dev)  # (n_b,)
        slots = starts.unsqueeze(1) + torch.arange(length, device=dev)    # (n_b, L)
        # Gathering only the span's slots leaves a padded slot NO grad path, not one zeroed.
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

    if num_loss is None:
        # Every slot padded, or MSE only: an exact zero on the loss dtype/device, not a 0/0 NaN.
        loss_D = median.new_zeros(())
        loss_D_shape = median.new_zeros(())
        loss_D_tdi = median.new_zeros(())
    else:
        denom = float(n_spans_total)
        loss_D = num_loss / denom
        loss_D_shape = num_shape / denom
        loss_D_tdi = num_tdi / denom

    loss_M = (mse_loss(median, y_risk, valid=valid) if alpha > 0.0
              else median.new_zeros(()))
    loss_DR = (1.0 - alpha) * loss_D + alpha * loss_M

    log_sigma_Q, log_sigma_D = weighting.clamped()
    total = (0.5 * torch.exp(-2.0 * log_sigma_Q) * loss_Q + log_sigma_Q
             + 0.5 * torch.exp(-2.0 * log_sigma_D) * loss_DR + log_sigma_D)

    components: Dict[str, torch.Tensor] = {
        "loss_Q": loss_Q.detach(),
        "loss_D": loss_D.detach(),
        "loss_M": loss_M.detach(),
        "loss_D_shape": loss_D_shape.detach(),
        "loss_D_tdi": loss_D_tdi.detach(),
        "log_sigma_Q": log_sigma_Q.detach(),
        "log_sigma_D": log_sigma_D.detach(),
    }
    # Configured UNION realised span lengths, so an empty-this-batch bucket reports 0, not dropped.
    zero = median.new_zeros(())
    for length in sorted(set(config.MASK_SPAN_LENGTHS) | set(buckets)):
        components[f"loss_D_L{length}"] = per_bucket.get(length, zero)
        components[f"n_spans_L{length}"] = torch.tensor(
            float(len(buckets.get(length, ((), ()))[0])))
    components["n_masked_mean"] = torch.tensor(float(n_masked_total) / max(b_size, 1))
    components["n_spans_mean"] = torch.tensor(float(n_spans_total) / max(b_size, 1))
    return total, components
