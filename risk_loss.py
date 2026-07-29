"""
T1DMAI Risk-space BG loss — pinball quantile loss + DILATE, Kendall-Gal weighted.
=================================================================================

The redesign forecasts BG in **Kovatchev risk space** (space (c) of the frozen
contract): the model emits an ascending bundle of quantiles per future timestep,
and the target glucose trajectory is risk-transformed exactly once here, at the
top of :func:`risk_total_loss`.

Two complementary terms supervise the forecast:

* **L_Q — pinball (quantile) loss.** The pinball / check loss over all seven
  quantile levels, including τ=0.5.  τ=0.5 is kept deliberately as a *pointwise*
  level anchor: it pins the median to the target value at each step, complementing
  DILATE's warp-invariant shape objective (which is insensitive to a constant
  level offset).  The pinball loss is computed in risk space against ``y_risk``.

* **L_D — DILATE on the median.** A shape (divergence soft-DTW) + temporal
  (TDI) loss (Le Guen & Thome, NeurIPS 2019) on the median trajectory only,
  reshaped to ``(B, P*S)`` patch-major.  DILATE is **recomputed at validation**
  (at ``VAL_BATCH_SIZE``) — the soft-DTW dynamic program runs in both phases.

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

from typing import Dict, Tuple

import torch
import torch.nn as nn

import config
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


def pinball_loss(
    q_tau: torch.Tensor,
    y_risk: torch.Tensor,
    levels: Tuple[float, ...],
) -> torch.Tensor:
    """Pinball (quantile / check) loss over all quantile levels, in risk space.

    For each quantile level ``τ`` and prediction ``b = q_τ`` against target
    ``a = y_risk``, the per-element check loss is
    ``ρ_τ(a, b) = (a - b) · (τ - 1[a < b])``.  The result is the mean over the
    horizon (patch × step) and over the quantile levels, reduced to a scalar.

    τ=0.5 is **retained** (a pointwise level anchor that complements DILATE's
    warp-invariant median shape loss).

    Args:
        q_tau: predicted quantiles in risk space,
            ``(B, PREDICTION_PATCHES, PATCH_SIZE, N_QUANTILES)``, ascending τ.
        y_risk: risk-space target glucose, ``(B, PREDICTION_PATCHES, PATCH_SIZE)``.
        levels: the quantile levels ``τ``, length ``N_QUANTILES``, ascending.

    Returns:
        Scalar pinball loss, mean over ``(B, P, S, τ)``.
    """
    assert q_tau.dim() == 4, f"q_tau must be (B,P,S,Q), got {tuple(q_tau.shape)}"
    assert y_risk.dim() == 3, f"y_risk must be (B,P,S), got {tuple(y_risk.shape)}"
    assert q_tau.shape[:3] == y_risk.shape, (
        f"q_tau {tuple(q_tau.shape)} and y_risk {tuple(y_risk.shape)} "
        "must share (B,P,S)"
    )
    assert q_tau.shape[-1] == len(levels), (
        f"q_tau has {q_tau.shape[-1]} quantiles but {len(levels)} levels given"
    )

    tau_key = (levels, q_tau.dtype, q_tau.device)
    tau = _TAU_CACHE.get(tau_key)
    if tau is None:
        tau = torch.as_tensor(levels, dtype=q_tau.dtype, device=q_tau.device)  # (Q,)
        _TAU_CACHE[tau_key] = tau
    a = y_risk.unsqueeze(-1)  # (B,P,S,1)  broadcast over τ
    b = q_tau  # (B,P,S,Q)
    diff = a - b  # (B,P,S,Q)
    rho = diff * (tau - (diff < 0).to(q_tau.dtype))  # (a-b)*(τ - 1[a<b])
    return rho.mean()


# Set of ``(p, s, device)`` for which the patch-major time-monotonicity sentinel
# has already been verified.  The probe depends only on the shape and device, and
# ``torch.sort`` + ``torch.equal`` force a host sync; running it once per unique
# key removes that sync from the hot path while preserving the assertion semantics.
_PATCH_MAJOR_PROBE_VERIFIED: set = set()


def _to_patch_major(x: torch.Tensor) -> torch.Tensor:
    """Reshape a ``(B, P, S)`` trajectory to ``(B, P*S)`` patch-major/step-minor.

    The flatten is C-contiguous so that patch ``p`` step ``s`` lands at flat index
    ``p*S + s`` — i.e. time runs monotonically along the flat axis (patch-major,
    step-minor).  This ordering is load-bearing for DILATE's temporal alignment;
    a P/S transpose would silently scramble the time axis.

    Args:
        x: ``(B, PREDICTION_PATCHES, PATCH_SIZE)``.

    Returns:
        ``(B, PREDICTION_PATCHES * PATCH_SIZE)``, time-monotone along axis 1.
    """
    assert x.dim() == 3, f"expected (B,P,S), got {tuple(x.shape)}"
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


def risk_total_loss(
    q_tau: torch.Tensor,
    median: torch.Tensor,
    true_bg_mgdl: torch.Tensor,
    weighting: KendallGalWeighting,
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
    risk space, on the ``(B, P*S)`` patch-major reshape of median and ``y_risk``.

    Args:
        q_tau: predicted quantiles in risk space,
            ``(B, PREDICTION_PATCHES, PATCH_SIZE, N_QUANTILES)``, ascending τ.
        median: predicted median in risk space (== ``q_tau[..., 3]``),
            ``(B, PREDICTION_PATCHES, PATCH_SIZE)``.
        true_bg_mgdl: target glucose in **mg/dL** (space (b), raw mg/dL),
            ``(B, PREDICTION_PATCHES, PATCH_SIZE)``.
        weighting: the :class:`KendallGalWeighting` module holding the two learned
            log-σ parameters combined here.

    Returns:
        ``(total, components)`` where ``total`` is the scalar Kendall-Gal combined
        loss and ``components`` is a dict of detached-for-logging scalars:
        ``{loss_Q, loss_D, loss_D_shape, loss_D_tdi, log_sigma_Q, log_sigma_D}``.
    """
    assert q_tau.dim() == 4, f"q_tau must be (B,P,S,Q), got {tuple(q_tau.shape)}"
    assert median.dim() == 3, f"median must be (B,P,S), got {tuple(median.shape)}"
    assert true_bg_mgdl.shape == median.shape, (
        f"true_bg_mgdl {tuple(true_bg_mgdl.shape)} must match median "
        f"{tuple(median.shape)}"
    )
    assert q_tau.shape[:3] == median.shape, (
        f"q_tau {tuple(q_tau.shape)} and median {tuple(median.shape)} "
        "must share (B,P,S)"
    )

    # (b)->(c) target bridge — applied EXACTLY ONCE, shared by pinball + DILATE.
    y_risk = kovatchev_f_target(true_bg_mgdl)  # (B,P,S) risk space

    # --- L_Q: pinball over all quantile levels (incl. τ=0.5) ---
    loss_Q = pinball_loss(q_tau, y_risk, config.QUANTILE_LEVELS)

    # --- L_D: DILATE on the median only, (B, P*S) patch-major ---
    median_flat = _to_patch_major(median)
    y_flat = _to_patch_major(y_risk)
    loss_D, loss_D_shape, loss_D_tdi = dilate_loss(
        median_flat,
        y_flat,
        alpha=config.DILATE_ALPHA,
        gamma=config.DILATE_GAMMA,
    )

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
    return total, components
