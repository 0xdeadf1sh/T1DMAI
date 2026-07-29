"""
T1DMAI Inference — standard, what-if, and rolling prediction modes.
====================================================================

This module is the runtime entry point for the model: given a trained
checkpoint and a context window, it builds the patches tensor, runs a
forward pass, and turns the model's RISK-space quantile head outputs into
mg/dL BG forecasts.

The pipeline occupies three spaces, crossed by two bridge pairs:

* (a) normalized z-space — the model **inputs** (bg_absolute, carbs,
  insulin), via ``normalization_stats``.  bg (feat 0) is normalized in
  **risk space**: the Kovatchev ``f`` is applied BEFORE the z-score, so the
  bg input the model sees is ``z(f(bg))`` (``normalize`` routes it through
  ``RISK_SPACE_CHANNELS``); carb / insulin stay ``z(log1p(·))``.
* (b) mg/dL physical — ``last_bg``, the BG forecast, every clinical metric.
* (c) risk space — the head **outputs** (``q_tau`` quantiles, ``median``).

``normalize/denormalize`` is the sole (a)<->(b) crossing;
``utils.kovatchev_f / utils.kovatchev_f_inv`` is the sole (b)<->(c) crossing.
Inference owns the ``kovatchev_f_inv`` (c)->(b) step: the model emits
risk-space quantiles and a median,
and this module inverts them to mg/dL band edges and a headline BG.

Three prediction modes are exposed:

* ``predict``         — single forward pass.  Returns ``N`` hours of
                        forecast (``N`` = the configured horizon).  The
                        headline BG forecast (``median_bg``) is
                        ``kovatchev.f_inv(median)``; the per-(h, τ) band
                        edges (``bands``) are ``kovatchev.f_inv(q_tau)``.
* ``predict_what_if`` — same, but lets the caller override carbs / insulin
                        in the prediction zone (e.g. "what if I eat 40 g
                        carbs at 6 PM?").  The announced values are written
                        straight into the carb (feat 1) / insulin (feat 2)
                        slots via ``CHANNEL_TO_FEAT`` — there are no mask
                        bits; the model is always conditioned on whatever
                        doses occupy those slots.
* ``predict_rolling`` — autoregressive rolling: predict one window, treat
                        the median BG forecast as new context, predict the
                        next window, repeat.  The re-feed is **BG-only**:
                        the median forecast (risk) -> ``f_inv`` -> mg/dL ->
                        ``normalize`` -> ``bg_absolute`` slot 0 of each new
                        context patch.  carbs / insulin come from the
                        caller's ``overrides_fn`` (else zero); there are no
                        dynamics outputs to write back.

Channel-layout note
-------------------
The 3 input features are ``[bg_absolute, carbs, insulin]`` — there are no
temporal (sin/cos) features.  The model has NO dynamics outputs: its head
emits risk-space quantiles over BG only.
"""

import argparse
from typing import Any

import numpy as np
import torch

from config import (
    PREDICTION_PATCHES, PATCH_SIZE, N_INPUT_FEATURES, N_QUANTILES,
    MAX_CONTEXT_PATCHES,
    CHANNEL_TO_FEAT, MASKABLE_FEATS, QUANTILE_LEVELS,
    TIME_PROBE_N_BINS,
)
from model import T1DMAI
from normalization import (
    load_normalization_stats, CHANNEL_NAMES, normalize,
)
from utils import (
    create_attention_mask,
    last_bg_mgdl_from_context, kovatchev_f_inv, kovatchev_f_np,
    time_of_day_decode_bins,
)

# Index of the τ=0.5 median in the ascending quantile fan — the column conformal
# recalibration holds fixed (it moves only the band edges).
_MEDIAN_IDX = QUANTILE_LEVELS.index(0.5)


def _conformal_to_np(delta: Any) -> np.ndarray:
    """Coerce a conformal delta (numpy array OR a torch tensor, possibly on CUDA) to a
    host numpy array — apply_quantile_conformal is pure numpy."""
    if torch.is_tensor(delta):
        return delta.detach().cpu().numpy()
    return np.asarray(delta)


def _build_patches_tensor(
    context: torch.Tensor,
    overrides: dict[int, torch.Tensor] | None = None,
    normalization_stats: dict[str, dict[str, float]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build the ``(T, PATCH_DIM)`` patches tensor and attention mask from
    context and prediction.  ``PATCH_DIM = PATCH_SIZE * N_INPUT_FEATURES`` — there
    are no mask bits; the model is always conditioned on the announced doses.

    Args:
        context: (n_ctx, PATCH_SIZE, N_INPUT_FEATURES) context data
                 (already normalized).
        overrides: Optional dict mapping an output-channel override index to a
                 (PREDICTION_PATCHES, PATCH_SIZE) tensor of normalized override
                 values.  Keys are output-channel indices {0: carb, 1: insulin};
                 each is routed to its input feature slot via the shared
                 ``CHANNEL_TO_FEAT`` (carb -> feat 1, insulin -> feat 2).  BG is
                 never overrideable.  Keys outside ``CHANNEL_TO_FEAT`` are
                 silently ignored.  A pred-zone dose slot left without an override
                 stays at the zero-RAW normalized baseline ``normalize(0)`` for
                 that channel (a genuine no-dose step, the value the model was
                 trained on) when ``normalization_stats`` is supplied — NOT the z=0
                 sentinel, which decodes through the sparse log1p inverse to a
                 phantom dose.
        normalization_stats: normalization statistics used to seed the pred-zone
                 no-dose baseline ``normalize(0)`` for the MASKABLE dose feats.
                 When ``None`` the dose slots fall back to the z=0 sentinel (the
                 legacy phantom-dose behavior); callers on the forecast path
                 (``predict``) always pass it.

    Returns:
        patches: (T, PATCH_DIM) float tensor.
        attn_mask: (T, T) bool tensor.
    """
    assert context.ndim == 3 and context.shape[-1] == N_INPUT_FEATURES, (
        f"context must be (n_ctx, PATCH_SIZE, {N_INPUT_FEATURES}), got {tuple(context.shape)}"
    )
    n_ctx = context.shape[0]

    # --- Context patches ---
    # Flatten (PATCH_SIZE, N_INPUT_FEATURES) → PATCH_DIM.  The context carries its
    # observed feature values (bg / carb / insulin) verbatim.
    ctx_patches = context.reshape(n_ctx, PATCH_SIZE * N_INPUT_FEATURES)  # (n_ctx, PATCH_DIM)

    # --- Prediction patches ---
    # bg (feat 0 / NON_MASKABLE_FEATS) stays 0 — it is what the model predicts.
    # The announced-dose slots (MASKABLE_FEATS = carb feat 1 / insulin feat 2) are
    # seeded to the zero-RAW normalized baseline normalize(0) per channel — a
    # genuine no-dose step — NOT z=0, which decodes through the sparse log1p inverse
    # to a phantom ~0.39 g / ~0.14 U per step (the exact hazard predict_rolling's
    # context fill guards against, and the value the model was TRAINED on for a
    # no-dose step). Overrides below overwrite these with the announced values.
    pred_features = torch.zeros(PREDICTION_PATCHES, PATCH_SIZE, N_INPUT_FEATURES)
    if normalization_stats is not None:
        zero_raw = normalize(
            np.zeros((1, len(CHANNEL_NAMES)), dtype=np.float32), normalization_stats,
        )[0]
        for feat_idx in MASKABLE_FEATS:
            pred_features[:, :, feat_idx] = float(zero_raw[feat_idx])
    pred_patches = pred_features.reshape(PREDICTION_PATCHES, PATCH_SIZE * N_INPUT_FEATURES)

    # Apply overrides — each entry announces output-channel ``ch_idx``'s values in
    # the prediction zone.  ``CHANNEL_TO_FEAT`` maps the output-channel index to its
    # input feature slot (carb -> feat 1, insulin -> feat 2), the SAME mapping
    # data.py uses, so there is no second independent ``+offset`` literal to drift.
    if overrides:
        for ch_idx, override_vals in overrides.items():
            if ch_idx not in CHANNEL_TO_FEAT:
                continue
            feat_idx = CHANNEL_TO_FEAT[ch_idx]
            # override_vals: (PREDICTION_PATCHES, PATCH_SIZE)
            for t in range(PATCH_SIZE):
                flat_col = t * N_INPUT_FEATURES + feat_idx
                pred_patches[:, flat_col] = override_vals[:, t]

    # Combine context + prediction along the sequence axis and build the
    # hybrid attention mask shared by every block.
    patches = torch.cat([ctx_patches, pred_patches], dim=0)  # (T, PATCH_DIM)
    attn_mask = create_attention_mask(n_ctx, PREDICTION_PATCHES)

    return patches, attn_mask


def predict(
    model: T1DMAI,
    context: torch.Tensor,
    patient_seed: int | None = None,
    normalization_stats: dict[str, dict[str, float]] | None = None,
    device: torch.device | None = None,
    overrides: dict[int, torch.Tensor] | None = None,
    conformal_delta: np.ndarray | None = None,
    return_time: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Standard prediction: single forward pass over the context window.

    The model emits RISK-space quantiles and a median; this function inverts
    them to mg/dL via ``kovatchev.f_inv``.

    Args:
        model: T1DMAI model in eval mode.
        context: (n_ctx, PATCH_SIZE, N_INPUT_FEATURES) normalized context data.
                 ``n_ctx`` must be >= ``MIN_CONTEXT_PATCHES``.
        patient_seed: Unused (kept for backwards-compatible call signature;
                 the model does not condition on a patient embedding).
        normalization_stats: stats for the mg/dL ``last_bg`` anchor.  The forward
                 STRUCTURALLY requires the anchor, so when ``None`` this falls
                 back to the on-disk ``normalization_stats.json`` to form it (and
                 raises if that file is absent) — stats are therefore effectively
                 required.  The ``stats is None`` branch differs only in the
                 RESULT: it returns just the raw risk-space ``q_tau`` / ``median``
                 (the mg/dL ``median_bg`` / ``bands`` / ``last_bg`` fields are
                 emitted only when stats are passed EXPLICITLY).
        device: Torch device (default: use the model's device).
        overrides: Optional dict mapping an output-channel index
                 ({0: carb, 1: insulin}) to a (PREDICTION_PATCHES, PATCH_SIZE)
                 tensor of NORMALIZED prediction-zone values, announcing that
                 channel's doses to the model.  Slots without an override stay at
                 the ``normalize(0)`` no-dose baseline (not z=0).  BG is never overrideable.
        return_time: opt-in diagnostic flag.  When ``False`` (default) the compute
                 and returned keys are BIT-IDENTICAL to the pre-existing behavior.
                 When ``True`` the forward runs with ``return_time=True`` and the
                 result gains a single ``time_pred`` key — the auxiliary time-of-day
                 probe's per-patch bin logits (never touches the BG forecast, loss,
                 or checkpoint selection).

    Returns:
        result dict::
            q_tau:     (PREDICTION_PATCHES, PATCH_SIZE, N_QUANTILES) risk-space
                       quantiles, ascending τ.
            median:    (PREDICTION_PATCHES, PATCH_SIZE) risk-space median.
            median_bg: (PREDICTION_PATCHES * PATCH_SIZE,) HEADLINE BG forecast,
                       ``f_inv(median)`` in mg/dL (only if stats provided).
            bands:     (PREDICTION_PATCHES, PATCH_SIZE, N_QUANTILES) mg/dL band
                       edges, ``f_inv(q_tau)`` (only if stats provided).
            last_bg:   scalar mg/dL anchor (only if stats provided).
            time_pred: (PREDICTION_PATCHES, TIME_PROBE_N_BINS) raw time-of-day probe
                       bin logits, or ``None`` when the probe is disabled — present
                       ONLY when ``return_time=True``.
    """
    del patient_seed  # no longer used by the model
    if device is None:
        device = next(model.parameters()).device

    # Resolve the normalization stats FIRST: they are needed both for the mg/dL
    # ``last_bg`` anchor (the median's risk anchor is ``f(last_bg)``) AND for the
    # prediction-zone no-dose baseline ``normalize(0)`` (see _build_patches_tensor —
    # z=0 would inject a phantom dose).  When the caller supplies ``normalization_stats``
    # we use them; otherwise we fall back to the on-disk stats and raise a clear
    # error if neither is available (instead of an opaque ``FileNotFoundError``).
    if normalization_stats is not None:
        anchor_stats = normalization_stats
    else:
        try:
            anchor_stats = load_normalization_stats()
        except FileNotFoundError as exc:
            raise ValueError(
                "predict needs the mg/dL last_bg anchor to call the model's "
                "forward, but normalization_stats was None and no on-disk "
                "normalization_stats.json was found. Pass normalization_stats "
                "explicitly (run `python normalization.py` to regenerate the "
                "stats file)."
            ) from exc

    # Build the (T, PATCH_DIM) patches tensor + mask, then add a batch axis of
    # size 1.  ``overrides`` announce the prediction-zone doses; un-overridden dose
    # slots get the ``normalize(0)`` no-dose baseline (via ``anchor_stats``), never z=0.
    patches, attn_mask = _build_patches_tensor(
        context, overrides=overrides, normalization_stats=anchor_stats,
    )
    patches = patches.unsqueeze(0).to(device)    # (1, T, PATCH_DIM)
    attn_mask = attn_mask.to(device)             # (T, T)

    last_bg = last_bg_mgdl_from_context(context, anchor_stats)
    last_bg_t = torch.tensor([float(last_bg)], dtype=torch.float32, device=device)

    with torch.no_grad():
        if return_time:
            q_tau, median, time_pred = model(
                patches, attn_mask, last_bg_t, return_time=True,
            )
        else:
            q_tau, median = model(patches, attn_mask, last_bg_t)

    # Drop the batch axis we added above.
    #   q_tau:  (PREDICTION_PATCHES, PATCH_SIZE, N_QUANTILES)
    #   median: (PREDICTION_PATCHES, PATCH_SIZE)
    q_tau = q_tau.squeeze(0)
    median = median.squeeze(0)
    assert q_tau.shape == (PREDICTION_PATCHES, PATCH_SIZE, N_QUANTILES), (
        f"q_tau shape {tuple(q_tau.shape)} != "
        f"{(PREDICTION_PATCHES, PATCH_SIZE, N_QUANTILES)}"
    )

    result: dict[str, torch.Tensor] = {
        'q_tau': q_tau,
        'median': median,
    }

    if return_time:
        # (PREDICTION_PATCHES, TIME_PROBE_N_BINS) raw logits, or None when disabled.
        # Decode/softmax stays in utils (single chokepoint) — emit raw here.
        result['time_pred'] = None if time_pred is None else time_pred.squeeze(0)

    if normalization_stats is not None:
        # (c)->(b): invert the risk-space head outputs to mg/dL.  ``f_inv`` is
        # the SOLE risk->mg/dL bridge and clamps to [BG_CLAMP_MIN, BG_CLAMP_MAX].
        result['median_bg'] = kovatchev_f_inv(median).flatten()   # (P*S,)
        bands = kovatchev_f_inv(q_tau)                             # (P, S, N_QUANTILES) mg/dL
        if conformal_delta is not None:
            # Split-conformal recalibration of the band edges (median untouched). The
            # per-(step, level) ``delta`` (P*S, N_QUANTILES) is fit on held-out data
            # (conformal.fit_quantile_conformal); ``apply`` keeps the fan monotone and
            # the median fixed.  ``None`` ⇒ raw bands (bit-identical).
            from conformal import apply_quantile_conformal
            bflat = bands.reshape(-1, N_QUANTILES).detach().cpu().numpy()
            bflat = apply_quantile_conformal(bflat, _conformal_to_np(conformal_delta), _MEDIAN_IDX)
            bands = torch.from_numpy(bflat.astype(np.float32)).to(bands.device).reshape(bands.shape)
        result['bands'] = bands
        result['last_bg'] = torch.tensor(float(last_bg), dtype=torch.float32)

    return result


def predict_origin_hour(
    model: T1DMAI,
    context: torch.Tensor,
    normalization_stats: dict[str, dict[str, float]] | None = None,
    device: torch.device | None = None,
) -> tuple[float, float]:
    """
    Decode the model's auxiliary time-of-day probe at the forecast origin.

    Shares ``predict``'s patch-building chokepoint exactly (``_build_patches_tensor``
    + ``last_bg_mgdl_from_context``), then runs a single ``return_time=True`` forward
    to read the ``(1, PREDICTION_PATCHES, TIME_PROBE_N_BINS)`` per-patch bin logits and
    decodes patch 0 (the forecast origin).  The forecast ``q_tau`` /
    ``median`` are computed identically to ``predict`` and discarded here — this is a
    read-only diagnostic that never touches the BG forecast.

    Args:
        model: T1DMAI model in eval mode.
        context: (n_ctx, PATCH_SIZE, N_INPUT_FEATURES) normalized context.
        normalization_stats: stats for the mg/dL ``last_bg`` anchor; falls back to
            the on-disk ``normalization_stats.json`` when ``None`` (as ``predict``
            does), raising if neither is available.
        device: Torch device (default: the model's device).

    Returns:
        ``(hour, R)`` — decoded origin hour-of-day in ``[0, 24)`` and the probe
        confidence ``R`` (resultant length in ``[0, 1]``; larger ⇒ more confident).
        Returns
        ``(nan, nan)`` when the probe is disabled (``TIME_PROBE_ENABLED`` is False,
        so the forward emits ``time_pred is None``).
    """
    if device is None:
        device = next(model.parameters()).device

    if normalization_stats is not None:
        anchor_stats = normalization_stats
    else:
        try:
            anchor_stats = load_normalization_stats()
        except FileNotFoundError as exc:
            raise ValueError(
                "predict_origin_hour needs the mg/dL last_bg anchor to call the "
                "model's forward, but normalization_stats was None and no on-disk "
                "normalization_stats.json was found. Pass normalization_stats "
                "explicitly (run `python normalization.py` to regenerate it)."
            ) from exc

    patches, attn_mask = _build_patches_tensor(
        context, overrides=None, normalization_stats=anchor_stats,
    )
    patches = patches.unsqueeze(0).to(device)    # (1, T, PATCH_DIM)
    attn_mask = attn_mask.to(device)             # (T, T)

    last_bg = last_bg_mgdl_from_context(context, anchor_stats)
    last_bg_t = torch.tensor([float(last_bg)], dtype=torch.float32, device=device)

    with torch.no_grad():
        _q_tau, _median, time_pred = model(
            patches, attn_mask, last_bg_t, return_time=True,
        )

    if time_pred is None:  # TIME_PROBE_ENABLED is False ⇒ probe head not built
        return float('nan'), float('nan')

    hours, R = time_of_day_decode_bins(time_pred[:, 0, :], TIME_PROBE_N_BINS)
    hour = float(hours.reshape(-1)[0].item())
    conf_r = float(R.reshape(-1)[0].item())
    return hour, conf_r


def predict_what_if(
    model: T1DMAI,
    context: torch.Tensor,
    patient_seed: int | None,
    overrides: dict[int, torch.Tensor],
    normalization_stats: dict[str, dict[str, float]] | None = None,
    device: torch.device | None = None,
    return_time: bool = False,
) -> dict[str, torch.Tensor]:
    """
    What-if prediction: announce carb / insulin in the prediction zone and run
    a forward pass conditioned on them.

    Override values must already be NORMALIZED (z-score / log1p) — use
    ``normalization.normalize`` if the caller has raw-unit values.

    Args:
        model: T1DMAI model in eval mode.
        context: (n_ctx, PATCH_SIZE, N_INPUT_FEATURES) normalized context.
        patient_seed: Unused (legacy positional arg; kept so existing callers
                   don't break).
        overrides: Dict mapping an output-channel index ({0: carb, 1: insulin})
                   to a (PREDICTION_PATCHES, PATCH_SIZE) tensor of normalized
                   values; routed to feat 1 / 2 via ``CHANNEL_TO_FEAT``.  BG
                   is never overrideable.
        normalization_stats: REQUIRED for the mg/dL ``last_bg`` anchor and the
                   ``median_bg`` / ``bands`` mg/dL fields.
        device: Torch device.
        return_time: opt-in diagnostic flag forwarded to ``predict`` — when
                   ``True`` the result gains the ``time_pred`` per-patch bin
                   logits key (default ``False`` ⇒ bit-identical).

    Returns:
        Same as ``predict`` but reflecting the conditioned carb/insulin.
    """
    # ``predict`` already routes overrides through ``CHANNEL_TO_FEAT`` and owns the
    # risk->mg/dL inversion, so the what-if path is just a conditioned ``predict``
    # call — no separate dynamics/physics reconstruction exists in the risk-space
    # design.
    return predict(
        model, context,
        patient_seed=patient_seed,
        normalization_stats=normalization_stats,
        device=device,
        overrides=overrides,
        return_time=return_time,
    )


def predict_rolling(
    model: T1DMAI,
    context: torch.Tensor,
    patient_seed: int | None = None,
    n_rolls: int = 3,
    normalization_stats: dict[str, dict[str, float]] | None = None,
    device: torch.device | None = None,
    overrides_fn: Any = None,
    conformal_delta: np.ndarray | None = None,
    return_time: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Autoregressive rolling prediction extending the horizon beyond one window.

    The re-feed is **BG-autoregressive only**.  Each roll:
      1. Predicts ``PREDICTION_PATCHES`` patches from current context.
      2. Inverts the risk-space median to a mg/dL BG forecast via
         ``kovatchev.f_inv`` (three-hop: median -> f_inv -> mg/dL).
      3. Builds new context patches: the mg/dL forecast is re-normalized into
         the ``bg_absolute`` slot 0 of each new context patch; carb (feat 1) /
         insulin (feat 2) come from the caller's ``overrides_fn`` if supplied,
         else are set to the **zero-RAW normalized baseline** (the z-score of a
         literal 0 g / 0 U through ``normalize`` — NOT ``torch.zeros``, whose
         z=0 decodes to a phantom ~0.39 g / ~0.14 U).  There are NO dynamics
         outputs to write back.
      4. Slides the context window forward, dropping oldest patches if it
         exceeds ``MAX_CONTEXT_PATCHES``.

    Uncertainty grows with each roll as forecast errors compound.  Each roll's
    band would otherwise *reset* to the model's near-flat init fan at the new
    context's last step, so the fan would sawtooth at every roll boundary.  To
    keep the fan monotone non-decreasing across boundaries we accumulate a
    running RISK-space half-width ``carry_spread`` — seeded each roll by the
    terminal-step half-width the model itself emitted — and widen the next
    roll's quantiles by it (the exact effect of ``assemble_quantiles``'
    ``carry_spread`` arg, applied post-forward since the head lives inside
    ``model.forward``).  The median is untouched; only the band edges fan out.

    Args:
        model: T1DMAI model in eval mode.
        context: (n_ctx, PATCH_SIZE, N_INPUT_FEATURES) initial context.
        patient_seed: Unused (legacy positional arg).
        n_rolls: Number of prediction rolls (each roll is one
            ``PREDICTION_PATCHES`` window).
        normalization_stats: REQUIRED.  Without stats the BG denormalization
            and re-normalization for context fill can't be computed.  Passing
            ``None`` raises ``ValueError``.
        device: Torch device.
        overrides_fn: optional callable
            ``(roll_idx: int, base_mu_np: np.ndarray, abs_n_ctx: int) ->
            tuple[dict[int, np.ndarray], dict[int, np.ndarray]] | None``.
            Invoked each roll.  If it returns ``(overrides_norm, overrides_raw)``
            (each a ``{ch_idx: (PREDICTION_PATCHES, PATCH_SIZE) array}`` over
            output channels {0: carb, 1: insulin}), the roll is re-predicted
            conditioned on ``overrides_norm`` and those carb/insulin values are
            written into the next window's context.  Returning ``None`` ⇒ no
            announced doses this roll (the zero-RAW baseline is used).  The
            ``base_mu_np`` arg is kept for backward-compatible signatures and is
            the empty/zero placeholder this design has no dynamics μ for;
            ``abs_n_ctx`` is the absolute patch index at the start of this
            roll's prediction zone.
        return_time: opt-in diagnostic flag.  When ``True`` the result gains a
            ``time_pred`` key holding roll 0's per-patch time-of-day bin logits
            (roll 0 is the sole true wall-clock origin; later rolls are synthetic
            BG re-feeds, so their probe read would be meaningless).  Default
            ``False`` ⇒ inner forwards stay ``return_time=False`` ⇒ bit-identical.

    Returns:
        result dict::
            pred_bg:  (n_rolls * PREDICTION_PATCHES * PATCH_SIZE,) predicted BG
                      trajectory (mg/dL), ``f_inv(median)`` per roll concatenated.
            q_tau:    (n_rolls * PREDICTION_PATCHES, PATCH_SIZE, N_QUANTILES)
                      concatenated risk-space quantiles.
            bands:    (n_rolls * PREDICTION_PATCHES, PATCH_SIZE, N_QUANTILES)
                      concatenated mg/dL band edges.
            time_pred: (PREDICTION_PATCHES, TIME_PROBE_N_BINS) roll-0 time-of-day
                      probe bin logits, or ``None`` (probe disabled) — present ONLY
                      when ``return_time=True``.
    """
    if normalization_stats is None:
        raise ValueError(
            "predict_rolling requires normalization_stats: the renormalized BG "
            "inputs for each new context patch depend on it."
        )
    if device is None:
        device = next(model.parameters()).device

    current_context = context.clone()
    all_pred_bgs: list[torch.Tensor] = []
    all_q_tau: list[torch.Tensor] = []
    all_bands: list[torch.Tensor] = []

    bg_mean = normalization_stats['bg_absolute']['mean']
    bg_std = normalization_stats['bg_absolute']['std']

    # Zero-RAW normalized baseline for the re-fed carb (feat 1 / carb_intake) and
    # insulin (feat 2 / insulin_combined) channels.  ``normalize`` of a literal 0
    # routes each sparse channel through ``log1p`` first, so the baseline is the
    # channel's ``-mean/std`` — NOT z=0 (which decodes to a phantom dose).  feat 0
    # (bg) is overwritten by the BG re-feed below, so its baseline value is unused.
    zero_raw = normalize(
        np.zeros((1, len(CHANNEL_NAMES)), dtype=np.float32), normalization_stats,
    )[0]                                                  # (n_channels,) z-space
    carb_feat = CHANNEL_TO_FEAT[0]                        # carb  -> feat 1
    insulin_feat = CHANNEL_TO_FEAT[1]                     # insulin -> feat 2
    carb_baseline_z = float(zero_raw[carb_feat])
    insulin_baseline_z = float(zero_raw[insulin_feat])

    n_ctx_orig = context.shape[0]

    # Running RISK-space band half-width carried across roll boundaries so the
    # fan does not sawtooth-reset.  ``_MEDIAN_IDX`` is the median column in the
    # ascending τ fan.
    carry_spread = 0.0

    # Roll 0 is the only true wall-clock origin — the diagnostic time-of-day probe
    # is read there and nowhere else (later rolls re-feed synthetic BG).
    time_pred_roll0: torch.Tensor | None = None

    for roll_idx in range(n_rolls):
        # Resolve conditioning BEFORE the forward so each roll runs a SINGLE
        # forward, not two.  The override callback does NOT depend on any
        # prediction result — ``base_mu`` is a fixed zero placeholder and the
        # other args are loop indices — so we can decide first, then dispatch
        # once: ``predict_what_if`` when the caller announced carb/insulin
        # (conditioned every roll, not just the first), else the plain
        # ``predict``.  ``base_mu`` has no meaning in the risk-space design (no
        # dynamics μ); the zero placeholder keeps legacy callbacks working.
        overrides_norm: dict[int, np.ndarray] | None = None
        torch_overrides: dict[int, torch.Tensor] | None = None
        if overrides_fn is not None:
            abs_n_ctx = n_ctx_orig + roll_idx * PREDICTION_PATCHES
            base_mu_placeholder = np.zeros(
                (PREDICTION_PATCHES, PATCH_SIZE, 0), dtype=np.float32
            )
            ov = overrides_fn(roll_idx, base_mu_placeholder, abs_n_ctx)
            if ov is not None:
                overrides_norm, _overrides_raw = ov
                if overrides_norm:
                    torch_overrides = {
                        ch: torch.from_numpy(v.astype(np.float32))
                        for ch, v in overrides_norm.items()
                    }

        roll_return_time = return_time and roll_idx == 0
        if torch_overrides is not None:
            result = predict_what_if(
                model, current_context, patient_seed,
                overrides=torch_overrides,
                normalization_stats=normalization_stats,
                device=device,
                return_time=roll_return_time,
            )
        else:
            result = predict(
                model, current_context, patient_seed,
                normalization_stats=normalization_stats, device=device,
                return_time=roll_return_time,
            )
        if roll_return_time:
            time_pred_roll0 = result.get('time_pred')

        q_tau = result['q_tau']            # (PREDICTION_PATCHES, PATCH_SIZE, N_QUANTILES) risk
        pred_bg_roll = result['median_bg'].to(device)  # (PREDICTION_PATCHES * PATCH_SIZE,)

        # THIS roll's NATIVE terminal-step half-width — the spread the model itself
        # emitted at the final forecast step (RISK space), measured BEFORE adding
        # any carry.  Measuring it on the POST-carry fan re-counts the carry and
        # makes it compound geometrically (carry → 2·carry + native each roll), the
        # runaway that pinned the long-horizon band to the physiological range.
        native_last = q_tau[-1, -1]        # (N_QUANTILES,) native risk quantiles
        native_halfwidth = float(
            (native_last[-1] - native_last[0]).clamp_min(0.0) * 0.5
        )

        # Widen the RISK-space fan by the accumulated carry so this roll's band
        # picks up where the previous roll's left off (no boundary reset).  The
        # carry is purely additive on the cumsum base, exactly mirroring
        # ``assemble_quantiles(carry_spread=...)``: τ>.5 edges shift up by the
        # carry, τ<.5 edges shift down by it, the median (idx ``_MEDIAN_IDX``) is
        # untouched.  ``bands`` is then re-derived from the widened risk fan.
        if carry_spread > 0.0:
            q_tau = q_tau.clone()
            q_tau[..., _MEDIAN_IDX + 1:] = q_tau[..., _MEDIAN_IDX + 1:] + carry_spread
            q_tau[..., :_MEDIAN_IDX] = q_tau[..., :_MEDIAN_IDX] - carry_spread
        bands = kovatchev_f_inv(q_tau)     # (PREDICTION_PATCHES, PATCH_SIZE, N_QUANTILES) mg/dL
        if conformal_delta is not None:
            # Split-conformal recalibration of THIS roll's bands (median untouched), on
            # top of the carry-spread widening.  ``None`` ⇒ raw (bit-identical).
            from conformal import apply_quantile_conformal
            _bf = bands.reshape(-1, N_QUANTILES).detach().cpu().numpy()
            _bf = apply_quantile_conformal(_bf, _conformal_to_np(conformal_delta), _MEDIAN_IDX)
            bands = torch.from_numpy(_bf.astype(np.float32)).to(bands.device).reshape(bands.shape)

        # Accumulate the NATIVE per-roll half-width so the carry grows ~linearly
        # with the number of rolls (a conservative random-walk band) rather than
        # doubling — the next roll's fan starts from where this one ended.
        carry_spread += native_halfwidth

        all_q_tau.append(q_tau)
        all_bands.append(bands)
        all_pred_bgs.append(pred_bg_roll)

        # Build new context patches (P, S, N_INPUT_FEATURES).  Three-hop BG re-feed: median
        # forecast was already inverted to mg/dL (``pred_bg_roll``); re-normalize
        # it into the bg_absolute slot 0.  carb (feat 1) / insulin (feat 2) default
        # to the zero-RAW normalized baseline (a literal 0 g / 0 U through
        # ``normalize``, NOT z=0 — z=0 decodes to a phantom dose); the override (if
        # any) writes real values over them.  No temporal features, no dynamics.
        new_ctx_patches = torch.zeros(PREDICTION_PATCHES, PATCH_SIZE, N_INPUT_FEATURES)
        new_ctx_patches[:, :, carb_feat] = carb_baseline_z
        new_ctx_patches[:, :, insulin_feat] = insulin_baseline_z

        # BG slot 0 from the (mg/dL) median forecast, re-normalized into the same
        # risk space feat 0 was trained in: bg feat 0 is ``z(f(bg))``, so apply the
        # Kovatchev f before the z-score (``pred_bg_np`` is already the clamped
        # ``f_inv(median)`` mg/dL, so f is well-defined).  ``RISK_SPACE_CHANNELS``
        # is always-on for bg — no mg/dL-input branch.
        pred_bg_np = pred_bg_roll.detach().cpu().numpy().reshape(
            PREDICTION_PATCHES, PATCH_SIZE
        )
        bg_input = kovatchev_f_np(pred_bg_np)
        bg_norm = (bg_input - bg_mean) / (bg_std + 1e-8)
        new_ctx_patches[:, :, 0] = torch.from_numpy(bg_norm.astype(np.float32))

        # carb (feat 1) / insulin (feat 2) from the announced overrides, if any —
        # written straight into the appended context patches (they become observed
        # history for the next roll).  Without an override they keep the zero-RAW
        # baseline set above.
        if overrides_norm:
            for ch_idx, norm_vals in overrides_norm.items():
                if ch_idx not in CHANNEL_TO_FEAT:
                    continue
                feat_idx = CHANNEL_TO_FEAT[ch_idx]
                new_ctx_patches[:, :, feat_idx] = torch.from_numpy(
                    norm_vals.astype(np.float32)
                )

        # Extend context (or slide window if at max) so the context never
        # exceeds MAX_CONTEXT_PATCHES (keeps the attention mask shape valid).
        new_context = torch.cat([current_context, new_ctx_patches], dim=0)
        if new_context.shape[0] > MAX_CONTEXT_PATCHES:
            new_context = new_context[-MAX_CONTEXT_PATCHES:]
        current_context = new_context

    result_out: dict[str, torch.Tensor] = {
        'pred_bg': torch.cat(all_pred_bgs, dim=0),
        'q_tau': torch.cat(all_q_tau, dim=0),
        'bands': torch.cat(all_bands, dim=0),
    }
    if return_time:
        result_out['time_pred'] = time_pred_roll0
    return result_out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='T1DMAI Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--seed', type=int, default=42,
                        help='Patient seed for simulation')
    parser.add_argument('--use-ema', action='store_true',
                        help='Load the EMA shadow weights from the checkpoint '
                             '(smoother, used at validation) instead of the '
                             'live training weights.  Errors out if the '
                             'checkpoint has no EMA state.')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model.  ``weights_only=True`` is the secure-load path — refuses to
    # unpickle arbitrary Python objects.
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)

    model = T1DMAI().to(device)
    if args.use_ema:
        if not ckpt.get('model_ema_state_dict'):
            raise RuntimeError(
                f"--use-ema was requested but the checkpoint at "
                f"{args.checkpoint!r} has no 'model_ema_state_dict'. "
                f"Either drop --use-ema or load a checkpoint trained with "
                f"EMA_DECAY > 0."
            )
        live_sd = ckpt['model_state_dict']
        ema_sd = ckpt['model_ema_state_dict']
        # EMA only tracks float tensors; fall back to live for the rest
        # (int buffers, attention masks, anything non-float).
        merged = {k: ema_sd.get(k, v) for k, v in live_sd.items()}
        model.load_state_dict(merged, strict=True)
        print("Loaded EMA weights from checkpoint.")
    else:
        model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    norm_stats = ckpt.get('normalization_stats') or load_normalization_stats()

    # Generate a context window from the simulator.
    from T1DMSIM.simulator import T1DMSimulator
    from data import simulate_discard_warmup

    sim = T1DMSimulator(seed=args.seed)
    # Max context window (MAX_CONTEXT_PATCHES × 30 min) after the
    # SIMULATOR_WARMUP_HOURS warmup drop handled by simulate_discard_warmup.
    # The model accepts anything in [MIN_CONTEXT_PATCHES, MAX_CONTEXT_PATCHES];
    # the smoke-test just exercises the ceiling.
    raw = simulate_discard_warmup(sim, 24)

    # Pull each retained channel from the simulator dict.  Use ``bg_observed``
    # (post-CGM-noise) so this matches training; the clean ``bg`` is not what the
    # model is normalized against.  IS / HGO / bg_delta and the temporal sin/cos
    # features are NO LONGER inputs (dropped from the 3-feature stack), so they are
    # not gathered here.  The model consumes the RAW post-noise signals (no
    # smoothing), mirroring ``data._build_sample`` (bg clamped to the physical BG
    # range so it is a legal Kovatchev-f argument, sparse carb/insulin floored at 0).
    from T1DMSIM.simulator import BG_CLAMP_MIN, BG_CLAMP_MAX
    bg_obs = np.clip(raw['bg_observed'], BG_CLAMP_MIN, BG_CLAMP_MAX).astype(np.float32)
    carb = np.maximum(raw['total_carb'], 0.0).astype(np.float32)
    insulin = np.maximum(raw['total_insulin'], 0.0).astype(np.float32)

    # 3-feature stack: [bg_absolute, carbs, insulin] — bg_delta / IS / HGO and the
    # temporal sin/cos features are dropped.
    assert list(CHANNEL_NAMES) == ['bg_absolute', 'carb_intake', 'insulin_combined'], (
        f"unexpected CHANNEL_NAMES {list(CHANNEL_NAMES)}"
    )
    raw_features = np.stack([bg_obs, carb, insulin], axis=-1)  # (N, N_INPUT_FEATURES)
    # ``normalize`` applies the Kovatchev ``f`` to bg (via RISK_SPACE_CHANNELS) and
    # log1p to carb/insulin BEFORE the z-score — the SAME path data.py uses, so bg
    # feat 0 lands in z(f(bg)) space.  Don't hand-roll the per-channel transform.
    features = normalize(raw_features, norm_stats)

    # Trim to a multiple of PATCH_SIZE so reshape() is clean.
    N = (len(features) // PATCH_SIZE) * PATCH_SIZE
    features = features[:N]
    context_t = torch.from_numpy(features).reshape(-1, PATCH_SIZE, N_INPUT_FEATURES)

    result = predict(model, context_t, patient_seed=args.seed,
                     normalization_stats=norm_stats, device=device)

    print(f"q_tau shape: {result['q_tau'].shape}")
    print(f"median_bg range: [{result['median_bg'].min():.1f}, {result['median_bg'].max():.1f}] mg/dL")
    print(f"bands shape: {result['bands'].shape}")
