"""T1DMAI inference: predict / predict_what_if / predict_rolling.

Three spaces per SPEC/inference.md: (a) z-space input, (b) mg/dL physical, (c) risk
space (head output). normalize/denormalize bridges (a)<->(b); kovatchev_f/f_inv bridges
(b)<->(c); this module owns the (c)->(b) step."""

import argparse
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from config import (
    PREDICTION_PATCHES, PATCH_SIZE, N_INPUT_FEATURES, N_QUANTILES, N_SPREADS,
    MAX_CONTEXT_PATCHES, MAX_MASKED_PATCHES,
    CHANNEL_TO_FEAT, MASKABLE_FEATS, NON_MASKABLE_FEATS, QUANTILE_LEVELS,
    TIME_PROBE_N_BINS,
)
# Slot expansion, anchor rule, bg_masked index: single definition in data.py — never re-derive.
from data import BG_MASKED_FEAT, _mask_slots
from model import T1DMAI
from normalization import (
    load_normalization_stats, CHANNEL_NAMES, normalize,
)
from utils import (
    create_attention_mask_from_visible,
    last_bg_mgdl_from_context, kovatchev_f_inv, kovatchev_f_np,
    time_of_day_decode_bins,
)

# τ=0.5 median index in the ascending fan; conformal recalibration holds this column fixed.
_MEDIAN_IDX = QUANTILE_LEVELS.index(0.5)

# One masked span as (start_patch, length) over the window's patch axis (data.sample_mask_spans).
MaskSpans = Sequence[tuple[int, int]]


def _conformal_to_np(delta: Any) -> np.ndarray:
    """Conformal delta → host numpy; ``apply_quantile_conformal`` is pure numpy."""
    if torch.is_tensor(delta):
        return delta.detach().cpu().numpy()
    return np.asarray(delta)


def _resolve_mask_spans(mask_spans: MaskSpans | None, n_ctx: int) -> list[tuple[int, int]]:
    """Validate a masked set over the window, or build the default trailing forecast.

    Rules: spans strictly increase and never abut (one visible separator patch is required
    between spans); sum(length) <= MAX_MASKED_PATCHES; every patch of [n_ctx, T) is masked
    (no visible future BG); at least one patch stays visible so every span has an anchor."""
    seq_len = n_ctx + PREDICTION_PATCHES
    if mask_spans is None:
        return [(n_ctx, PREDICTION_PATCHES)]
    spans = [(int(s), int(L)) for s, L in mask_spans]
    assert spans, "mask_spans is empty — the head must be given at least one masked patch"
    total = sum(L for _s, L in spans)
    assert total <= MAX_MASKED_PATCHES, (
        f"masked set of {total} patches exceeds the head's "
        f"MAX_MASKED_PATCHES={MAX_MASKED_PATCHES} slots"
    )
    prev_end = -1
    for start, length in spans:
        assert length >= 1, f"span length must be >= 1, got {length}"
        assert 0 <= start and start + length <= seq_len, (
            f"span ({start}, {length}) leaves the {seq_len}-patch window"
        )
        assert start > prev_end, (
            f"masked spans {spans} abut or overlap — one mandatory visible patch "
            f"must separate neighbours"
        )
        prev_end = start + length          # a separator patch sits at prev_end
    masked = {p for start, length in spans for p in range(start, start + length)}
    future = set(range(n_ctx, seq_len))
    assert future <= masked, (
        f"future patches {sorted(future - masked)} are not masked — the prediction "
        f"zone carries no observed BG, so leaving it visible announces z = 0 as a "
        f"reading"
    )
    assert len(masked) < seq_len, "the whole window is masked — no visible evidence left"
    return spans


def _anchor_cells(
    mask_idx: np.ndarray, valid: np.ndarray, anchor_step: np.ndarray, n_ctx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """(patch_idx, step_idx) of each head slot's anchor cell in the context.

    Anchor is ONE-SIDED, LEFT-PREFERRING: last step of the left neighbour, or first step of
    the right neighbour only when the span starts at patch 0; same value for a whole span.
    Padded slots take slot 0's cell (legal mg/dL); every anchor cell is a VISIBLE context cell."""
    patch_idx = anchor_step // PATCH_SIZE
    step_idx = anchor_step % PATCH_SIZE
    patch_idx = np.where(valid, patch_idx, patch_idx[0])
    step_idx = np.where(valid, step_idx, step_idx[0])
    assert bool((patch_idx < n_ctx).all()), (
        f"anchor patch {patch_idx.max()} is outside the {n_ctx}-patch context — "
        f"its bg is not observed"
    )
    assert not bool(np.isin(patch_idx, mask_idx[valid]).any()), (
        f"anchor patches {patch_idx.tolist()} intersect the masked set "
        f"{mask_idx[valid].tolist()} — a masked cell decodes to a plausible "
        f"anchor rather than an error"
    )
    return patch_idx, step_idx


def _assert_mask_announced(
    patches: torch.Tensor, mask_idx: torch.Tensor, valid: torch.Tensor,
) -> None:
    """Assert feat 4 of ``patches`` reproduces the requested masked set, exactly.
    Run before EVERY forward: unwritten by MASKABLE_FEATS/NON_MASKABLE_FEATS, so a forgetful
    builder leaves it 0.0 (forecast zone announced as observed). Checks the bit is 0.0/1.0,
    uniform across PATCH_SIZE columns, and matches the requested set (valid slots only)."""
    bits = patches[..., BG_MASKED_FEAT::N_INPUT_FEATURES]        # (B, T, PATCH_SIZE)
    assert bits.shape[-1] == PATCH_SIZE, (
        f"feat {BG_MASKED_FEAT} spans {bits.shape[-1]} columns, expected "
        f"PATCH_SIZE={PATCH_SIZE} — the row layout is not step-major"
    )
    assert bool(((bits == 0.0) | (bits == 1.0)).all()), (
        f"bg_masked is a BIT: feat {BG_MASKED_FEAT} must hold only 0.0 or 1.0"
    )
    assert bool((bits == bits[..., :1]).all()), (
        f"the bg_masked bit is per PATCH — all {PATCH_SIZE} columns of feat "
        f"{BG_MASKED_FEAT} must agree"
    )
    announced = bits[..., 0] > 0.5                               # (B, T)
    requested = torch.zeros_like(announced)
    batch = torch.arange(
        mask_idx.shape[0], device=mask_idx.device
    ).unsqueeze(1).expand_as(mask_idx)
    requested[batch[valid], mask_idx[valid]] = True
    assert torch.equal(announced, requested), (
        f"feat {BG_MASKED_FEAT} announces patches "
        f"{announced.nonzero().tolist()} but the requested masked set is "
        f"{requested.nonzero().tolist()}"
    )


def _build_patches_tensor(
    context: torch.Tensor,
    overrides: dict[int, torch.Tensor] | None = None,
    normalization_stats: dict[str, dict[str, float]] | None = None,
    mask_spans: MaskSpans | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(T, PATCH_DIM) patches tensor and (T, T) attn mask for one masked set.
    Un-overridden dose slots use normalize(0) (needs normalization_stats), NOT the z=0
    sentinel, which decodes to a phantom dose. Overrides route via CHANNEL_TO_FEAT; BG is
    never overrideable; mask_spans=None selects the trailing forecast span."""
    assert context.ndim == 3 and context.shape[-1] == N_INPUT_FEATURES, (
        f"context must be (n_ctx, PATCH_SIZE, {N_INPUT_FEATURES}), got {tuple(context.shape)}"
    )
    n_ctx = context.shape[0]
    seq_len = n_ctx + PREDICTION_PATCHES
    spans = _resolve_mask_spans(mask_spans, n_ctx)

    # Flatten (PATCH_SIZE, N_INPUT_FEATURES) -> PATCH_DIM; feature values pass through verbatim.
    ctx_patches = context.reshape(n_ctx, PATCH_SIZE * N_INPUT_FEATURES)  # (n_ctx, PATCH_DIM)

    # bg stays 0 (predicted); MASKABLE_FEATS dose slots seed normalize(0), not z=0 (phantom dose).
    pred_features = torch.zeros(PREDICTION_PATCHES, PATCH_SIZE, N_INPUT_FEATURES)
    if normalization_stats is not None:
        zero_raw = normalize(
            np.zeros((1, len(CHANNEL_NAMES)), dtype=np.float32), normalization_stats,
        )[0]
        for feat_idx in MASKABLE_FEATS:
            pred_features[:, :, feat_idx] = float(zero_raw[feat_idx])
    pred_patches = pred_features.reshape(PREDICTION_PATCHES, PATCH_SIZE * N_INPUT_FEATURES)

    # CHANNEL_TO_FEAT routes ch_idx to its feature slot, same mapping data.py uses.
    if overrides:
        for ch_idx, override_vals in overrides.items():
            if ch_idx not in CHANNEL_TO_FEAT:
                continue
            feat_idx = CHANNEL_TO_FEAT[ch_idx]
            # override_vals: (PREDICTION_PATCHES, PATCH_SIZE)
            for t in range(PATCH_SIZE):
                flat_col = t * N_INPUT_FEATURES + feat_idx
                pred_patches[:, flat_col] = override_vals[:, t]

    # ``cat`` allocates, so the writes below never reach back into the caller's context.
    patches = torch.cat([ctx_patches, pred_patches], dim=0)  # (T, PATCH_DIM)

    # Withhold bg then announce it; slot expansion via data._mask_slots, shared with training.
    mask_idx, valid, _d, _anchor_step = _mask_slots(spans, seq_len)
    masked_rows = torch.from_numpy(mask_idx[valid])
    # Zeroes bg on masked CONTEXT patches (backcast/infill); carb/insulin/exercise pass through.
    for feat_idx in NON_MASKABLE_FEATS:
        patches[masked_rows, feat_idx::N_INPUT_FEATURES] = 0.0
    # feat 4 spans PATCH_SIZE step-major cols; unwritten, masked patches announce as observed.
    patches[:, BG_MASKED_FEAT::N_INPUT_FEATURES] = 0.0
    patches[masked_rows, BG_MASKED_FEAT::N_INPUT_FEATURES] = 1.0

    # Built from visible/masked labels, not position; single sample so (1,T,T) collapses to (T,T).
    visible = torch.ones(1, seq_len, dtype=torch.bool)
    visible[0, masked_rows] = False
    attn_mask = create_attention_mask_from_visible(visible)[0]

    return patches, attn_mask


def _run_forward(
    model: T1DMAI,
    context: torch.Tensor,
    anchor_stats: dict[str, dict[str, float]],
    overrides: dict[int, torch.Tensor] | None = None,
    mask_spans: MaskSpans | None = None,
    device: torch.device | None = None,
    return_time: bool = False,
    grad: bool = False,
    return_crossing: bool = False,
) -> dict[str, Any]:
    """Build one sample, announce its masked set, check it, then forward.
    Sole chokepoint for every forward here: the feat-4 assert always runs, and anchors cross
    the denormalize bridge (last_bg_mgdl_from_context) exactly once per call. grad=True builds
    a live autograd graph for attribution; the forward VALUE is unchanged either way."""
    if device is None:
        device = next(model.parameters()).device
    n_ctx = int(context.shape[0])
    spans = _resolve_mask_spans(mask_spans, n_ctx)
    seq_len = n_ctx + PREDICTION_PATCHES

    patches, attn_mask = _build_patches_tensor(
        context, overrides=overrides, normalization_stats=anchor_stats,
        mask_spans=spans,
    )
    patches = patches.unsqueeze(0).to(device)    # (1, T, PATCH_DIM)
    attn_mask = attn_mask.to(device)             # (T, T)

    mask_idx, valid, _d, anchor_step = _mask_slots(spans, seq_len)
    anchor_patch, anchor_within = _anchor_cells(mask_idx, valid, anchor_step, n_ctx)
    # M anchors + edge (-1,-1), one transfer; edge = last_bg = bg_window[n_ctx*PATCH_SIZE-1].
    cells_p = np.concatenate([anchor_patch, np.array([-1], dtype=np.int64)])
    cells_s = np.concatenate([anchor_within, np.array([-1], dtype=np.int64)])
    anchors = last_bg_mgdl_from_context(
        context, anchor_stats, patch_idx=cells_p, step_idx=cells_s,
    )                                            # (M + 1,) mg/dL
    M = mask_idx.shape[0]
    anchor_bg = anchors[:M].to(device).unsqueeze(0).float()          # (1, M) mg/dL
    last_bg = float(anchors[M].item())
    mask_idx_t = torch.from_numpy(mask_idx).to(device).unsqueeze(0)  # (1, M) int64
    valid_t = torch.from_numpy(valid).to(device).unsqueeze(0)        # (1, M) bool

    # feat 4 must match the requested masked set — checked here, not trusted from the builder.
    _assert_mask_announced(patches, mask_idx_t, valid_t)

    if grad:
        patches.requires_grad_(True)
        out = model(
            patches, attn_mask, anchor_bg, mask_idx_t,
            return_time=return_time, return_crossing=return_crossing,
        )
    else:
        with torch.no_grad():
            out = model(
                patches, attn_mask, anchor_bg, mask_idx_t,
                return_time=return_time, return_crossing=return_crossing,
            )
    q_tau, median = out[0], out[1]
    time_pred = out[2] if return_time else None
    crossing = out[3] if return_crossing else None

    return {
        'q_tau': q_tau.squeeze(0),                          # (M, PATCH_SIZE, N_QUANTILES)
        'median': median.squeeze(0),                        # (M, PATCH_SIZE)
        'time_pred': None if time_pred is None else time_pred.squeeze(0),
        'crossing': None if crossing is None else crossing.squeeze(0),  # (M, S, 2) logits
        'mask_idx': mask_idx_t.squeeze(0),                  # (M,) patch index per slot
        'valid': valid_t.squeeze(0),                        # (M,) bool
        'anchor_bg': anchor_bg.squeeze(0),                  # (M,) mg/dL
        'last_bg': last_bg,
        'patches': patches,                                 # (1, T, PATCH_DIM)
        'anchor_patch': anchor_patch,                       # (M,) context patch of each anchor
        'anchor_within': anchor_within,                     # (M,) step inside that patch
    }


def predict(
    model: T1DMAI,
    context: torch.Tensor,
    patient_seed: int | None = None,
    normalization_stats: dict[str, dict[str, float]] | None = None,
    device: torch.device | None = None,
    overrides: dict[int, torch.Tensor] | None = None,
    conformal_delta: np.ndarray | None = None,
    return_time: bool = False,
    mask_spans: MaskSpans | None = None,
    return_crossing: bool = False,
) -> dict[str, torch.Tensor]:
    """Standard prediction: one forward pass over one masked set.
    RISK-space quantiles/median, inverted to mg/dL via kovatchev_f_inv. Default masked set
    is the trailing PREDICTION_PATCHES zone (a forecast). normalization_stats=None returns
    only raw q_tau/median and falls back to the on-disk file, raising if absent."""
    del patient_seed  # unused by the model
    if device is None:
        device = next(model.parameters()).device

    # Stats first: needed for the last_bg anchor and the normalize(0) no-dose baseline (not z=0).
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

    # overrides announce prediction-zone doses; unset dose slots take normalize(0), never z=0.
    out = _run_forward(
        model, context, anchor_stats, overrides=overrides,
        mask_spans=mask_spans, device=device, return_time=return_time,
        return_crossing=return_crossing,
    )

    # Keep VALID slots only — padded slots gather patch 0, a plausible forecast nobody asked for.
    valid = out['valid']
    q_tau = out['q_tau'][valid]                  # (P, PATCH_SIZE, N_QUANTILES)
    median = out['median'][valid]                # (P, PATCH_SIZE)
    mask_idx = out['mask_idx'][valid]            # (P,)
    last_bg = out['last_bg']
    n_masked = int(valid.sum().item())
    assert q_tau.shape == (n_masked, PATCH_SIZE, N_QUANTILES), (
        f"q_tau shape {tuple(q_tau.shape)} != "
        f"{(n_masked, PATCH_SIZE, N_QUANTILES)}"
    )

    result: dict[str, torch.Tensor] = {
        'q_tau': q_tau,
        'median': median,
        'mask_idx': mask_idx,
    }

    if return_time:
        # (P, TIME_PROBE_N_BINS) raw logits or None; decode/softmax stays in utils, emit raw here.
        time_pred = out['time_pred']
        result['time_pred'] = None if time_pred is None else time_pred[valid]

    if return_crossing:
        # (P, PATCH_SIZE, 2) cumulative crossing PROBABILITIES: col 0 hypo, col 1 hyper; None off.
        crossing = out['crossing']
        result['crossing'] = None if crossing is None else torch.sigmoid(crossing[valid].float())

    if normalization_stats is not None:
        # (c)->(b): f_inv is the SOLE risk->mg/dL bridge, clamped to [BG_CLAMP_MIN, BG_CLAMP_MAX].
        result['median_bg'] = kovatchev_f_inv(median).flatten()   # (P*S,)
        bands = kovatchev_f_inv(q_tau)                             # (P, S, N_QUANTILES) mg/dL
        if conformal_delta is not None:
            # Recalibrate band edges (median untouched); delta fit on held-out data, None = raw.
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
    """Decode the auxiliary time-of-day probe at the forecast origin.
    Shares predict's forward chokepoint (_run_forward); slot 0 of the per-slot bin logits is
    the origin patch since the masked set is the trailing forecast. normalization_stats=None
    falls back to the on-disk file, raising if absent. Returns (hour, R) in [0,24)x[0,1], or
    (nan, nan) when TIME_PROBE_ENABLED is False."""
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

    out = _run_forward(
        model, context, anchor_stats, overrides=None, mask_spans=None,
        device=device, return_time=True,
    )
    time_pred = out['time_pred']                 # (M, TIME_PROBE_N_BINS) or None

    if time_pred is None:  # TIME_PROBE_ENABLED is False ⇒ probe head not built
        return float('nan'), float('nan')

    # Slot 0 of the trailing forecast is the origin patch, valid by construction.
    hours, R = time_of_day_decode_bins(time_pred[:1, :], TIME_PROBE_N_BINS)
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
    return_crossing: bool = False,
) -> dict[str, torch.Tensor]:
    """What-if prediction: announce carb / insulin / exercise in the prediction zone.
    overrides must already be NORMALIZED (normalization.normalize), routed to feat 1/2/3 via
    CHANNEL_TO_FEAT; BG is never overrideable. normalization_stats is REQUIRED for the
    mg/dL last_bg anchor and median_bg/bands. Returns the same as predict, reflecting the
    conditioned channels."""
    # predict already routes overrides and owns risk->mg/dL; this is just a conditioned call.
    return predict(
        model, context,
        patient_seed=patient_seed,
        normalization_stats=normalization_stats,
        device=device,
        overrides=overrides,
        return_time=return_time,
        return_crossing=return_crossing,
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
    return_rolls: bool = False,
) -> dict[str, torch.Tensor]:
    """Autoregressive rolling prediction, extending the horizon beyond one window.

    Re-feed is BG-AUTOREGRESSIVE ONLY: f_inv(median) mg/dL -> normalize -> bg_absolute slot 0;
    carb/insulin/exercise from overrides_fn or ZERO-RAW baseline (normalize(0), NOT torch.zeros).
    carry_spread accumulates PER LEVEL in QUADRATURE, never additive/shared; median untouched."""
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

    # Zero-RAW baseline via normalize(0) per MASKABLE_FEATS — NOT z=0, a phantom dose/session.
    zero_raw = normalize(
        np.zeros((1, len(CHANNEL_NAMES)), dtype=np.float32), normalization_stats,
    )[0]                                                  # (n_channels,) z-space
    baseline_z = {feat: float(zero_raw[feat]) for feat in MASKABLE_FEATS}

    n_ctx_orig = context.shape[0]

    # RISK-space carry, PER LEVEL (SPEC/inference.md §8.1); one shared scalar flattens the fan.
    carry_spread: "torch.Tensor | None" = None

    # Roll 0 is the only wall-clock origin — probe reads it only, later rolls re-feed synthetic BG.
    time_pred_roll0: torch.Tensor | None = None

    roll_inputs: list[dict[str, Any]] = []

    for roll_idx in range(n_rolls):
        # Patch index where this roll's prediction zone starts; hoisted so it isn't duplicated.
        abs_n_ctx = n_ctx_orig + roll_idx * PREDICTION_PATCHES
        # Resolve overrides before the forward; base_mu is a zero placeholder for legacy callbacks.
        overrides_norm: dict[int, np.ndarray] | None = None
        torch_overrides: dict[int, torch.Tensor] | None = None
        if overrides_fn is not None:
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

        if return_rolls:
            # Captured before the forward/slide; current_context is rebound, not mutated.
            roll_inputs.append({
                'context': current_context,
                'overrides': torch_overrides,
                'offset': abs_n_ctx - int(current_context.shape[0]),
            })

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

        # Native per-level terminal spread, risk space, measured pre-carry (else compounds).
        native_last = q_tau[-1, -1]        # (N_QUANTILES,) native risk quantiles
        native_med = native_last[_MEDIAN_IDX]
        native = torch.cat([
            (native_last[_MEDIAN_IDX + 1:] - native_med).clamp_min(0.0),          # .75/.9/.95
            (native_med - native_last[:_MEDIAN_IDX]).clamp_min(0.0).flip(0),      # .25/.1/.05
        ])                                 # (2*N_SPREADS,) the carry's own layout

        # Widen the fan per level in quadrature with carry (assemble_quantiles' rule); median fixed.
        if carry_spread is not None:
            q_tau = q_tau.clone()
            m_col = q_tau[..., _MEDIAN_IDX].unsqueeze(-1)
            up_off = (q_tau[..., _MEDIAN_IDX + 1:] - m_col).clamp_min(0.0)
            dn_off = (m_col - q_tau[..., :_MEDIAN_IDX]).clamp_min(0.0)
            q_tau[..., _MEDIAN_IDX + 1:] = m_col + torch.hypot(carry_spread[:N_SPREADS], up_off)
            q_tau[..., :_MEDIAN_IDX] = m_col - torch.hypot(carry_spread[N_SPREADS:].flip(0), dn_off)
        bands = kovatchev_f_inv(q_tau)     # (PREDICTION_PATCHES, PATCH_SIZE, N_QUANTILES) mg/dL
        if conformal_delta is not None:
            # Recalibrate this roll's bands atop the carry widening; None = raw, bit-identical.
            from conformal import apply_quantile_conformal
            _bf = bands.reshape(-1, N_QUANTILES).detach().cpu().numpy()
            _bf = apply_quantile_conformal(_bf, _conformal_to_np(conformal_delta), _MEDIAN_IDX)
            bands = torch.from_numpy(_bf.astype(np.float32)).to(bands.device).reshape(bands.shape)

        # Accumulate native spread in quadrature: carry grows like √n over n (SPEC/inference.md §9).
        carry_spread = native if carry_spread is None else torch.hypot(carry_spread, native)

        all_q_tau.append(q_tau)
        all_bands.append(bands)
        all_pred_bgs.append(pred_bg_roll)

        # Dose feats default to the zero-RAW baseline, not 0.0 — exercise baseline is z=-0.1387.

        # feat 4 stays 0.0 here; correct, since next roll's builder rewrites it wholesale.
        new_ctx_patches = torch.zeros(PREDICTION_PATCHES, PATCH_SIZE, N_INPUT_FEATURES)
        for feat_idx, feat_baseline_z in baseline_z.items():
            new_ctx_patches[:, :, feat_idx] = feat_baseline_z

        # feat 0 is z(f(bg)); pred_bg_np is the clamped f_inv(median), so f is well-defined.
        pred_bg_np = pred_bg_roll.detach().cpu().numpy().reshape(
            PREDICTION_PATCHES, PATCH_SIZE
        )
        bg_input = kovatchev_f_np(pred_bg_np)
        bg_norm = (bg_input - bg_mean) / (bg_std + 1e-8)
        new_ctx_patches[:, :, 0] = torch.from_numpy(bg_norm.astype(np.float32))

        # Overridden doses become observed history next roll; unannounced ones invent nothing.
        if overrides_norm:
            for ch_idx, norm_vals in overrides_norm.items():
                if ch_idx not in CHANNEL_TO_FEAT:
                    continue
                feat_idx = CHANNEL_TO_FEAT[ch_idx]
                new_ctx_patches[:, :, feat_idx] = torch.from_numpy(
                    norm_vals.astype(np.float32)
                )

        # Slide the window so context never exceeds MAX_CONTEXT_PATCHES (attn mask shape).
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
    if return_rolls:
        result_out['roll_inputs'] = roll_inputs      # pyright: ignore[reportArgumentType]
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

    # ``weights_only=True`` is the secure-load path — it unpickles no arbitrary objects.
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
        # EMA tracks only float tensors; non-float buffers fall back to live weights.
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
    # Max context after SIMULATOR_WARMUP_HOURS drop; the smoke test exercises the ceiling.
    raw = simulate_discard_warmup(sim, 24)

    # bg_observed (post-CGM-noise), raw and unsmoothed, mirroring data._build_sample.

    # total_exercise is the simulator's carb-equivalent g/step curve, never rescaled.
    from T1DMSIM.simulator import BG_CLAMP_MIN, BG_CLAMP_MAX
    bg_obs = np.clip(raw['bg_observed'], BG_CLAMP_MIN, BG_CLAMP_MAX).astype(np.float32)
    carb = np.maximum(raw['total_carb'], 0.0).astype(np.float32)
    insulin = np.maximum(raw['total_insulin'], 0.0).astype(np.float32)
    exercise = np.maximum(raw['total_exercise'], 0.0).astype(np.float32)

    # Stack order: [bg_absolute, carb_intake, insulin_combined, exercise_equiv].

    # Count is len(CHANNEL_NAMES), not N_INPUT_FEATURES — bg_masked is a bit, appended below.
    assert len(CHANNEL_NAMES) == BG_MASKED_FEAT, (
        f"CHANNEL_NAMES has {len(CHANNEL_NAMES)} entries but bg_masked sits at "
        f"feat {BG_MASKED_FEAT}: {list(CHANNEL_NAMES)}"
    )
    raw_features = np.stack(
        [bg_obs, carb, insulin, exercise], axis=-1
    )                                             # (N, len(CHANNEL_NAMES))
    assert raw_features.shape[-1] == len(CHANNEL_NAMES), (
        f"raw signal stack has {raw_features.shape[-1]} columns but "
        f"CHANNEL_NAMES has {len(CHANNEL_NAMES)}"
    )
    # normalize applies Kovatchev f (bg) / log1p (sparse) before z — same path as data.py.
    features = normalize(raw_features, norm_stats)
    # bg_masked column starts all-visible; _build_patches_tensor rewrites it per masked set.
    features = np.concatenate(
        [features, np.zeros((len(features), 1), dtype=np.float32)], axis=-1,
    )                                             # (N, N_INPUT_FEATURES)
    assert features.shape[-1] == N_INPUT_FEATURES, (
        f"feature stack has {features.shape[-1]} columns but "
        f"N_INPUT_FEATURES={N_INPUT_FEATURES}"
    )

    # Trim to a multiple of PATCH_SIZE so reshape() is clean.
    N = (len(features) // PATCH_SIZE) * PATCH_SIZE
    features = features[:N]
    context_t = torch.from_numpy(features).reshape(-1, PATCH_SIZE, N_INPUT_FEATURES)

    result = predict(model, context_t, patient_seed=args.seed,
                     normalization_stats=norm_stats, device=device)

    print(f"q_tau shape: {result['q_tau'].shape}")
    print(f"median_bg range: [{result['median_bg'].min():.1f}, {result['median_bg'].max():.1f}] mg/dL")
    print(f"bands shape: {result['bands'].shape}")
