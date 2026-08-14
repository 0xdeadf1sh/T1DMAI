"""
T1DMAI Inference — standard, what-if, and rolling prediction modes.
====================================================================

This module is the runtime entry point for the model: given a trained
checkpoint and a context window, it builds the patches tensor, runs a
forward pass, and turns the model's RISK-space quantile head outputs into
mg/dL BG forecasts.

The pipeline occupies three spaces, crossed by two bridge pairs:

* (a) normalized z-space — the model **inputs** (bg_absolute, carbs,
  insulin, exercise), via ``normalization_stats``.  bg (feat 0) is normalized
  in **risk space**: the Kovatchev ``f`` is applied BEFORE the z-score, so the
  bg input the model sees is ``z(f(bg))`` (``normalize`` routes it through
  ``RISK_SPACE_CHANNELS``); carb / insulin / exercise stay ``z(log1p(·))``.
* (b) mg/dL physical — ``last_bg``, the BG forecast, every clinical metric.
* (c) risk space — the head **outputs** (``q_tau`` quantiles, ``median``).

``normalize/denormalize`` is the sole (a)<->(b) crossing;
``utils.kovatchev_f / utils.kovatchev_f_inv`` is the sole (b)<->(c) crossing.
Inference owns the ``kovatchev_f_inv`` (c)->(b) step: the model emits
risk-space quantiles and a median,
and this module inverts them to mg/dL band edges and a headline BG.

Three prediction modes are exposed:

* ``predict``         — single forward pass over one masked set.  By default
                        that set is the trailing ``PREDICTION_PATCHES`` zone —
                        a forecast — and the returned rows are that zone's
                        ``N`` hours (``N`` = the configured horizon).  The
                        headline BG forecast (``median_bg``) is
                        ``kovatchev.f_inv(median)``; the per-(h, τ) band
                        edges (``bands``) are ``kovatchev.f_inv(q_tau)``.
                        ``mask_spans`` names any other masked set: a span at
                        patch 0 is a backcast, one between visible patches an
                        infill.
* ``predict_what_if`` — same, but lets the caller override carbs / insulin /
                        exercise in the prediction zone (e.g. "what if I eat
                        40 g carbs at 6 PM?").  The announced values are
                        written straight into the carb (feat 1) / insulin
                        (feat 2) / exercise (feat 3) slots via
                        ``CHANNEL_TO_FEAT``; the model is always conditioned
                        on whatever values occupy those slots.
* ``predict_rolling`` — autoregressive rolling: predict one window, treat
                        the median BG forecast as new context, predict the
                        next window, repeat.  The re-feed is **BG-only**:
                        the median forecast (risk) -> ``f_inv`` -> mg/dL ->
                        ``normalize`` -> ``bg_absolute`` slot 0 of each new
                        context patch.  carbs / insulin / exercise come from
                        the caller's ``overrides_fn`` (else the zero-RAW
                        normalized baseline); there are no dynamics outputs
                        to write back.

Channel-layout note
-------------------
The ``N_INPUT_FEATURES`` input features are
``[bg_absolute, carb_intake, insulin_combined, exercise_equiv, bg_masked]`` —
there are no temporal (sin/cos) features.  The model has NO dynamics outputs:
its head emits risk-space quantiles over BG only.

The trailing ``bg_masked`` is the one feature that is not a normalized signal:
a per-PATCH BIT, 1.0 where feat 0 is withheld and the head predicts it.  It
carries no statistics, so ``normalization.CHANNEL_NAMES`` stays at four names
and a context handed to this module carries the four signals plus an
all-visible bit column.  The row layout is step-major, so the bit occupies all
``PATCH_SIZE`` columns of feat 4 — ``_build_patches_tensor`` is the only writer
of that column on this path, and it writes it from the masked set it was given.
Announcing it is not optional: masking is no longer inferable from position
(forecast, backcast and infill are the same objective), and ``z = 0`` in a
withheld bg slot decodes to an ordinary reading, not a sentinel.

``exercise_equiv`` is the simulator's carbohydrate-EQUIVALENT glucose-disposal
curve in g/step (it enters the simulator as
``glucose_in = total_carb + hgo - total_exercise``), encoded ``log1p`` + z like
carb — never through the Kovatchev transform, since it is not a glucose — and
never rescaled to an intensity: the trained scale is g/step.  It is a **plan**
channel: only a session the patient announced is ever written into the
prediction zone.
"""

import argparse
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from config import (
    PREDICTION_PATCHES, PATCH_SIZE, N_INPUT_FEATURES, N_QUANTILES,
    MAX_CONTEXT_PATCHES, MAX_MASKED_PATCHES,
    CHANNEL_TO_FEAT, MASKABLE_FEATS, NON_MASKABLE_FEATS, QUANTILE_LEVELS,
    TIME_PROBE_N_BINS,
)
# The masked set's slot expansion, the anchor rule and the bg_masked column
# index are ONE definition, in data.py, shared with the training builder.  A
# local re-derivation here is exactly the second copy that drifts: the anchor
# rule in particular is what makes a right-edge span reproduce the old
# ``last_bg``, and two copies of it agree only until one is edited.
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

# Index of the τ=0.5 median in the ascending quantile fan — the column conformal
# recalibration holds fixed (it moves only the band edges).
_MEDIAN_IDX = QUANTILE_LEVELS.index(0.5)

# One masked span, given as ``(start_patch, length)`` over the window's own
# patch axis — the same pair ``data.sample_mask_spans`` draws at training time.
MaskSpans = Sequence[tuple[int, int]]


def _conformal_to_np(delta: Any) -> np.ndarray:
    """Coerce a conformal delta (numpy array OR a torch tensor, possibly on CUDA) to a
    host numpy array — apply_quantile_conformal is pure numpy."""
    if torch.is_tensor(delta):
        return delta.detach().cpu().numpy()
    return np.asarray(delta)


def _resolve_mask_spans(mask_spans: MaskSpans | None, n_ctx: int) -> list[tuple[int, int]]:
    """Validate a masked set over the ``n_ctx + PREDICTION_PATCHES`` window, or
    build the default one.

    The default is the single trailing span ``(n_ctx, PREDICTION_PATCHES)`` — a
    forecast, which is one case of the masked-BG objective and not a mode of its
    own.  An explicit ``mask_spans`` may put spans anywhere: a span at patch 0 is
    a backcast, one between visible patches an infill.

    Four rules, each a correctness requirement rather than a style choice:

    * spans are strictly increasing and NEVER abut — the mandatory visible patch
      between neighbours is what makes the anchor, the per-span median basis and
      the DILATE bucket well defined per span; two spans with nothing between
      them are one longer span, and the sampler this mirrors
      (``data.sample_mask_spans``) never emits that.
    * ``sum(length) <= MAX_MASKED_PATCHES`` — the head has that many slots.
    * every patch of the FUTURE zone ``[n_ctx, T)`` is masked.  There is no
      observed BG there at all, so a future patch left visible announces a
      fabricated ``z = 0`` (~142 mg/dL on the balanced pool) as an observation.
    * at least one patch stays visible, since every span anchors on a visible
      neighbour and a window of pure prediction has nothing to read.
    """
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
    """``(patch_idx, step_idx)`` of each head slot's anchor cell in the context.

    ``data._mask_slots`` returns the anchor as a window-relative STEP index under
    the single anchor rule — ONE-SIDED and LEFT-PREFERRING: the last step of the
    span's left neighbour, or the first step of the right neighbour when the span
    starts at patch 0, the same value for every slot of a contiguous span.  This
    splits that index into the ``(patch, step)`` pair
    ``utils.last_bg_mgdl_from_context`` reads, and pins the two properties the
    inference path needs on top:

    * padded slots take slot 0's cell, so all ``M`` anchors are legal mg/dL and
      the forward's ``(B, M)`` units tripwire never fires on a slot ``valid`` is
      about to discard.
    * every anchor cell is a VISIBLE cell of the CONTEXT.  feat 0 of a masked
      patch is a legal-looking ``z`` that decodes to an ordinary mg/dL, so a
      wrong cell yields a plausible anchor rather than an error.
    """
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

    Run before EVERY forward on this path.  feat 4 is in neither
    ``MASKABLE_FEATS`` nor ``NON_MASKABLE_FEATS``, so no signal loop writes it and
    a builder that forgets it leaves the column at 0.0 — the forecast zone
    announced as observed, with every shape still matching and every fan assert
    still green.

    Three separate claims:  the column holds only 0.0 / 1.0 (it is a BIT, not a
    normalized signal); the bit is identical across all ``PATCH_SIZE`` step-major
    columns of feat 4 (it is per PATCH); and the announced set equals the
    requested one patch for patch.  Padded slots gather patch 0, so ``valid``
    is what keeps them out of the comparison.

    Args:
        patches:  ``(B, T, PATCH_DIM)`` the tensor about to be forwarded.
        mask_idx: ``(B, M)`` int64 patch index per head slot.
        valid:    ``(B, M)`` bool, False on padded slots.
    """
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
    """
    Build the ``(T, PATCH_DIM)`` patches tensor and attention mask for one
    masked set.  ``PATCH_DIM = PATCH_SIZE * N_INPUT_FEATURES``, of which the
    trailing ``PATCH_SIZE`` step-major columns are the ``bg_masked`` bit.

    The masked set is an ARGUMENT, not the implied trailing zone: this builder
    mirrors ``data._build_sample``, which withholds feat 0 on every masked patch,
    leaves carb / insulin / exercise at their true or announced values, and
    announces the set in feat 4.  ``mask_spans=None`` selects the trailing
    forecast and reproduces the old right-edge behaviour.

    Args:
        context: (n_ctx, PATCH_SIZE, N_INPUT_FEATURES) context data
                 (already normalized).  Its feat 4 column is not read: the bit
                 is written here from ``mask_spans``, which is the only
                 declaration of the masked set this path has.
        overrides: Optional dict mapping an output-channel override index to a
                 (PREDICTION_PATCHES, PATCH_SIZE) tensor of normalized override
                 values.  Keys are output-channel indices
                 {0: carb, 1: insulin, 2: exercise}; each is routed to its input
                 feature slot via the shared ``CHANNEL_TO_FEAT`` (carb -> feat 1,
                 insulin -> feat 2, exercise -> feat 3).  BG is never
                 overrideable.  Keys outside ``CHANNEL_TO_FEAT`` are silently
                 ignored.  A pred-zone slot left without an override stays at the
                 zero-RAW normalized baseline ``normalize(0)`` for that channel (a
                 genuine no-dose / no-session step, the value the model was
                 trained on) when ``normalization_stats`` is supplied — NOT the z=0
                 sentinel, which decodes through the sparse log1p inverse to a
                 phantom dose or a phantom exercise session.  These are **plan**
                 slots: nothing the patient did not announce is written into them.
        normalization_stats: normalization statistics used to seed the pred-zone
                 no-event baseline ``normalize(0)`` for the MASKABLE feats.
                 When ``None`` the announced slots fall back to the z=0 sentinel
                 (the legacy phantom-dose behavior); callers on the forecast path
                 (``predict``) always pass it.
        mask_spans: the masked set, ``[(start_patch, length), ...]`` over the
                 ``n_ctx + PREDICTION_PATCHES`` window.  ``None`` selects the
                 trailing forecast span ``(n_ctx, PREDICTION_PATCHES)``.  See
                 ``_resolve_mask_spans`` for the four rules it must satisfy.

    Returns:
        patches: (T, PATCH_DIM) float tensor.
        attn_mask: (T, T) bool tensor — the single-sample 2-D form
            ``model.forward`` accepts alongside the batched ``(B, T, T)``.
    """
    assert context.ndim == 3 and context.shape[-1] == N_INPUT_FEATURES, (
        f"context must be (n_ctx, PATCH_SIZE, {N_INPUT_FEATURES}), got {tuple(context.shape)}"
    )
    n_ctx = context.shape[0]
    seq_len = n_ctx + PREDICTION_PATCHES
    spans = _resolve_mask_spans(mask_spans, n_ctx)

    # --- Context patches ---
    # Flatten (PATCH_SIZE, N_INPUT_FEATURES) → PATCH_DIM.  The context carries its
    # observed feature values (bg / carb / insulin / exercise) verbatim.
    ctx_patches = context.reshape(n_ctx, PATCH_SIZE * N_INPUT_FEATURES)  # (n_ctx, PATCH_DIM)

    # --- Prediction patches ---
    # bg (feat 0 / NON_MASKABLE_FEATS) stays 0 — it is what the model predicts,
    # and every future patch is masked by construction (_resolve_mask_spans).
    # The announced slots (MASKABLE_FEATS = carb feat 1 / insulin feat 2 /
    # exercise feat 3) are seeded to the zero-RAW normalized baseline normalize(0)
    # per channel — a genuine no-dose, no-session step — NOT z=0, which decodes
    # through the sparse log1p inverse to a phantom ~0.39 g / ~0.14 U / ~0.025 g
    # per step (the exact hazard predict_rolling's context fill guards against, and
    # the value the model was TRAINED on for a no-event step). Overrides below
    # overwrite these with the announced values.
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
    # input feature slot (carb -> feat 1, insulin -> feat 2, exercise -> feat 3),
    # the SAME mapping data.py uses, so there is no second independent ``+offset``
    # literal to drift.
    if overrides:
        for ch_idx, override_vals in overrides.items():
            if ch_idx not in CHANNEL_TO_FEAT:
                continue
            feat_idx = CHANNEL_TO_FEAT[ch_idx]
            # override_vals: (PREDICTION_PATCHES, PATCH_SIZE)
            for t in range(PATCH_SIZE):
                flat_col = t * N_INPUT_FEATURES + feat_idx
                pred_patches[:, flat_col] = override_vals[:, t]

    # Combine context + prediction along the sequence axis.  ``cat`` allocates,
    # so the writes below never reach back into the caller's ``context``.
    patches = torch.cat([ctx_patches, pred_patches], dim=0)  # (T, PATCH_DIM)

    # --- The masked set: withhold bg, then announce it ---
    # Slot expansion comes from data._mask_slots, the same function the training
    # builder uses, so the two paths cannot disagree about which patch a slot is.
    mask_idx, valid, _d, _anchor_step = _mask_slots(spans, seq_len)
    masked_rows = torch.from_numpy(mask_idx[valid])
    # A masked patch withholds bg (feat 0, the only NON_MASKABLE_FEATS entry) —
    # that is what the model is asked to emit.  The future zone is already 0
    # there; this is what withholds bg on a masked CONTEXT patch (backcast /
    # infill), where the caller handed us a real observation.  carb / insulin /
    # exercise pass through untouched: the model is always conditioned on the
    # announced plan.
    for feat_idx in NON_MASKABLE_FEATS:
        patches[masked_rows, feat_idx::N_INPUT_FEATURES] = 0.0
    # Announce the masked set.  feat 4 is in neither MASKABLE_FEATS nor
    # NON_MASKABLE_FEATS, so nothing above writes it and it would otherwise stay
    # at the allocation's 0.0 — every masked patch announced as OBSERVED.  The
    # bit is per PATCH and the layout step-major, so it goes into all PATCH_SIZE
    # columns of feat 4; a column outside that block would break both
    # PATCH_DIM = PATCH_SIZE * N_INPUT_FEATURES and the f::N_INPUT_FEATURES
    # stride idiom.  Both halves are written explicitly, so a stale bit riding in
    # on the caller's context cannot survive as a phantom announcement.
    patches[:, BG_MASKED_FEAT::N_INPUT_FEATURES] = 0.0
    patches[masked_rows, BG_MASKED_FEAT::N_INPUT_FEATURES] = 1.0

    # --- Attention mask ---
    # Built from the visible/masked labelling, not from a position rule: a
    # visible row reads visible columns, a masked row reads everything.  Single
    # sample, so there is no padding and the (1, T, T) general form collapses to
    # the (T, T) one model.forward broadcasts over batch and head.
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
) -> dict[str, Any]:
    """Build one sample, announce its masked set, check the announcement, forward.

    The single chokepoint every forward in this module goes through, so the feat-4
    assert cannot be skipped on one path and the ``M`` anchors cross the
    denormalize bridge exactly once per call.

    Anchors: ``utils.last_bg_mgdl_from_context`` reads all ``M`` cells plus the
    context edge in ONE host transfer, z-inverse then ``kovatchev_f_inv`` back to
    mg/dL — the inference path has no raw mg/dL array to read the way the
    training builder does, so this round trip IS the anchor.  Every value it
    returns is clipped into the physical BG range, so the forward's ``(B, M)``
    units tripwire has a legal value in every slot including the padded ones.

    Returns a dict with the ``(M, ...)`` head outputs, the ``(M,)`` slot
    bookkeeping and the context-edge ``last_bg``.
    """
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
    # M anchor cells plus the context edge (-1, -1): one transfer, one float64
    # inverse.  The edge read is ``last_bg`` — the scalar the GUI and the metric
    # paths plot as the last observed BG — and for the default trailing forecast
    # it IS slot 0's anchor, the same cell the training builder reads at
    # ``bg_window[n_ctx * PATCH_SIZE - 1]``.
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

    # feat 4 must reproduce the requested masked set — checked HERE, on the exact
    # tensor about to be forwarded, rather than trusted from the builder.
    _assert_mask_announced(patches, mask_idx_t, valid_t)

    with torch.no_grad():
        out = model(
            patches, attn_mask, anchor_bg, mask_idx_t, return_time=return_time,
        )
    q_tau, median = out[0], out[1]
    time_pred = out[2] if return_time else None

    return {
        'q_tau': q_tau.squeeze(0),                          # (M, PATCH_SIZE, N_QUANTILES)
        'median': median.squeeze(0),                        # (M, PATCH_SIZE)
        'time_pred': None if time_pred is None else time_pred.squeeze(0),
        'mask_idx': mask_idx_t.squeeze(0),                  # (M,) patch index per slot
        'valid': valid_t.squeeze(0),                        # (M,) bool
        'anchor_bg': anchor_bg.squeeze(0),                  # (M,) mg/dL
        'last_bg': last_bg,
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
) -> dict[str, torch.Tensor]:
    """
    Standard prediction: single forward pass over one masked set.

    The model emits RISK-space quantiles and a median for every masked patch;
    this function inverts them to mg/dL via ``kovatchev.f_inv``.  The default
    masked set is the trailing ``PREDICTION_PATCHES`` zone, so the default call
    is the forecast it has always been — one case of the objective, not a mode.

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
                 ({0: carb, 1: insulin, 2: exercise}) to a
                 (PREDICTION_PATCHES, PATCH_SIZE) tensor of NORMALIZED
                 prediction-zone values, announcing that channel to the model.
                 Slots without an override stay at the ``normalize(0)`` no-event
                 baseline (not z=0).  BG is never overrideable.
        return_time: opt-in diagnostic flag.  When ``False`` (default) the compute
                 and returned keys are BIT-IDENTICAL to the pre-existing behavior.
                 When ``True`` the forward runs with ``return_time=True`` and the
                 result gains a single ``time_pred`` key — the auxiliary time-of-day
                 probe's per-patch bin logits (never touches the BG forecast, loss,
                 or checkpoint selection).
        mask_spans: the masked set, ``[(start_patch, length), ...]`` over the
                 ``n_ctx + PREDICTION_PATCHES`` window.  ``None`` (default) is the
                 trailing forecast span.  A span at patch 0 is a backcast, one
                 between visible patches an infill; the whole future zone is
                 masked either way, since it carries no observed BG.  The head
                 has ``MAX_MASKED_PATCHES`` slots and the surplus is padded, so a
                 masked set may total at most that many patches.
                 ``conformal_delta`` is fit per (step, level) against the
                 forecast protocol, so it means nothing under another masked set
                 of the same row count — pass it with the default set only.

    Returns:
        result dict, with ``P`` = the number of masked patches
        (``PREDICTION_PATCHES`` for the default forecast, and the rows are in
        ``mask_idx`` order)::

            q_tau:     (P, PATCH_SIZE, N_QUANTILES) risk-space quantiles,
                       ascending τ.
            median:    (P, PATCH_SIZE) risk-space median.
            mask_idx:  (P,) int64 patch index each row predicts.
            median_bg: (P * PATCH_SIZE,) HEADLINE BG forecast,
                       ``f_inv(median)`` in mg/dL (only if stats provided).
            bands:     (P, PATCH_SIZE, N_QUANTILES) mg/dL band edges,
                       ``f_inv(q_tau)`` (only if stats provided).
            last_bg:   scalar mg/dL reading at the context edge (only if stats
                       provided) — the forecast anchor in the default case.
            time_pred: (P, TIME_PROBE_N_BINS) raw time-of-day probe bin logits,
                       or ``None`` when the probe is disabled — present ONLY when
                       ``return_time=True``.
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

    # Build the sample, announce its masked set, check the announcement and run
    # the forward — all inside ``_run_forward``.  ``overrides`` announce the
    # prediction-zone doses; un-overridden dose slots get the ``normalize(0)``
    # no-dose baseline (via ``anchor_stats``), never z=0.
    out = _run_forward(
        model, context, anchor_stats, overrides=overrides,
        mask_spans=mask_spans, device=device, return_time=return_time,
    )

    # Keep the VALID slots only.  The head always emits MAX_MASKED_PATCHES slots
    # and the surplus gathers patch 0, so returning the raw M axis would hand the
    # caller a plausible forecast of a patch nobody asked about.  What is left is
    # one row per masked patch in mask_idx order — the trailing forecast zone,
    # in order, for the default masked set.
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
        # (P, TIME_PROBE_N_BINS) raw logits, or None when disabled.
        # Decode/softmax stays in utils (single chokepoint) — emit raw here.
        time_pred = out['time_pred']
        result['time_pred'] = None if time_pred is None else time_pred[valid]

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

    Shares ``predict``'s forward chokepoint exactly (``_run_forward``: the same
    patch build, the same feat-4 announcement check, the same ``M`` anchors
    through ``last_bg_mgdl_from_context``), then reads the
    ``(M, TIME_PROBE_N_BINS)`` per-SLOT bin logits and decodes slot 0.  The
    masked set is the trailing forecast, so slot 0 is the forecast origin patch.
    The forecast ``q_tau`` / ``median`` are computed identically to ``predict``
    and discarded here — this is a read-only diagnostic that never touches the BG
    forecast.

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

    out = _run_forward(
        model, context, anchor_stats, overrides=None, mask_spans=None,
        device=device, return_time=True,
    )
    time_pred = out['time_pred']                 # (M, TIME_PROBE_N_BINS) or None

    if time_pred is None:  # TIME_PROBE_ENABLED is False ⇒ probe head not built
        return float('nan'), float('nan')

    # Slot 0 of the trailing forecast span is the origin patch, and it is valid
    # by construction (the default masked set always fills at least slot 0).
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
) -> dict[str, torch.Tensor]:
    """
    What-if prediction: announce carb / insulin / exercise in the prediction
    zone and run a forward pass conditioned on them.

    Override values must already be NORMALIZED (z-score / log1p) — use
    ``normalization.normalize`` if the caller has raw-unit values.

    Args:
        model: T1DMAI model in eval mode.
        context: (n_ctx, PATCH_SIZE, N_INPUT_FEATURES) normalized context.
        patient_seed: Unused (legacy positional arg; kept so existing callers
                   don't break).
        overrides: Dict mapping an output-channel index
                   ({0: carb, 1: insulin, 2: exercise}) to a
                   (PREDICTION_PATCHES, PATCH_SIZE) tensor of normalized values;
                   routed to feat 1 / 2 / 3 via ``CHANNEL_TO_FEAT``.  BG is
                   never overrideable.
        normalization_stats: REQUIRED for the mg/dL ``last_bg`` anchor and the
                   ``median_bg`` / ``bands`` mg/dL fields.
        device: Torch device.
        return_time: opt-in diagnostic flag forwarded to ``predict`` — when
                   ``True`` the result gains the ``time_pred`` per-patch bin
                   logits key (default ``False`` ⇒ bit-identical).

    Returns:
        Same as ``predict`` but reflecting the conditioned
        carb / insulin / exercise.
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
         insulin (feat 2) / exercise (feat 3) come from the caller's
         ``overrides_fn`` if supplied, else are set to the **zero-RAW normalized
         baseline** (the z-score of a literal 0 g / 0 U / 0 g-equivalent through
         ``normalize`` — NOT ``torch.zeros``, whose z=0 decodes to a phantom
         ~0.39 g / ~0.14 U / ~0.025 g).  There are NO dynamics outputs to write
         back.
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
            output channels {0: carb, 1: insulin, 2: exercise}), the roll is
            re-predicted conditioned on ``overrides_norm`` and those announced
            values are written into the next window's context.  Returning
            ``None`` ⇒ nothing announced this roll (the zero-RAW baseline is
            used for every announceable slot).  The
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

    # Zero-RAW normalized baseline for every re-fed announceable channel — carb
    # (feat 1 / carb_intake), insulin (feat 2 / insulin_combined) and exercise
    # (feat 3 / exercise_equiv).  ``normalize`` of a literal 0 routes each sparse
    # channel through ``log1p`` first, so the baseline is the channel's
    # ``-mean/std`` — NOT z=0, which decodes to a phantom dose or a phantom
    # ~0.025 g/step of exercise disposal.  Keyed over ``MASKABLE_FEATS`` so the
    # next announceable channel needs no edit here; feat 0 (bg) is excluded and is
    # overwritten by the BG re-feed below anyway.
    zero_raw = normalize(
        np.zeros((1, len(CHANNEL_NAMES)), dtype=np.float32), normalization_stats,
    )[0]                                                  # (n_channels,) z-space
    baseline_z = {feat: float(zero_raw[feat]) for feat in MASKABLE_FEATS}

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
        # once: ``predict_what_if`` when the caller announced anything
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
        # it into the bg_absolute slot 0.  Every announceable slot — carb (feat 1),
        # insulin (feat 2), exercise (feat 3) — defaults to the zero-RAW normalized
        # baseline (a literal 0 g / 0 U / 0 g-equivalent through ``normalize``, NOT
        # z=0 — z=0 decodes to a phantom dose or a phantom exercise session); the
        # override (if any) writes real values over them.  Leaving a slot at the
        # literal 0.0 this tensor is allocated with is NOT "no event": for
        # exercise_equiv the no-session baseline is z = -0.1387 under the balanced
        # pool.  No temporal features, no dynamics.
        # feat 4 (bg_masked) stays 0.0 here, and that is the correct value rather
        # than an omission: these patches become the NEXT roll's context, where
        # the re-fed BG is the evidence the next window reads.  The builder
        # rewrites the whole feat-4 column from that roll's masked set anyway.
        new_ctx_patches = torch.zeros(PREDICTION_PATCHES, PATCH_SIZE, N_INPUT_FEATURES)
        for feat_idx, feat_baseline_z in baseline_z.items():
            new_ctx_patches[:, :, feat_idx] = feat_baseline_z

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

        # carb (feat 1) / insulin (feat 2) / exercise (feat 3) from the announced
        # overrides, if any — written straight into the appended context patches
        # (they become observed history for the next roll).  Without an override
        # they keep the zero-RAW baseline set above: nothing the patient did not
        # announce is ever invented here.
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
    # features are NO LONGER inputs (dropped from the input stack), so they are
    # not gathered here.  The model consumes the RAW post-noise signals (no
    # smoothing), mirroring ``data._build_sample`` (bg clamped to the physical BG
    # range so it is a legal Kovatchev-f argument, the sparse carb / insulin /
    # exercise channels floored at 0).  ``total_exercise`` is the simulator's
    # carbohydrate-equivalent glucose-disposal curve in g/step — fed at that
    # scale, never rescaled to an intensity.
    from T1DMSIM.simulator import BG_CLAMP_MIN, BG_CLAMP_MAX
    bg_obs = np.clip(raw['bg_observed'], BG_CLAMP_MIN, BG_CLAMP_MAX).astype(np.float32)
    carb = np.maximum(raw['total_carb'], 0.0).astype(np.float32)
    insulin = np.maximum(raw['total_insulin'], 0.0).astype(np.float32)
    exercise = np.maximum(raw['total_exercise'], 0.0).astype(np.float32)

    # Signal stack in CHANNEL_NAMES order:
    # [bg_absolute, carb_intake, insulin_combined, exercise_equiv] — bg_delta / IS
    # / HGO and the temporal sin/cos features are dropped.  The check is on the
    # COUNT, not on the literal names: the names live in ``normalization`` alone,
    # and the stack below is what has to move when a channel is added.  The count
    # is len(CHANNEL_NAMES), NOT N_INPUT_FEATURES: the trailing bg_masked feature
    # is a bit, carries no statistics and is appended below rather than stacked
    # and normalized here.
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
    # ``normalize`` applies the Kovatchev ``f`` to bg (via RISK_SPACE_CHANNELS) and
    # log1p to carb / insulin / exercise BEFORE the z-score — the SAME path data.py
    # uses, so bg feat 0 lands in z(f(bg)) space.  Don't hand-roll the per-channel
    # transform.
    features = normalize(raw_features, norm_stats)
    # Append the bg_masked column, all-visible: every step here is an observed
    # reading.  ``_build_patches_tensor`` rewrites the column from the masked set
    # it is given, so this is the context's own announcement, not the model's
    # input bit.
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
