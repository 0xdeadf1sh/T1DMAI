"""Tests for config.py — verify all constants exist and are self-consistent."""


def test_config_imports():
    """All constants import from config.py without error."""
    from config import (D_MODEL, N_LAYERS, N_HEADS, HEAD_DIM, FFN_DIM,
                        PATCH_SIZE, N_INPUT_FEATURES, N_QUANTILES,
                        MAX_CONTEXT_PATCHES, MIN_CONTEXT_PATCHES,
                        PREDICTION_PATCHES, MAX_SEQ_LEN,
                        PREDICTION_HORIZON_HOURS)
    # Architecture constants are tunable via resize_model.py — only assert
    # the invariants that the rest of the project depends on.
    assert D_MODEL == N_HEADS * HEAD_DIM
    # FFN_DIM = ffn-mult × D_MODEL with ffn-mult a positive int (resize_model.py
    # --ffn-mult), so the invariant is ≥, not >: ffn-mult=1 (FFN_DIM == D_MODEL)
    # is a valid experimental config.
    assert FFN_DIM >= D_MODEL
    # PATCH_SIZE is tunable via resize_model.py --patch-size (e.g. 3 = 15-min
    # patches); the real invariant is that an integer number of patches tiles
    # each hour, not any specific size.
    assert 60 % (PATCH_SIZE * 5) == 0
    # Risk-space redesign: 5 input features
    # [bg_absolute, carb_intake, insulin_combined, exercise_equiv, bg_masked] —
    # the 4 temporal sin/cos features are dropped and feat 4 announces the masked
    # set.  No per-channel output set — the BG head emits the 7 risk-space
    # quantiles.
    assert N_INPUT_FEATURES == 5
    assert N_QUANTILES == 7
    assert MAX_SEQ_LEN == MAX_CONTEXT_PATCHES + PREDICTION_PATCHES
    # Prediction patches tile the horizon at (60 / (PATCH_SIZE*5)) patches/hour.
    assert PREDICTION_PATCHES == PREDICTION_HORIZON_HOURS * (60 // (PATCH_SIZE * 5))
    # Horizon is bounded by the 24h cycle.
    assert 0 < PREDICTION_HORIZON_HOURS <= 24


def test_quantile_levels_consistent():
    """QUANTILE_LEVELS is the 7-level ascending grid, with the median at index 3,
    and N_SPREADS = (N_QUANTILES - 1) // 2."""
    from config import QUANTILE_LEVELS, N_QUANTILES, N_SPREADS
    assert len(QUANTILE_LEVELS) == N_QUANTILES == 7
    assert list(QUANTILE_LEVELS) == sorted(QUANTILE_LEVELS), "levels must be ascending"
    assert QUANTILE_LEVELS[3] == 0.5, "median must sit at index 3"
    assert N_SPREADS == (N_QUANTILES - 1) // 2 == 3


def test_head_dim_consistency():
    """d_model == n_heads × head_dim."""
    from config import D_MODEL, N_HEADS, HEAD_DIM
    assert D_MODEL == N_HEADS * HEAD_DIM, (
        f"d_model ({D_MODEL}) != n_heads ({N_HEADS}) * head_dim ({HEAD_DIM})"
    )


def test_patch_dim_consistency():
    """PATCH_DIM == PATCH_SIZE * N_INPUT_FEATURES == 30.

    The ``bg_masked`` announcement bit is feat 4 INSIDE the step-major block, not
    a trailing tier: PATCH_DIM grows by PATCH_SIZE to carry one bit per patch, so
    the identity above and the ``[:, f::N_INPUT_FEATURES]`` stride idiom both
    survive.  The retired ``N_MASK_BITS`` tier — a separate trailing width — must
    stay deleted."""
    import config
    from config import PATCH_DIM, PATCH_SIZE, N_INPUT_FEATURES
    expected = PATCH_SIZE * N_INPUT_FEATURES
    assert PATCH_DIM == expected, f"PATCH_DIM {PATCH_DIM} != {expected}"
    assert PATCH_DIM == 30, f"expected PATCH_DIM 30 at the active config, got {PATCH_DIM}"
    assert not hasattr(config, 'N_MASK_BITS'), \
        "config.N_MASK_BITS must be deleted (the trailing mask-bit tier is gone)"


def test_mask_sampler_constants():
    """The masked-BG sampler knobs are well formed and mutually consistent.

    The span ceiling and the budget are TUNED between runs — together they set the
    ``d`` histogram the masked objective is supervised on — so what is asserted
    here is the shape of the three constants and the constraints that tie them to
    each other and to the window the picker draws, never one particular tuple.
    ``MAX_MASKED_PATCHES`` is two things at once: the sampler's cap on ``sum(L)``
    and ``M``, the head's slot count."""
    from config import (MASK_MAX_SPANS, MASK_SPAN_LENGTHS, MAX_MASKED_PATCHES,
                        MIN_CONTEXT_PATCHES, PREDICTION_PATCHES)

    assert isinstance(MASK_SPAN_LENGTHS, tuple) and MASK_SPAN_LENGTHS, \
        f"MASK_SPAN_LENGTHS must be a non-empty tuple, got {MASK_SPAN_LENGTHS!r}"
    assert all(isinstance(L, int) and L >= 1 for L in MASK_SPAN_LENGTHS), \
        f"MASK_SPAN_LENGTHS must hold positive ints, got {MASK_SPAN_LENGTHS}"
    assert list(MASK_SPAN_LENGTHS) == sorted(set(MASK_SPAN_LENGTHS)), \
        f"MASK_SPAN_LENGTHS must be strictly ascending, got {MASK_SPAN_LENGTHS}"
    assert isinstance(MASK_MAX_SPANS, int) and MASK_MAX_SPANS >= 1, \
        f"MASK_MAX_SPANS must be a positive int, got {MASK_MAX_SPANS!r}"
    assert isinstance(MAX_MASKED_PATCHES, int), \
        "MAX_MASKED_PATCHES is M, the head's slot count, so it must be an int, " \
        f"got {MAX_MASKED_PATCHES!r}"
    assert MAX_MASKED_PATCHES >= MASK_SPAN_LENGTHS[-1], (
        f"a single longest span ({MASK_SPAN_LENGTHS[-1]}) must fit the masked-patch "
        f"budget ({MAX_MASKED_PATCHES}), or the sampler rejects for ever")
    # The shortest window the picker can draw must hold the largest legal masked
    # set plus its mandatory separators, or sample_mask_spans raises.
    shortest_T = MIN_CONTEXT_PATCHES + PREDICTION_PATCHES
    assert MAX_MASKED_PATCHES + (MASK_MAX_SPANS - 1) <= shortest_T, (
        f"{MAX_MASKED_PATCHES} masked patches + {MASK_MAX_SPANS - 1} separators "
        f"do not fit the shortest window ({shortest_T} patches)")
    print(f"\n[DUMP] mask knobs | spans<={MASK_MAX_SPANS} lengths={MASK_SPAN_LENGTHS} "
          f"budget={MAX_MASKED_PATCHES} (== M); shortest window {shortest_T} ✓")


def test_channel_count_is_not_the_feature_count():
    """``len(CHANNEL_NAMES) == 4`` while ``N_INPUT_FEATURES == 5``.

    ``bg_masked`` is a BIT, not a normalized signal: it has no mean, no std and
    no log1p encoding, so it gets no CHANNEL_NAMES entry.  The two numbers used to
    be the same and an identity between them is what breaks first."""
    from config import N_INPUT_FEATURES
    from normalization import CHANNEL_NAMES, N_CHANNELS

    assert N_CHANNELS == len(CHANNEL_NAMES) == 4, \
        f"CHANNEL_NAMES must stay at 4 signal channels, got {CHANNEL_NAMES}"
    assert N_INPUT_FEATURES == N_CHANNELS + 1, \
        "N_INPUT_FEATURES is the signal channels plus the bg_masked bit"


def test_architecture_redesign_constants():
    """The risk-v4 redesign with DILATE restored: RoPE base 1000, DILATE loss knobs
    present, the learned Kendall-Gal weighting restored (KENDALL_LOGVAR_INIT), the
    median-curvature (L_smooth) penalty and the static loss weights removed, the
    seam (L_seam) penalty still retired, and the TILDE-Q / risk-input-flag /
    temporal-slice constants all deleted.  Provenance stamps: risk-v4 / the
    kendall-pinball-dilate loss schema.  The clinical hypo/hyper detectors fire off
    the selectable lower / upper band edges (default τ=0.10 / τ=0.90)."""
    import config

    # (2) RoPE base frequency 10000 -> 1000.
    assert config.ROPE_BASE == 1000, f"ROPE_BASE must be 1000, got {config.ROPE_BASE}"

    # (3) DILATE (restored) knobs: shape/TDI mix, softmin softness, TDI FD step.
    assert hasattr(config, 'DILATE_ALPHA'), "DILATE_ALPHA must exist"
    assert hasattr(config, 'DILATE_GAMMA'), "DILATE_GAMMA must exist"
    assert hasattr(config, 'DILATE_TDI_FD_EPS'), "DILATE_TDI_FD_EPS must exist"
    assert 0.0 <= config.DILATE_ALPHA <= 1.0, \
        f"DILATE_ALPHA must be in [0, 1], got {config.DILATE_ALPHA}"
    assert config.DILATE_GAMMA > 0.0, \
        f"DILATE_GAMMA must be > 0, got {config.DILATE_GAMMA}"
    assert config.DILATE_TDI_FD_EPS > 0.0, \
        f"DILATE_TDI_FD_EPS must be > 0, got {config.DILATE_TDI_FD_EPS}"

    # (3b) The learned Kendall-Gal weighting is restored: its log-variance init
    # exists again, and the median-curvature penalty (L_smooth) is removed entirely.
    assert hasattr(config, 'KENDALL_LOGVAR_INIT'), \
        "KENDALL_LOGVAR_INIT must exist (learned Kendall-Gal weighting restored)"
    for gone in ('MEDIAN_SMOOTHNESS_ENABLED', 'MEDIAN_SMOOTHNESS_WEIGHT'):
        assert not hasattr(config, gone), \
            f"config.{gone} must be deleted (L_smooth penalty removed entirely)"

    # (3c) TILDE-Q is retired, the seam penalty (L_seam) stays removed, and the
    # static loss weights are gone (the learned Kendall-Gal combine replaces them).
    for gone in ('TILDEQ_ALPHA', 'TILDEQ_GAMMA',
                 'MEDIAN_SEAM_PENALTY_ENABLED', 'MEDIAN_SEAM_PENALTY_WEIGHT',
                 'PINBALL_LOSS_WEIGHT', 'DILATE_LOSS_WEIGHT'):
        assert not hasattr(config, gone), f"config.{gone} must be deleted"

    # (3d) The band-edge clinical detectors: hypo off a LOWER band edge, hyper off
    # an UPPER one. The τ is SELECTABLE — any QUANTILE_LEVELS entry on the correct
    # side — so assert membership + side, not a hardcoded index.
    from config import QUANTILE_LEVELS
    assert (config.HYPO_ALARM_QUANTILE_TAU in QUANTILE_LEVELS
            and config.HYPO_ALARM_QUANTILE_TAU < 0.5), \
        f"HYPO_ALARM_QUANTILE_TAU must be a lower-half level, got {config.HYPO_ALARM_QUANTILE_TAU}"
    assert (config.HYPER_ALARM_QUANTILE_TAU in QUANTILE_LEVELS
            and config.HYPER_ALARM_QUANTILE_TAU > 0.5), \
        f"HYPER_ALARM_QUANTILE_TAU must be an upper-half level, got {config.HYPER_ALARM_QUANTILE_TAU}"
    # Current default: the 25 / 75 band.
    assert config.HYPO_ALARM_QUANTILE_TAU == 0.25
    assert config.HYPER_ALARM_QUANTILE_TAU == 0.75
    # Precision-only forgiveness band (recall stays strict).
    assert hasattr(config, 'EXCURSION_PRECISION_TOLERANCE_MGDL'), \
        "EXCURSION_PRECISION_TOLERANCE_MGDL must exist (precision forgiveness band)"
    assert config.EXCURSION_PRECISION_TOLERANCE_MGDL >= 0.0

    # (5) The temporal-feature slice marker is gone (no temporal inputs).
    assert not hasattr(config, 'TEMPORAL_FEAT_START'), \
        "config.TEMPORAL_FEAT_START must be deleted (temporal inputs removed)"

    # (6) The BG-input-risk toggle is gone — risk space is the sole path.
    assert not hasattr(config, 'BG_INPUT_RISK_SPACE'), \
        "config.BG_INPUT_RISK_SPACE must be deleted (risk-space input is unconditional)"

    # Provenance stamps advance with the redesign.
    assert config.ARCH_VERSION == 'risk-v4', \
        f"ARCH_VERSION must be 'risk-v4', got {config.ARCH_VERSION!r}"
    assert config.LOSS_SCHEMA == 'kendall-pinball-dilate-v3', \
        f"LOSS_SCHEMA must be 'kendall-pinball-dilate-v3', got {config.LOSS_SCHEMA!r}"
    print(f"\n[DUMP] redesign | ROPE_BASE={config.ROPE_BASE} "
          f"DILATE_ALPHA={config.DILATE_ALPHA} DILATE_GAMMA={config.DILATE_GAMMA} "
          f"ARCH={config.ARCH_VERSION} LOSS={config.LOSS_SCHEMA} ✓")
