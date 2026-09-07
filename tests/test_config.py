def test_config_imports():
    from config import (D_MODEL, N_LAYERS, N_HEADS, HEAD_DIM, FFN_DIM,
                        PATCH_SIZE, N_INPUT_FEATURES, N_QUANTILES,
                        MAX_CONTEXT_PATCHES, MIN_CONTEXT_PATCHES,
                        PREDICTION_PATCHES, MAX_SEQ_LEN,
                        PREDICTION_HORIZON_HOURS)
    # dims are tunable via resize_model.py; assert only the invariants
    assert D_MODEL == N_HEADS * HEAD_DIM
    # >= not >: resize_model.py --ffn-mult 1 is a legal config
    assert FFN_DIM >= D_MODEL
    # the invariant is whole patches tiling an hour, not any one PATCH_SIZE
    assert 60 % (PATCH_SIZE * 5) == 0
    assert N_INPUT_FEATURES == 5
    assert N_QUANTILES == 7
    assert MAX_SEQ_LEN == MAX_CONTEXT_PATCHES + PREDICTION_PATCHES
    assert PREDICTION_PATCHES == PREDICTION_HORIZON_HOURS * (60 // (PATCH_SIZE * 5))
    # the 24 h cycle bounds the horizon
    assert 0 < PREDICTION_HORIZON_HOURS <= 24


def test_quantile_levels_consistent():
    from config import QUANTILE_LEVELS, N_QUANTILES, N_SPREADS
    assert len(QUANTILE_LEVELS) == N_QUANTILES == 7
    assert list(QUANTILE_LEVELS) == sorted(QUANTILE_LEVELS), "levels must be ascending"
    assert QUANTILE_LEVELS[3] == 0.5, "median must sit at index 3"
    assert N_SPREADS == (N_QUANTILES - 1) // 2 == 3


def test_head_dim_consistency():
    from config import D_MODEL, N_HEADS, HEAD_DIM
    assert D_MODEL == N_HEADS * HEAD_DIM, (
        f"d_model ({D_MODEL}) != n_heads ({N_HEADS}) * head_dim ({HEAD_DIM})"
    )


def test_patch_dim_consistency():
    """``bg_masked`` is feat 4 inside the step-major block, not a trailing tier —
    the ``[:, f::N_INPUT_FEATURES]`` stride idiom depends on it."""
    import config
    from config import PATCH_DIM, PATCH_SIZE, N_INPUT_FEATURES
    expected = PATCH_SIZE * N_INPUT_FEATURES
    assert PATCH_DIM == expected, f"PATCH_DIM {PATCH_DIM} != {expected}"
    assert PATCH_DIM == 30, f"expected PATCH_DIM 30 at the active config, got {PATCH_DIM}"
    assert not hasattr(config, 'N_MASK_BITS'), \
        "config.N_MASK_BITS must be deleted (the trailing mask-bit tier is gone)"


def test_mask_sampler_constants():
    """Tuned between runs, so assert shape and constraints, never one tuple.

    ``MAX_MASKED_PATCHES`` is both the sampler's cap on ``sum(L)`` and ``M``, the
    head's slot count."""
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
    # the shortest drawable window must hold the largest masked set plus its
    # mandatory separators, or sample_mask_spans raises
    shortest_T = MIN_CONTEXT_PATCHES + PREDICTION_PATCHES
    assert MAX_MASKED_PATCHES + (MASK_MAX_SPANS - 1) <= shortest_T, (
        f"{MAX_MASKED_PATCHES} masked patches + {MASK_MAX_SPANS - 1} separators "
        f"do not fit the shortest window ({shortest_T} patches)")
    print(f"\n[DUMP] mask knobs | spans<={MASK_MAX_SPANS} lengths={MASK_SPAN_LENGTHS} "
          f"budget={MAX_MASKED_PATCHES} (== M); shortest window {shortest_T} ✓")


def test_channel_count_is_not_the_feature_count():
    """``bg_masked`` is a bit — no mean, no std, no log1p — so it gets no
    CHANNEL_NAMES entry: 4 channels, 5 features."""
    from config import N_INPUT_FEATURES
    from normalization import CHANNEL_NAMES, N_CHANNELS

    assert N_CHANNELS == len(CHANNEL_NAMES) == 4, \
        f"CHANNEL_NAMES must stay at 4 signal channels, got {CHANNEL_NAMES}"
    assert N_INPUT_FEATURES == N_CHANNELS + 1, \
        "N_INPUT_FEATURES is the signal channels plus the bg_masked bit"


def test_architecture_redesign_constants():
    import config

    assert config.ROPE_BASE == 1000, f"ROPE_BASE must be 1000, got {config.ROPE_BASE}"

    assert hasattr(config, 'DILATE_ALPHA'), "DILATE_ALPHA must exist"
    assert hasattr(config, 'DILATE_GAMMA'), "DILATE_GAMMA must exist"
    assert hasattr(config, 'DILATE_TDI_FD_EPS'), "DILATE_TDI_FD_EPS must exist"
    assert 0.0 <= config.DILATE_ALPHA <= 1.0, \
        f"DILATE_ALPHA must be in [0, 1], got {config.DILATE_ALPHA}"
    assert hasattr(config, 'MSE_ALPHA'), "MSE_ALPHA must exist"
    assert 0.0 <= config.MSE_ALPHA <= 1.0, \
        f"MSE_ALPHA must be in [0, 1], got {config.MSE_ALPHA}"
    assert config.DILATE_GAMMA > 0.0, \
        f"DILATE_GAMMA must be > 0, got {config.DILATE_GAMMA}"
    assert config.DILATE_TDI_FD_EPS > 0.0, \
        f"DILATE_TDI_FD_EPS must be > 0, got {config.DILATE_TDI_FD_EPS}"

    assert hasattr(config, 'KENDALL_LOGVAR_INIT'), \
        "KENDALL_LOGVAR_INIT must exist (learned Kendall-Gal weighting restored)"
    for gone in ('MEDIAN_SMOOTHNESS_ENABLED', 'MEDIAN_SMOOTHNESS_WEIGHT'):
        assert not hasattr(config, gone), \
            f"config.{gone} must be deleted (L_smooth penalty removed entirely)"

    for gone in ('TILDEQ_ALPHA', 'TILDEQ_GAMMA',
                 'MEDIAN_SEAM_PENALTY_ENABLED', 'MEDIAN_SEAM_PENALTY_WEIGHT',
                 'PINBALL_LOSS_WEIGHT', 'DILATE_LOSS_WEIGHT'):
        assert not hasattr(config, gone), f"config.{gone} must be deleted"

    # τ is selectable: any QUANTILE_LEVELS entry on the right side, so assert
    # membership and side, not an index
    from config import QUANTILE_LEVELS
    assert (config.HYPO_ALARM_QUANTILE_TAU in QUANTILE_LEVELS
            and config.HYPO_ALARM_QUANTILE_TAU < 0.5), \
        f"HYPO_ALARM_QUANTILE_TAU must be a lower-half level, got {config.HYPO_ALARM_QUANTILE_TAU}"
    assert (config.HYPER_ALARM_QUANTILE_TAU in QUANTILE_LEVELS
            and config.HYPER_ALARM_QUANTILE_TAU > 0.5), \
        f"HYPER_ALARM_QUANTILE_TAU must be an upper-half level, got {config.HYPER_ALARM_QUANTILE_TAU}"
    assert config.HYPO_ALARM_QUANTILE_TAU == 0.25
    assert config.HYPER_ALARM_QUANTILE_TAU == 0.75
    # forgives precision only; recall stays strict
    assert hasattr(config, 'EXCURSION_PRECISION_TOLERANCE_MGDL'), \
        "EXCURSION_PRECISION_TOLERANCE_MGDL must exist (precision forgiveness band)"
    assert config.EXCURSION_PRECISION_TOLERANCE_MGDL >= 0.0

    assert not hasattr(config, 'TEMPORAL_FEAT_START'), \
        "config.TEMPORAL_FEAT_START must be deleted (temporal inputs removed)"

    assert not hasattr(config, 'BG_INPUT_RISK_SPACE'), \
        "config.BG_INPUT_RISK_SPACE must be deleted (risk-space input is unconditional)"

    assert config.ARCH_VERSION == 'risk-v6', \
        f"ARCH_VERSION must be 'risk-v6', got {config.ARCH_VERSION!r}"
    # the head is one MLP over spline step states: width and init scale are the only knobs
    assert sorted(n for n in dir(config) if n.startswith('BG_HEAD_')) == \
        ['BG_HEAD_HIDDEN', 'BG_HEAD_INIT_SCALE'], \
        f"stale BG_HEAD_* constant: {[n for n in dir(config) if n.startswith('BG_HEAD_')]}"
    assert config.LOSS_SCHEMA == 'kendall-pinball-dilate-mse-v4', \
        f"LOSS_SCHEMA must be 'kendall-pinball-dilate-mse-v4', got {config.LOSS_SCHEMA!r}"
    print(f"\n[DUMP] redesign | ROPE_BASE={config.ROPE_BASE} "
          f"DILATE_ALPHA={config.DILATE_ALPHA} DILATE_GAMMA={config.DILATE_GAMMA} "
          f"ARCH={config.ARCH_VERSION} LOSS={config.LOSS_SCHEMA} ✓")
