"""Guard the training values migrated from the deleted training_config.json.

The JSON tier used to carry the real run hyperparameters and silently override
config.py; with it gone, config.py is the single source of truth and nothing
validates the values anymore. These tests assert the migrated values exist and
are in sane ranges — so a future edit that reverts a value fails loudly instead
of silently training at the wrong LR/batch.
"""


def test_active_config_migrated_values():
    """The active config carries the migrated live values.

    ``NUM_WORKERS`` is not among them: it is a property of the machine the run is
    on, not of the training recipe, and pinning it here fails on every host that
    is not the one the value was written for. Its range is checked below.
    """
    import config
    assert config.BATCH_SIZE == 512
    assert config.PATIENT_UNIFORM_SAMPLE_PROB == 0.0
    assert config.PREDICTION_HORIZON_HOURS == 2
    assert config.NIGHT_LONG_HORIZON_HOURS == 8
    print(f"[DUMP] active config: B={config.BATCH_SIZE} workers={config.NUM_WORKERS} "
          f"muon={config.MUON_LR} adam={config.ADAM_LR} det={config.DETERMINISTIC}")


def test_migrated_values_in_sane_ranges():
    import config
    assert 0 < config.BATCH_SIZE <= 4096
    assert 0 < config.NUM_WORKERS <= 256
    assert 0.0 <= config.PATIENT_UNIFORM_SAMPLE_PROB <= 1.0
    assert 0.0 < config.MUON_LR < 1.0
    assert 0.0 < config.ADAM_LR < 1.0
    assert 0.0 < config.LR_MIN_RATIO <= 1.0


def test_redesign_provenance_and_knobs_migrated():
    """The risk-v4 redesign advances the provenance stamps; DILATE is restored on
    the risk-v4 arch — ARCH_VERSION / LOSS_SCHEMA name the architecture and loss,
    RoPE drops to base 1000, and the DILATE knobs replace the retired TILDE-Q ones."""
    import config
    assert config.ARCH_VERSION == 'risk-v4'
    assert config.LOSS_SCHEMA == 'kendall-pinball-dilate-v3'
    assert config.ROPE_BASE == 1000
    assert (hasattr(config, 'DILATE_ALPHA') and hasattr(config, 'DILATE_GAMMA')
            and hasattr(config, 'DILATE_TDI_FD_EPS'))
    assert hasattr(config, 'KENDALL_LOGVAR_INIT'), \
        "the learned Kendall-Gal combine is restored (KENDALL_LOGVAR_INIT must exist)"
    for gone in ('PINBALL_LOSS_WEIGHT', 'DILATE_LOSS_WEIGHT'):
        assert not hasattr(config, gone), \
            f"{gone} must be retired (the static combine is replaced by Kendall-Gal)"
    assert not hasattr(config, 'TILDEQ_ALPHA'), "TILDEQ_ALPHA must be retired"
    assert config.N_INPUT_FEATURES == 5, \
        ("temporal inputs removed, exercise_equiv at feat 3, bg_masked at feat 4 "
         "-> 5 features")
    print(f"[DUMP] redesign provenance | ARCH={config.ARCH_VERSION} "
          f"LOSS={config.LOSS_SCHEMA} ROPE_BASE={config.ROPE_BASE}")
