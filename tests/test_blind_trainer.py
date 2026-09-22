"""``train_blind.py`` is a copy of ``train.py``; nothing keeps it honest by construction.

Pins: both protocol forwards blind after the sample is built; no cf_* column survives;
writes only to *_blind dirs.
"""

import ast
import re

import numpy as np
import pytest
import torch

from config import (
    MASKABLE_FEATS, N_INPUT_FEATURES, PATCH_SIZE, PREDICTION_PATCHES,
)
from data import T1DMDataset, collate_fn, zero_dose_fill, masked_channel_policy
from normalization import load_normalization_stats

import train
import train_blind

N_SAMPLES = 4
SEED = 20_260_815


@pytest.fixture(scope='module')
def stats():
    return load_normalization_stats()


@pytest.fixture(scope='module')
def blind_batch(stats):
    """What the fork's validation sees."""
    ds = T1DMDataset(master_seed=SEED, total_steps=N_SAMPLES, batch_size=1,
                     normalization_stats=stats, patient_uniform_sample_prob=0.0,
                     blind=True)
    return collate_fn([ds[i] for i in range(N_SAMPLES)])


def _dose_cells(patches: torch.Tensor, rows, cols) -> dict[int, torch.Tensor]:
    """``{feat: (n, len(cols), PATCH_SIZE)}`` for the three dose channels."""
    return {f: patches[rows][:, cols, f::N_INPUT_FEATURES] for f in MASKABLE_FEATS}


def test_the_forecast_protocol_blinds_the_zone_it_masks(blind_batch, stats):
    """``_forecast_protocol`` masks ``[T-PREDICTION_PATCHES, T)`` and must fill its doses.

    The zone is announced here, as a batch off any other sampler carries it. Withhold only
    bg and the clinical suite measures a conditioned forecast this model never trains on.
    """
    fill = zero_dose_fill(stats)
    T = blind_batch['patches'].shape[1]
    zone = list(range(T - PREDICTION_PATCHES, T))
    # The boundary tail is masked at the dataset, so blind already filled it; announce a dose.
    patches = blind_batch['patches'].clone()
    for feat in MASKABLE_FEATS:
        patches[:, zone, feat::N_INPUT_FEATURES] = float(fill[feat]) + 1.0
    fc = train_blind._forecast_protocol(
        patches, blind_batch['bg_formula_data']['mask_idx'].long(),
        blind_batch['bg_formula_data']['valid'], blind_batch['n_context_patches'],
        fill)
    assert fc is not None, "no row survived the forecast protocol's anchor filter"

    rows = torch.arange(fc['patches'].shape[0])
    print(f"\n[DUMP] forecast protocol: {len(rows)}/{N_SAMPLES} rows kept, "
          f"T={T}, zone={zone[0]}..{zone[-1]}")

    # The conditioned protocol on the SAME batch: the input the blind model must NOT validate on.
    fc_announced = train._forecast_protocol(
        patches, blind_batch['bg_formula_data']['mask_idx'].long(),
        blind_batch['bg_formula_data']['valid'], blind_batch['n_context_patches'])
    assert fc_announced is not None

    moved = False
    for feat, cells in _dose_cells(fc['patches'], rows, zone).items():
        assert cells.shape[-1] == PATCH_SIZE
        expected = torch.full_like(cells, float(fill[feat]))
        assert torch.equal(cells, expected), (
            f"feat {feat} in the forecast zone is not the fill: max|delta| "
            f"{float((cells - expected).abs().max()):.6g}")
        was = fc_announced['patches'][rows][:, zone, feat::N_INPUT_FEATURES]
        moved |= not torch.equal(was, cells)
    assert moved, (
        "the announced protocol built the same tensor — this batch announces no "
        "dose in its forecast zone, so the assertions above have no subject")

    # And only there: blinding the whole window satisfies the above and destroys the context.
    T_all = list(range(T - PREDICTION_PATCHES))
    assert torch.equal(fc['patches'][:, T_all], fc_announced['patches'][:, T_all]), (
        "the forecast protocol changed a patch outside its masked zone")
    print(f"[DUMP] {len(T_all)} context patches identical to the announced build")


def test_the_infill_protocol_blinds_the_spans_it_masks(blind_batch, stats):
    """``_infill_protocol`` REPLACES the training mask, restoring bg and drawing its own spans.

    Blinding follows ITS masked set: a revealed patch keeps what the sample left there; a
    masked patch is blinded whether the sampler masked it or not.
    """
    fill = zero_dose_fill(stats)
    bf = blind_batch['bg_formula_data']
    infill = train_blind._infill_protocol(
        blind_batch['patches'], bf['mask_idx'].long(), bf['valid'],
        blind_batch['targets'].float(), blind_batch['n_context_patches'],
        stats, np.random.default_rng(0), fill)
    if infill is None:
        pytest.skip("no row's context could hold the infill protocol")

    from data import BG_MASKED_FEAT
    p = infill['patches']
    masked = p[:, :, BG_MASKED_FEAT::N_INPUT_FEATURES][:, :, 0] > 0.5   # (n, T)
    n_masked = int(masked.sum())
    assert n_masked > 0
    print(f"\n[DUMP] infill protocol: {p.shape[0]} rows, {n_masked} masked patches")

    for feat in MASKABLE_FEATS:
        block = p[:, :, feat::N_INPUT_FEATURES]                          # (n, T, S)
        got = block[masked]
        expected = torch.full_like(got, float(fill[feat]))
        assert torch.equal(got, expected), (
            f"feat {feat} on a patch this protocol masks is not the fill")

    # And ONLY they: wiping the window satisfies the above while blinding the infill's evidence.
    _revealed = ~masked
    for feat in MASKABLE_FEATS:
        got = p[:, :, feat::N_INPUT_FEATURES][_revealed]
        was = blind_batch['patches'][
            torch.tensor([b for b, _ms, _c in infill['sets']])
        ][:, :, feat::N_INPUT_FEATURES][_revealed]
        assert torch.equal(got, was), (
            f"feat {feat} moved on a patch this protocol REVEALS — the blinding "
            "reached past its own masked set")
    print(f"[DUMP] {int(_revealed.sum())} revealed patches unchanged")


def test_no_counterfactual_column_survives(blind_batch):
    """A blind model reads a perturbed dose as constant, so cf_insulin_dir would sit at
    chance — train.py's signature for a model that stopped responding to insulin.

    Asserted absent here AND present in train.py, or a rename passes on nothing.
    """
    blind_cols = [name for name, _ in train_blind._val_log_columns()]
    plain_cols = [name for name, _ in train._val_log_columns()]
    assert not [c for c in blind_cols if c.startswith('cf_')], (
        f"blind header still carries {[c for c in blind_cols if c.startswith('cf_')]}")
    dropped = [c for c in plain_cols if c.startswith('cf_')]
    assert dropped, "train.py's header has no cf_* column — this test has no subject"

    # SEQUENCE equality, not set: a set hides a duplicate column, which shifts every later index.
    assert len(blind_cols) == len(set(blind_cols)), (
        "the blind header repeats a column: "
        f"{sorted({c for c in blind_cols if blind_cols.count(c) > 1})}")
    assert len(plain_cols) == len(set(plain_cols)), (
        "train.py's header repeats a column: "
        f"{sorted({c for c in plain_cols if plain_cols.count(c) > 1})}")
    assert blind_cols == [c for c in plain_cols if not c.startswith('cf_')], (
        "the blind header is not train.py's minus the cf_* block, in order — "
        "the two runs' CSVs are no longer positionally comparable")
    print(f"\n[DUMP] {len(dropped)} cf_* columns dropped, "
          f"{len(blind_cols)} of {len(plain_cols)} columns kept, in order, "
          "no duplicates")

    # feed the values that WOULD render, so the absence is evidence
    synthetic = {'cf_n': 96, 'cf_carb_sign': 0.94, 'cf_insulin_sign': 0.88,
                 'cf_insulin_monotonic': 0.71,
                 'cf_carb_monotonic': 0.8, 'cf_carb_gain': 0.7, 'cf_insulin_gain': 0.6,
                 'cf_carb_linearity': 1.9,
                 'cf_insulin_linearity': 1.7, 'cf_insulin_linearity_ref': 1.8,
                 'cf_insulin_preaction_dbg': 0.3, 'cf_carb_onset_frac': 1.0,
                 'cf_insulin_onset_frac': 0.9, 'cf_carb_onset_lag_min': 5.0,
                 'cf_insulin_onset_lag_min': -5.0, 'cf_meal_coverage': 0.7,
                 'cf_meal_coverage_ref': 0.8}
    blind_page = train_blind._render_validation_table(1, dict(synthetic))
    plain_page = train._render_validation_table(1, dict(synthetic))
    for label in ('Counterfactual', 'carb sign', 'insulin sign',
                  'insulin monotonic', 'insulin gain', 'insulin pre-action',
                  'matched-bolus coverage'):
        assert label in plain_page, (
            f"train.py's table does not render {label!r} — no subject")
        assert label not in blind_page, (
            f"the blind table still renders {label!r}")


def test_the_fork_never_writes_to_the_conditioned_run_s_directories():
    """Not one ``checkpoints/`` or ``logs/`` path literal survives in the fork.

    train.py writes its CSVs and checkpoints/t1dmai_best.pt in place, so a blind run beside
    a live conditioned one takes its logs and best checkpoint with it.
    """
    src = open(train_blind.__file__).read()
    bad = sorted({
        node.value for node in ast.walk(ast.parse(src))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
        and re.search(r"(^|[^_a-z])(checkpoints|logs)/", node.value)
    })
    assert not bad, f"the blind trainer writes conditioned-run paths: {bad}"
    # an empty file would pass the check above
    for expected in ('checkpoints_blind', 'logs_blind'):
        assert expected in src, f"{expected} appears nowhere in the fork"


def test_the_conformal_fit_blinds_the_interior_span(stats):
    """The infill span sits INSIDE the context; its doses arrive with the context and
    inference._build_patches_tensor withholds only bg there. A delta fit with doses
    announced does not describe the blind model's interval.
    """
    import calibrate_conformal as C
    from config import MAX_CONTEXT_PATCHES

    fill = zero_dose_fill(stats)
    n_ctx = MAX_CONTEXT_PATCHES
    ctx = torch.rand(n_ctx, PATCH_SIZE, N_INPUT_FEATURES)
    out = C._blind_context(ctx, fill)

    span = range(C.INFILL_START_PATCH, C.INFILL_START_PATCH + C.INFILL_SPAN_LEN)
    outside = [p for p in range(n_ctx) if p not in span]
    print(f"\n[DUMP] infill span patches {span.start}..{span.stop - 1} of {n_ctx}")
    for feat in MASKABLE_FEATS:
        got = out[list(span), :, feat]
        assert torch.equal(got, torch.full_like(got, float(fill[feat]))), (
            f"feat {feat} in the infill span is not the fill")
    assert torch.equal(out[outside], ctx[outside]), (
        "the blind context changed a patch outside the infill span")
    assert torch.equal(out[list(span), :, 0], ctx[list(span), :, 0]), (
        "the blind context touched bg — that is the masked forward's business")


def test_the_conformal_fit_refuses_a_policy_it_was_not_asked_for():
    """The flag and the checkpoint's stamp must agree, both ways.

    This is the delta that SHIPS: fitted under the wrong policy it is a wrong interval
    on the phone, not a wrong figure on a page, and nothing downstream can tell.
    """
    import calibrate_conformal as C

    blind_ck = {'training_config': {'masked_channel_policy':
                                    masked_channel_policy(blind=True)}}
    plain_ck = {'training_config': {'masked_channel_policy':
                                    masked_channel_policy(blind=False)}}
    assert C._check_policy(blind_ck, blind=True) == masked_channel_policy(blind=True)
    assert C._check_policy(plain_ck, blind=False) == masked_channel_policy(blind=False)
    assert C._check_policy({}, blind=False) == masked_channel_policy(blind=False)
    for ck, blind in ((blind_ck, False), (plain_ck, True), ({}, True)):
        with pytest.raises(SystemExit):
            C._check_policy(ck, blind=blind)


def _guard_module():
    """``calibrate_conformal.py`` holds the policy guard."""
    import calibrate_conformal
    return calibrate_conformal


def _ckpt(policy) -> dict:
    """A checkpoint's ``training_config``, with or without the policy key."""
    tc = {'mask_span_lengths': [1], 'max_masked_patches': 12}
    if policy is not None:
        tc['masked_channel_policy'] = policy
    return {'training_config': tc}


def test_a_blind_checkpoint_is_refused_by_a_conditioned_fit():
    """The shipping direction: ``train_blind.py`` writes 'blind', and a fit without
    ``--blind`` announces doses on the span it predicts."""
    F = _guard_module()
    with pytest.raises(SystemExit) as exc:
        F._check_policy(_ckpt(masked_channel_policy(blind=True)), blind=False)
    msg = str(exc.value)
    print(f"\n[DUMP] refusal:\n{msg}")
    assert 'blind' in msg and 'announced' in msg, (
        "the refusal names neither policy — the operator cannot act on it")


def test_a_conditioned_checkpoint_is_refused_by_a_blind_fit():
    """An equality, not a one-sided blacklist: "refuse 'blind'" passes every test above
    and ships a blind-fitted band on a conditioned checkpoint."""
    F = _guard_module()
    for stored in (masked_channel_policy(blind=False), None):
        with pytest.raises(SystemExit):
            F._check_policy(_ckpt(stored), blind=True)


def test_an_unstamped_checkpoint_reads_as_announced():
    """Absence is information, not ignorance: a blind run always stamps the key, so an
    unstamped checkpoint was trained announced — as every checkpoint on disk was."""
    F = _guard_module()
    assert F._check_policy(_ckpt(None), blind=False) == masked_channel_policy(blind=False)
    assert F._check_policy({}, blind=False) == masked_channel_policy(blind=False)
