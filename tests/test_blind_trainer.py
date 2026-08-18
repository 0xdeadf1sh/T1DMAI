"""``train_blind.py`` — the seven ways it must differ from ``train.py``, tested.

The fork is a copy, so nothing keeps it honest by construction. Five of its seven
divergences can fail silently and are pinned here:

* both PROTOCOL forwards must blind the spans they mask. They place their own
  masked sets after the dataset has built the sample, so the dataset's blinding
  does not reach them — and an announced dose surviving there would score the
  blind model on a conditioned task while every number stayed plausible;
* the long-horizon roll must announce NOTHING;
* the same roll's OBSERVED context must un-blind the doses the mask withheld.
  Blinding is a property of the objective and the roll's context is history, so
  restoring the withheld bg without the doses beside it does not leave a gap —
  it asserts that a meal and a bolus did not happen, and the night rows are
  scored on a patient who did not eat;
* no ``cf_*`` column may survive, in the CSV header or on the page;
* the run must write to ``checkpoints_blind/`` and ``logs_blind/`` only. A
  conditioned run may be live in the same checkout, and its logs are overwritten
  on the first step of whatever starts next.

Plus the provenance guard in ``calibrate_conformal.py``, which is the only thing
standing between a blind checkpoint and a conditioned band fit: no parameter shape
records the policy, so the weights load either way.
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
    """A collated batch from a BLIND dataset — what the fork's validation sees."""
    ds = T1DMDataset(master_seed=SEED, total_steps=N_SAMPLES, batch_size=1,
                     normalization_stats=stats, patient_uniform_sample_prob=0.0,
                     blind=True)
    return collate_fn([ds[i] for i in range(N_SAMPLES)])


def _dose_cells(patches: torch.Tensor, rows, cols) -> dict[int, torch.Tensor]:
    """``{feat: (n, len(cols), PATCH_SIZE)}`` for the three dose channels."""
    return {f: patches[rows][:, cols, f::N_INPUT_FEATURES] for f in MASKABLE_FEATS}


def test_the_forecast_protocol_blinds_the_zone_it_masks(blind_batch, stats):
    """Every dose cell of the trailing forecast zone carries the fill.

    ``_forecast_protocol`` masks ``[T - PREDICTION_PATCHES, T)`` on a window the
    sampler masked somewhere else. Those trailing patches are usually VISIBLE in
    the sample and so carry their true announced doses; if the protocol withholds
    only bg there, the whole horizon-keyed clinical suite — every ``bg_rmse_@h``,
    every coverage row, the alarm curve — is measured on a conditioned forecast
    that this model never trains on.
    """
    fill = zero_dose_fill(stats)
    fc = train_blind._forecast_protocol(
        blind_batch['patches'], blind_batch['bg_formula_data']['mask_idx'].long(),
        blind_batch['bg_formula_data']['valid'], blind_batch['n_context_patches'],
        fill)
    assert fc is not None, "no row survived the forecast protocol's anchor filter"

    T = blind_batch['patches'].shape[1]
    zone = list(range(T - PREDICTION_PATCHES, T))
    rows = torch.arange(fc['patches'].shape[0])
    print(f"\n[DUMP] forecast protocol: {len(rows)}/{N_SAMPLES} rows kept, "
          f"T={T}, zone={zone[0]}..{zone[-1]}")

    # The conditioned protocol on the SAME batch is what gives this a subject: it
    # is the input the blind model must NOT be validated on.
    fc_announced = train._forecast_protocol(
        blind_batch['patches'], blind_batch['bg_formula_data']['mask_idx'].long(),
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

    # "and only there": outside the zone this protocol masks, its window must be
    # the announced protocol's byte for byte. Blinding the whole window satisfies
    # every assertion above and destroys the context the forecast reads.
    T_all = list(range(T - PREDICTION_PATCHES))
    assert torch.equal(fc['patches'][:, T_all], fc_announced['patches'][:, T_all]), (
        "the forecast protocol changed a patch outside its masked zone")
    print(f"[DUMP] {len(T_all)} context patches identical to the announced build")


def test_the_infill_protocol_blinds_the_spans_it_masks(blind_batch, stats):
    """The interior spans this protocol places are blinded, and only they are.

    ``_infill_protocol`` REPLACES the training mask: it restores every withheld bg
    and then draws its own spans. So the blinding has to follow ITS masked set,
    not the sample's — a patch it reveals keeps what the sample left there, and a
    patch it masks is blinded whether or not the sampler had masked it.
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

    # "and ONLY they are" — the half the first draft of this test left out. An
    # implementation that wiped the dose channels of the whole window satisfies
    # everything above, and would blind the visible evidence the infill task is
    # defined against. The revealed patches must still carry what the sample left
    # there, which is what the untouched input says.
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


def test_the_rolling_validation_announces_nothing(blind_batch, stats, monkeypatch):
    """``predict_rolling`` is called with no ``overrides_fn``.

    ``train.py`` passes the sample's true future carb / insulin / exercise here to
    tame a zero-basal OOD runaway. Passing it in the blind fork would condition
    the long-horizon rows on a plan the model cannot read at any other horizon,
    so ``night_bg_rmse_*`` would answer a question no other row on the page does.

    Observed at the call, not read off the source: the argument is what matters,
    and a builder left in place but returning None would look identical in a grep.
    """
    import inference
    seen: list[dict] = []

    def _stub(model, context, **kw):
        seen.append(kw)
        n = PREDICTION_PATCHES * PATCH_SIZE
        return {'pred_bg': torch.full((n,), 120.0),
                'bands': torch.full((PREDICTION_PATCHES, PATCH_SIZE, 7), 120.0)}

    monkeypatch.setattr(inference, 'predict_rolling', _stub)

    ds = T1DMDataset(master_seed=SEED, total_steps=N_SAMPLES, batch_size=1,
                     normalization_stats=stats, patient_uniform_sample_prob=0.0,
                     blind=True)
    agg: dict[str, float] = {}
    train_blind._accumulate_long_horizon_bg_metrics(
        None, [ds[i] for i in range(N_SAMPLES)], stats, torch.device('cpu'),
        n_rolls=2, agg=agg)

    assert seen, "no roll ran — the metric this test is about was never reached"
    print(f"\n[DUMP] {len(seen)} rolls, kwargs {sorted(seen[0])}")
    for kw in seen:
        assert kw.get('overrides_fn') is None, (
            f"the blind roll announced a plan: overrides_fn={kw['overrides_fn']!r}")


def test_the_roll_s_observed_context_restores_the_doses_the_mask_blinded(stats):
    """``_observed_patches`` un-blinds feats 1-3, not feat 0 alone.

    The roll's context is the patient's OBSERVED history, which is why the
    function restores the withheld bg at all. Under the blind policy the same
    patches had their doses overwritten with the ``zero_dose_fill`` constant, so
    restoring bg and stopping there does not leave a gap — it leaves an
    ASSERTION, that a half-hour was seen and carried no carbs and no insulin,
    over spans that carried a meal or a bolus. The roll conditions on it and the
    ``night_bg_rmse_*`` rows then answer a question about a patient who did not
    eat.

    Pinned against the ANNOUNCED sample at the same seed, which is the ground
    truth for what those cells held: ``tests/test_blind_dataset.py`` establishes
    that the two policies differ in the masked dose cells and nowhere else, so
    equality here is the restore being exact rather than merely non-constant.
    """
    kw = dict(master_seed=SEED, total_steps=N_SAMPLES, batch_size=1,
              normalization_stats=stats, patient_uniform_sample_prob=0.0)
    blind_ds = T1DMDataset(blind=True, **kw)
    plain_ds = T1DMDataset(blind=False, **kw)
    fill = zero_dose_fill(stats)

    checked = 0
    for i in range(N_SAMPLES):
        blind_s, plain_s = blind_ds[i], plain_ds[i]
        rows = blind_s['bg_formula_data'].get('unblinded_dose_rows')
        assert rows is not None and len(rows) > 0, (
            'the blind sample carries no un-blinded dose rows, so the restore '
            'below has nothing to work from')
        assert 'unblinded_dose_rows' not in plain_s['bg_formula_data'], (
            'the announced sample grew a blind-only key; the default path is '
            'digested byte-for-byte by tests/test_blind_dataset.py')

        r = torch.as_tensor(np.asarray(rows)).long()
        before = torch.as_tensor(np.asarray(blind_s['patches'])).float()
        after = train_blind._observed_patches(blind_s, stats)
        truth = torch.as_tensor(np.asarray(plain_s['patches'])).float()

        for feat in MASKABLE_FEATS:
            cols = slice(feat, None, N_INPUT_FEATURES)
            cells = before[r][:, cols]
            assert torch.allclose(cells, torch.full_like(cells, fill[feat])), (
                f'feat {feat} on a masked patch is not at the blind fill before '
                f'the restore, so this test is not measuring the restore')
            assert torch.equal(after[r][:, cols], truth[r][:, cols]), (
                f'feat {feat} was not restored to what the patient actually did: '
                f'the roll conditions on a history that denies the dose')

        keep = torch.ones(before.shape[0], dtype=torch.bool)
        keep[r] = False
        for feat in MASKABLE_FEATS:
            cols = slice(feat, None, N_INPUT_FEATURES)
            assert torch.equal(after[keep][:, cols], before[keep][:, cols]), (
                f'the restore reached feat {feat} on a VISIBLE patch')
        checked += 1

    # The announced fork has nothing to undo and must not have grown a restore.
    plain_out = train._observed_patches(plain_s, stats)
    plain_in = torch.as_tensor(np.asarray(plain_s['patches'])).float()
    for feat in MASKABLE_FEATS:
        cols = slice(feat, None, N_INPUT_FEATURES)
        assert torch.equal(plain_out[:, cols], plain_in[:, cols]), (
            f'train.py moved feat {feat}; it never blinds one')
    print(f"\n[DUMP] roll context | {checked} blind samples: masked doses "
          f"restored exactly, visible cells and the announced fork untouched ✓")


def test_no_counterfactual_column_survives(blind_batch):
    """No ``cf_*`` column in the header, and no counterfactual row on the page.

    The probe perturbs the announced doses of a masked span. A blind model reads
    a constant there, so every row would report the perturbation's own absence as
    a model property — ``cf_insulin_dir`` at chance, which in ``train.py`` is the
    signature of a model that has stopped responding to insulin.

    Both halves are asserted: absent here AND present in ``train.py``, off the
    same input. A rename would otherwise pass the first half on nothing.
    """
    blind_cols = [name for name, _ in train_blind._val_log_columns()]
    plain_cols = [name for name, _ in train._val_log_columns()]
    assert not [c for c in blind_cols if c.startswith('cf_')], (
        f"blind header still carries {[c for c in blind_cols if c.startswith('cf_')]}")
    dropped = [c for c in plain_cols if c.startswith('cf_')]
    assert dropped, "train.py's header has no cf_* column — this test has no subject"

    # SEQUENCE equality, not set equality. A set comparison cannot see a
    # duplicated column, and a duplicate is exactly what a copy-paste edit to
    # this list produces: the two logs then differ in width and every column
    # after the duplication sits at a different index, so a positional reader
    # silently reports one metric under another's name. That defect was here.
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

    # Feed the values that WOULD render, so the absence is evidence.
    synthetic = {'cf_n': 96, 'cf_carb_dir': 0.94, 'cf_insulin_dir': 0.88,
                 'cf_insulin_monotonic': 0.71, 'cf_carb_dbg': 12.4,
                 'cf_insulin_dbg': -9.1, 'cf_hypo_rescue': 0.5,
                 'cf_hyper_rescue': 0.4, 'cf_carb_monotonic': 0.8,
                 'cf_hypo_n': 20, 'cf_hyper_n': 18}
    blind_page = train_blind._render_validation_table(1, dict(synthetic))
    plain_page = train._render_validation_table(1, dict(synthetic))
    for label in ('Counterfactual', 'carb→BG direction', 'insulin→BG direction',
                  'insulin monotonic'):
        assert label in plain_page, (
            f"train.py's table does not render {label!r} — no subject")
        assert label not in blind_page, (
            f"the blind table still renders {label!r}")


def test_the_fork_never_writes_to_the_conditioned_run_s_directories():
    """Not one ``checkpoints/`` or ``logs/`` path literal survives in the fork.

    This is the failure that costs a run rather than a number: ``train.py`` opens
    its CSVs in ``'w'`` mode on step 0 and writes ``checkpoints/t1dmai_best.pt``
    on every improvement, so a blind run started beside a live conditioned one
    would take its logs and its best checkpoint with it. Every path literal is
    checked, not the four that were edited.
    """
    src = open(train_blind.__file__).read()
    bad = sorted({
        node.value for node in ast.walk(ast.parse(src))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
        and re.search(r"(^|[^_a-z])(checkpoints|logs)/", node.value)
    })
    assert not bad, f"the blind trainer writes conditioned-run paths: {bad}"
    # And the blind ones are actually there — an empty file would also pass above.
    for expected in ('checkpoints_blind', 'logs_blind'):
        assert expected in src, f"{expected} appears nowhere in the fork"


def test_the_conformal_fit_blinds_the_interior_span(stats):
    """``calibrate_conformal --blind`` withholds the infill span's doses.

    That span sits INSIDE the context, so its doses arrive with the context
    rather than through an override and ``inference._build_patches_tensor``
    withholds only bg there. Its residuals are what the (unshipped) infill delta
    is fitted on, and a delta fitted on a conditioned span does not describe the
    blind model's interval.
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

    This delta is the one that ships: ``metrics/core/report.py`` lifts it onto the
    model and every ``inference.predict`` handed it applies it. Fitted under the
    wrong policy it is not a wrong figure on a page but a wrong interval on the
    phone, and nothing downstream can tell.
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


# ---------------------------------------------------------------------------
# The provenance guard
# ---------------------------------------------------------------------------

def _guard_module():
    """``calibrate_conformal.py`` — where the policy guard now lives."""
    import calibrate_conformal
    return calibrate_conformal


def _ckpt(policy) -> dict:
    """A checkpoint's ``training_config``, with or without the policy key."""
    tc = {'mask_span_lengths': [1], 'max_masked_patches': 12}
    if policy is not None:
        tc['masked_channel_policy'] = policy
    return {'training_config': tc}


def test_a_blind_checkpoint_is_refused_by_a_conditioned_fit():
    """The direction that actually ships: ``train_blind.py`` writes 'blind', and
    a fit without ``--blind`` announces doses on the span it predicts."""
    F = _guard_module()
    with pytest.raises(SystemExit) as exc:
        F._check_policy(_ckpt(masked_channel_policy(blind=True)), blind=False)
    msg = str(exc.value)
    print(f"\n[DUMP] refusal:\n{msg}")
    assert 'blind' in msg and 'announced' in msg, (
        "the refusal names neither policy — the operator cannot act on it")


def test_a_conditioned_checkpoint_is_refused_by_a_blind_fit():
    """The guard is an equality, not a one-sided blacklist.

    A guard written as "refuse 'blind'" would pass every test above and let the
    opposite mismatch through — a band fitted blind shipping on a conditioned
    checkpoint.
    """
    F = _guard_module()
    for stored in (masked_channel_policy(blind=False), None):
        with pytest.raises(SystemExit):
            F._check_policy(_ckpt(stored), blind=True)


def test_an_unstamped_checkpoint_reads_as_announced():
    """Absence is information, not ignorance.

    The key was introduced WITH the blind trainer and a blind run always stamps
    it, so a checkpoint without it was trained under the announced policy — which
    is what every checkpoint on disk today is. Reading absence as "unknown" would
    skip the check on exactly the population it has to accept.
    """
    F = _guard_module()
    assert F._check_policy(_ckpt(None), blind=False) == masked_channel_policy(blind=False)
    assert F._check_policy({}, blind=False) == masked_channel_policy(blind=False)
