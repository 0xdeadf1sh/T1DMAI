"""``train_blind.py`` — the seven ways it must differ from ``train.py``, tested.

The fork is a copy, so nothing keeps it honest by construction. Four of its seven
divergences can fail silently and are pinned here:

* both PROTOCOL forwards must blind the spans they mask. They place their own
  masked sets after the dataset has built the sample, so the dataset's blinding
  does not reach them — and an announced dose surviving there would score the
  blind model on a conditioned task while every number stayed plausible;
* the long-horizon roll must announce NOTHING;
* no ``cf_*`` column may survive, in the CSV header or on the page;
* the run must write to ``checkpoints_blind/`` and ``logs_blind/`` only. A
  conditioned run may be live in the same checkout, and its logs are overwritten
  on the first step of whatever starts next.

Plus the provenance guard in ``finetune/finetune.py``, which is the only thing
standing between a blind checkpoint and a conditioned fine-tune: no parameter
shape records the policy, so the weights load either way.
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

    This delta is the one that ships: ``realdata/report.py`` lifts it onto the
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

def _finetune_module():
    """``finetune/finetune.py``, imported the way the repo imports it.

    ``finetune/`` carries no ``__init__.py``, so ``from finetune import finetune``
    would bind the name ``finetune`` to the implicit NAMESPACE PACKAGE — and
    ``finetune_personal.py`` does a plain ``import finetune`` expecting the
    module. Whichever ran first would decide what the other got, and
    ``tests/test_personal_adapter.py`` would fail on an attribute the package has
    no reason to carry. Same path insert as ``_personal_module`` there.
    """
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'finetune'))
    import finetune
    return finetune


def _ckpt(policy) -> dict:
    """A checkpoint's ``training_config``, with or without the policy key."""
    tc = {'mask_span_lengths': [1], 'max_masked_patches': 12}
    if policy is not None:
        tc['masked_channel_policy'] = policy
    return {'training_config': tc}


def test_a_blind_checkpoint_is_refused_by_a_conditioned_finetune():
    """The direction that actually ships: ``train_blind.py`` writes 'blind', and
    ``finetune.py`` announces doses on the span it predicts."""
    F = _finetune_module()
    with pytest.raises(SystemExit) as exc:
        F._policy_guard(_ckpt(masked_channel_policy(blind=True)))
    msg = str(exc.value)
    print(f"\n[DUMP] refusal:\n{msg}")
    assert 'blind' in msg and 'announced' in msg, (
        "the refusal names neither policy — the operator cannot act on it")


def test_a_conditioned_checkpoint_is_refused_by_a_blind_finetune(monkeypatch):
    """The guard is an equality, not a one-sided blacklist.

    Nothing fine-tunes blind today, so this direction is exercised by moving the
    live policy rather than by a second script. A guard written as "refuse
    'blind'" would pass every test above and let the opposite mismatch through
    the day a blind fine-tune exists.
    """
    F = _finetune_module()
    monkeypatch.setattr(F, 'LIVE_MASKED_CHANNEL_POLICY',
                        masked_channel_policy(blind=True))
    for stored in (masked_channel_policy(blind=False), None):
        with pytest.raises(SystemExit):
            F._policy_guard(_ckpt(stored))


def test_an_unstamped_checkpoint_reads_as_announced():
    """Absence is information, not ignorance.

    The key was introduced WITH the blind trainer and a blind run always stamps
    it, so a checkpoint without it was trained under the announced policy — which
    is what every checkpoint on disk today is. Reading absence as "unknown" would
    skip the check on exactly the population it has to accept.
    """
    F = _finetune_module()
    F._policy_guard(_ckpt(None))                      # must not raise
    F._policy_guard({})                               # no training_config at all
    assert (F._stored_masked_channel_policy({})
            == masked_channel_policy(blind=False))
    assert F.LIVE_MASKED_CHANNEL_POLICY == masked_channel_policy(blind=False), (
        "finetune.py announces doses on the span it predicts, so its live policy "
        "is the announced one")
