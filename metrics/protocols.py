"""The two fixed evaluation protocols, and the axes every masked-BG figure is reported on.

Validation does not follow the training mask distribution: a metric averaged over it is dominated by the easy
regime and improves for free. These two are the only comparable figures:

    protocol   mask                                     columns          baseline
    forecast   right edge, exactly PREDICTION_PATCHES   every existing   persistence
    infill     sampled interior spans                   infill_*, per d  LINEAR INTERPOLATION

FORECAST puts exactly ONE masked patch at each of d = 1, 2, 3, 4 per window, which is what makes per-``d``
calibration well populated (§2.10's Mondrian fit calibrates this protocol and gives infill a coarse one) and
keeps its columns element-for-element comparable with the historical tables. Its names are the EXISTING ones:
``metrics.core.suite.compute_suite`` computes them and nothing here restates one.

INFILL is scored against LINEAR INTERPOLATION between the bracketing visible BGs, NEVER persistence, which is
a forecasting baseline and against a two-sided task a strawman. Its columns take the ``infill_`` prefix and a
``d``, so the two protocols cannot be averaged by accident — ``column()`` refuses an infill column without one.

THE d AXIS — the distance in patches to the nearest visible evidence ON EITHER SIDE. Every masked-BG metric
bins on it, never on span length, which confounds one-sided and two-sided cases at equal difficulty, and never
on arm. Forecast @30/@60/@90/@120 IS d = 1..4 one-sided. ``data._mask_slots`` is the single definition.

POOLING IS FORBIDDEN. The sampler concentrates supervision at small ``d`` and on the two-sided case;
``SAMPLER_REFERENCE`` below is the only copy of the exact shares. A pooled masked-BG scalar averages a mask
distribution rather than a difficulty: it improves for free and must never become a selection metric.

EVERY RUN REPORTS, via ``RunReport``: its realised ``d`` histogram and mean masked-patch count against
``SAMPLER_REFERENCE`` — a departure means the sampler changed; its ``n_ctx`` (§3.25: headline numbers are
measured at ``MAX_CONTEXT_PATCHES``, a length training barely samples); and per cohort the KEPT and DROPPED
segment and window counts (§3.24). Eligibility is PINNED to the 24 h footprint, which is what fixes the
window set across runs: segments are cut at every CGM gap over 30 min and short ones dropped, so a wider
context would otherwise re-select the cohort.
"""
from __future__ import annotations

import math
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config import (                                     # noqa: E402
    PATCH_SIZE, PREDICTION_PATCHES, MAX_CONTEXT_PATCHES, MIN_CONTEXT_PATCHES,
    MASK_MAX_SPANS, MASK_RIGHT_EDGE_QUOTA, MASK_SPAN_LENGTHS, MAX_MASKED_PATCHES,
    _PATCHES_PER_HOUR,
    QUANTILE_LEVELS,
)
# slot layout, ``d`` rule and anchor rule: ONE definition, in data.py, shared with the training builder
from data import sample_mask_spans, _mask_slots          # noqa: E402

GRID_MIN = 5                                  # minutes
SPAN_STEPS = PREDICTION_PATCHES * PATCH_SIZE  # steps in one right-edge forecast span


# Selection thresholds, UNSET deliberately: §2.7's reference pretrain has not run, so no measured
# distribution exists to set a level against. Asking for one raises rather than returning a plausible number.
THRESHOLDS: dict[str, float] = {}


def threshold(name: str) -> float:
    """The selection threshold named ``name``, or a loud failure — a default here would be a guess."""
    if name in THRESHOLDS:
        return THRESHOLDS[name]
    raise LookupError(
        f"threshold '{name}' is UNSET: no reference pretrain exists yet, so "
        f"there is no run to read it from. Measure it, then add it to "
        f"protocols.THRESHOLDS — do not substitute an estimate."
    )


@dataclass(frozen=True)
class Protocol:
    """One fixed evaluation protocol: a mask rule, a column namespace, a baseline."""
    name: str
    prefix: str                # column-name prefix ('' keeps the historical names)
    baseline: str              # the ONLY baseline this protocol may be scored against
    sided: str                 # 'one-sided' | 'two-sided'

    def __str__(self) -> str:                       # pragma: no cover - display only
        return self.name


FORECAST = Protocol(
    name='forecast', prefix='', baseline='persistence', sided='one-sided')
INFILL = Protocol(
    name='infill', prefix='infill_', baseline='interpolation', sided='two-sided')
PROTOCOLS = (FORECAST, INFILL)

# The inference builder requires the whole future zone masked — a visible future patch would announce a
# fabricated reading — so the trailing span is mandatory, costs PREDICTION_PATCHES of MAX_MASKED_PATCHES,
# and rides UNSCORED: scoring it would put one-sided rows in the infill namespace.
INFILL_BUDGET_PATCHES = MAX_MASKED_PATCHES - PREDICTION_PATCHES
assert INFILL_BUDGET_PATCHES >= min(MASK_SPAN_LENGTHS), (
    f"the head's {MAX_MASKED_PATCHES} slots leave {INFILL_BUDGET_PATCHES} for "
    f"interior spans after the mandatory {PREDICTION_PATCHES}-patch forecast "
    f"tail — too few for the shortest span in MASK_SPAN_LENGTHS"
)


def reachable_d(protocol: Protocol) -> tuple[int, ...]:
    """The ``d`` bins a protocol can populate, derived from the geometry.

    FORECAST: no right neighbour, so slot ``j`` sits at ``d = j + 1`` and the span covers
    d = 1..PREDICTION_PATCHES, one patch each.
    INFILL: an interior span of length ``L`` is two-sided and caps at ``d = ceil(L / 2)``, so the set follows
    MASK_SPAN_LENGTHS and the interior budget. The two protocols do not even cover the same axis.
    """
    if protocol is FORECAST:
        return tuple(range(1, PREDICTION_PATCHES + 1))
    if protocol is INFILL:
        usable = [L for L in MASK_SPAN_LENGTHS if L <= INFILL_BUDGET_PATCHES]
        return tuple(range(1, max((L + 1) // 2 for L in usable) + 1))
    raise ValueError(f"unknown protocol {protocol!r}")


def column(protocol: Protocol, base: str, d: int | None = None) -> str:
    """Column name for ``base`` under ``protocol``.

    FORECAST keeps every EXISTING name and takes no ``d`` suffix: its reported horizons already ARE the d bins
    (``metrics.core.run_eval.horizon_d_patches``).
    INFILL takes the ``infill_`` namespace and REQUIRES a ``d``: without one the column is a pooled masked-BG
    scalar, which improves for free under any mask distribution weighted toward small d.
    """
    if protocol is FORECAST:
        if d is not None:
            raise ValueError(
                f"the forecast protocol carries the existing column names and "
                f"takes no d suffix: its reported horizons already ARE the d "
                f"bins (metrics.core.run_eval.horizon_d_patches). Asked for "
                f"'{base}' at d={d}."
            )
        assert not base.startswith(INFILL.prefix), (
            f"'{base}' is already in the infill namespace; the forecast "
            f"protocol must not claim a name from it"
        )
        return base
    if protocol is INFILL:
        if d is None:
            raise ValueError(
                f"infill column '{base}' must name its d. A pooled masked-BG "
                f"scalar improves for free — the sampler concentrates supervision "
                f"at small d (SAMPLER_REFERENCE) — so it must never become a "
                f"comparable figure or a selection metric."
            )
        allowed = reachable_d(INFILL)
        if int(d) not in allowed:
            raise ValueError(
                f"d={d} is unreachable under the infill protocol; interior "
                f"spans of MASK_SPAN_LENGTHS={MASK_SPAN_LENGTHS} within the "
                f"{INFILL_BUDGET_PATCHES}-patch budget cover d in {allowed}"
            )
        return f"{INFILL.prefix}{base}@d{int(d)}"
    raise ValueError(f"unknown protocol {protocol!r}")


@dataclass(frozen=True)
class MaskedSet:
    """One protocol's masked set over an ``n_ctx + PREDICTION_PATCHES`` window.

    ``spans`` is every masked span in the inference builder's order; ``scored`` the subset this protocol
    scores. They differ only for INFILL, where the mandatory trailing span rides unscored.
    ``mask_idx`` / ``valid`` / ``d`` / ``anchor_step`` are ``data._mask_slots``' own arrays, untouched.
    """
    protocol: Protocol
    n_ctx: int
    spans: tuple[tuple[int, int], ...]
    scored: tuple[tuple[int, int], ...]
    mask_idx: np.ndarray
    valid: np.ndarray
    d: np.ndarray
    anchor_step: np.ndarray
    scored_slot: np.ndarray

    @property
    def seq_len(self) -> int:
        return self.n_ctx + PREDICTION_PATCHES

    def scored_d(self) -> np.ndarray:
        """``d`` of every scored slot, in slot order."""
        return self.d[self.scored_slot]

    def scored_patches(self) -> np.ndarray:
        """Window-relative PATCH index of every scored slot, in slot order."""
        return self.mask_idx[self.scored_slot]

    def scored_rows(self) -> np.ndarray:
        """Row indices of the scored slots in ``inference.predict``'s output.

        ``predict`` returns one row per VALID slot in slot order, padded ones dropped; this maps a scored slot
        onto its row without assuming the rows are a trailing zone.
        """
        row_of_slot = np.cumsum(self.valid) - 1
        return row_of_slot[self.scored_slot]

    def scored_steps(self) -> np.ndarray:
        """Window-relative STEP index of every scored element, row-major.

        Length ``n_scored * PATCH_SIZE``, aligned element-for-element with ``median_bg`` at ``scored_rows()``.
        """
        base = self.scored_patches()[:, None] * PATCH_SIZE
        return (base + np.arange(PATCH_SIZE)[None, :]).ravel()

    def scored_step_d(self) -> np.ndarray:
        """``d`` of every scored element, row-major — the per-step d axis."""
        return np.repeat(self.scored_d(), PATCH_SIZE)


def _expand(protocol: Protocol, n_ctx: int,
            spans: Sequence[tuple[int, int]],
            scored: Sequence[tuple[int, int]]) -> MaskedSet:
    seq_len = n_ctx + PREDICTION_PATCHES
    spans_t = tuple((int(s), int(L)) for s, L in spans)
    scored_t = tuple((int(s), int(L)) for s, L in scored)
    mask_idx, valid, d, anchor_step = _mask_slots(list(spans_t), seq_len)
    scored_patches = {p for s, L in scored_t for p in range(s, s + L)}
    scored_slot = np.array(
        [bool(v) and int(i) in scored_patches for i, v in zip(mask_idx, valid)],
        dtype=bool)
    assert int(scored_slot.sum()) == sum(L for _s, L in scored_t), (
        f"{int(scored_slot.sum())} scored slots for scored spans {scored_t}")
    return MaskedSet(protocol=protocol, n_ctx=int(n_ctx), spans=spans_t,
                     scored=scored_t, mask_idx=mask_idx, valid=valid, d=d,
                     anchor_step=anchor_step, scored_slot=scored_slot)


def forecast_masked_set(n_ctx: int) -> MaskedSet:
    """The FORECAST protocol at ``n_ctx``: one right-edge span, whole context visible.

    One masked patch lands at each of d = 1..PREDICTION_PATCHES, so every window fills every bin once.
    """
    spans = [(int(n_ctx), PREDICTION_PATCHES)]
    ms = _expand(FORECAST, n_ctx, spans, spans)
    assert tuple(int(x) for x in ms.scored_d()) == reachable_d(FORECAST), (
        f"forecast d {ms.scored_d().tolist()} is not one-sided "
        f"1..{PREDICTION_PATCHES} — the right-edge span lost its geometry")
    return ms


def infill_masked_set(n_ctx: int, rng: np.random.Generator) -> MaskedSet:
    """The INFILL protocol at ``n_ctx``: sampled INTERIOR spans, plus the mandatory tail.

    Placement reuses ``data.sample_mask_spans`` over ``[1, n_ctx - 1)`` and shifts by one patch, so the
    stars-and-bars rule, the visible separator and the whole-vector rejection keep one definition. The shift
    is what makes this INFILL and not a mix: patch 0 stays visible, so no span is a backcast, and patch
    ``n_ctx - 1`` stays visible, so no interior span abuts the trailing forecast span. Every interior span is
    two-sided and ``d`` measures real observed evidence on both sides.
    The budget is ``MAX_MASKED_PATCHES - PREDICTION_PATCHES``, and a draw over it is rejected WHOLE, which
    reweights toward fewer spans than the training sampler — expected: this is a fixed protocol with its own
    reported ``d`` histogram, not a reproduction of the training mixture.
    """
    interior_len = int(n_ctx) - 2
    need = MAX_MASKED_PATCHES + MASK_MAX_SPANS - 1
    if interior_len < need:
        raise ValueError(
            f"n_ctx={n_ctx} leaves an interior of {interior_len} patches, too "
            f"short to hold {MASK_MAX_SPANS} spans totalling "
            f"{MAX_MASKED_PATCHES} with separators (needs {need})"
        )
    while True:
        drawn = sample_mask_spans(interior_len, rng)
        if sum(L for _s, L in drawn) <= INFILL_BUDGET_PATCHES:
            break
    interior = [(int(s) + 1, int(L)) for s, L in drawn]
    spans = interior + [(int(n_ctx), PREDICTION_PATCHES)]
    ms = _expand(INFILL, n_ctx, spans, interior)
    allowed = reachable_d(INFILL)
    assert all(int(x) in allowed for x in ms.scored_d()), (
        f"infill d {ms.scored_d().tolist()} left the reachable set {allowed} — "
        f"an interior span touched an edge")
    return ms


def persistence_baseline(anchor_bg: float, n_steps: int) -> np.ndarray:
    """FORECAST's baseline: the last observed reading held flat, as ``compute_suite`` already scores it."""
    return np.full(int(n_steps), float(anchor_bg), dtype=np.float64)


def interpolation_baseline(left_bg: float, right_bg: float,
                           n_steps: int) -> np.ndarray:
    """INFILL's baseline: linear interpolation between the bracketing visible BGs.

    ``left_bg`` is the left neighbour's last step, ``right_bg`` the right neighbour's first, so the
    ``n_steps`` withheld steps sit on ``n_steps + 1`` equal intervals.
    """
    n = int(n_steps)
    t = np.arange(1, n + 1, dtype=np.float64) / (n + 1)
    return float(left_bg) + (float(right_bg) - float(left_bg)) * t


def baseline_for(masked_set: MaskedSet, cgm: np.ndarray,
                 window_start: int) -> np.ndarray:
    """The protocol's own baseline over its scored steps -> ``(n_scored * PATCH_SIZE,)``, mg/dL.

    Aligned with ``masked_set.scored_steps()``; ``cgm`` is the truth over the trajectory the window is cut
    from and ``window_start`` its patch 0. The protocol picks the baseline, so a caller cannot pair them
    wrongly: infill takes interpolation ONLY.
    """
    cgm = np.asarray(cgm, dtype=np.float64)
    out: list[np.ndarray] = []
    for start, L in masked_set.scored:
        a = window_start + start * PATCH_SIZE
        b = a + L * PATCH_SIZE
        if masked_set.protocol is FORECAST:
            out.append(persistence_baseline(cgm[a - 1], b - a))
        elif masked_set.protocol is INFILL:
            out.append(interpolation_baseline(cgm[a - 1], cgm[b], b - a))
        else:                                          # pragma: no cover
            raise ValueError(f"unknown protocol {masked_set.protocol!r}")
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float64)


# Exact ENUMERATION of ``data.sample_mask_spans`` over n_ctx ~ U{MIN..MAX_CONTEXT_PATCHES}, both placement
# branches, by ``d_balance.d_distribution``; checked against 4e5 live draws at every quota, within 1 sigma at
# each ``d``. Not a measurement: a realised histogram that departs means the SAMPLER changed.
# Pinned to the knobs below, and ``sampler_reference_applies`` refuses the comparison once one moves.
SAMPLER_REFERENCE = {
    'max_context_patches': 336,
    'min_context_patches': 168,
    'prediction_patches': 4,
    'mask_max_spans': 3,
    'mask_span_lengths': (1, 2, 3, 4, 5, 6, 7, 8),
    'max_masked_patches': 12,
    'mask_right_edge_quota': 0.50,
    'share_pct': {1: 43.230, 2: 27.775, 3: 15.628, 4: 13.367},
    'patches_per_sample': {1: 3.2003, 2: 2.0561, 3: 1.1570, 4: 0.9895},
    'mean_masked': 7.4029,
}


def sampler_reference_applies() -> tuple[bool, str]:
    """Whether ``SAMPLER_REFERENCE`` still describes the live sampler.

    After a knob change it enumerates a different sampler, and the comparison would report a config edit as a
    departure.
    """
    live = {
        'max_context_patches': MAX_CONTEXT_PATCHES,
        'min_context_patches': MIN_CONTEXT_PATCHES,
        'prediction_patches': PREDICTION_PATCHES,
        'mask_max_spans': MASK_MAX_SPANS,
        'mask_span_lengths': tuple(MASK_SPAN_LENGTHS),
        'max_masked_patches': MAX_MASKED_PATCHES,
        'mask_right_edge_quota': MASK_RIGHT_EDGE_QUOTA,
    }
    diffs = [f"{k}: live {v} vs reference {SAMPLER_REFERENCE[k]}"
             for k, v in live.items() if v != SAMPLER_REFERENCE[k]]
    if diffs:
        return False, "; ".join(diffs)
    return True, "knobs match the enumerated reference"


class DHistogram:
    """Per-``d`` accumulator: shares, patches per sample, mean masked.

    Counts are kept PER SAMPLE, so every reported standard error is empirical, not assumed.
    """

    def __init__(self, label: str):
        self.label = label
        self.n_samples = 0
        self._sum: Counter = Counter()
        self._sumsq: Counter = Counter()

    def add(self, d_values: Iterable[int]) -> None:
        """Record one sample's scored ``d`` values."""
        self.n_samples += 1
        per = Counter(int(x) for x in np.asarray(list(d_values)).ravel())
        for d, c in per.items():
            self._sum[d] += c
            self._sumsq[d] += c * c

    def add_masked_set(self, masked_set: MaskedSet) -> None:
        self.add(masked_set.scored_d())

    @property
    def bins(self) -> tuple[int, ...]:
        return tuple(sorted(self._sum))

    @property
    def total_masked(self) -> int:
        return int(sum(self._sum.values()))

    def mean_masked(self) -> float:
        """Mean scored masked patches per sample."""
        return self.total_masked / self.n_samples if self.n_samples else float('nan')

    def patches_per_sample(self) -> dict[int, float]:
        n = max(self.n_samples, 1)
        return {d: self._sum[d] / n for d in self.bins}

    def patches_per_sample_sem(self) -> dict[int, float]:
        """Standard error of each per-``d`` mean, from the per-sample counts."""
        n = self.n_samples
        out: dict[int, float] = {}
        for d in self.bins:
            if n < 2:
                out[d] = float('nan')
                continue
            mean = self._sum[d] / n
            var = max(self._sumsq[d] / n - mean * mean, 0.0)
            out[d] = math.sqrt(var / n)
        return out

    def share_pct(self) -> dict[int, float]:
        tot = max(self.total_masked, 1)
        return {d: 100.0 * self._sum[d] / tot for d in self.bins}

    def format(self) -> str:
        pps = self.patches_per_sample()
        sem = self.patches_per_sample_sem()
        shr = self.share_pct()
        lines = [f"  d histogram [{self.label}] over {self.n_samples} samples "
                 f"({self.total_masked} scored patches, mean masked "
                 f"{self.mean_masked():.4f})",
                 "    d   share%    patches/sample  (sem)"]
        for d in self.bins:
            lines.append(f"    {d:<3} {shr[d]:7.3f}   {pps[d]:14.4f}  "
                         f"(±{sem[d]:.4f})")
        return "\n".join(lines)

    def compare_to_sampler_reference(self) -> str:
        """Realised figures against ``SAMPLER_REFERENCE``, in units of the run's own sem.

        No pass/fail level (§2.7's reference pretrain has not run); sem units separate a sampler change from
        sampling noise.
        """
        ok, why = sampler_reference_applies()
        if not ok:
            return ("  sampler reference does NOT apply — " + why +
                    "\n    (the pinned figures enumerate a different sampler; "
                    "re-enumerate before comparing)")
        pps = self.patches_per_sample()
        sem = self.patches_per_sample_sem()
        ref = SAMPLER_REFERENCE['patches_per_sample']
        lines = ["  vs SAMPLER_REFERENCE (exact enumeration at "
                 f"MAX_CONTEXT_PATCHES={SAMPLER_REFERENCE['max_context_patches']})",
                 "    d   realised   reference   delta      sem-units"]
        for d in sorted(ref):
            got, exp, s = pps.get(d, 0.0), ref[d], sem.get(d, float('nan'))
            z = (got - exp) / s if s and math.isfinite(s) and s > 0 else float('nan')
            lines.append(f"    {d:<3} {got:9.4f}  {exp:9.4f}  {got - exp:+9.4f}  "
                         f"{z:+9.2f}")
        exp_mean = SAMPLER_REFERENCE['mean_masked']
        got_mean = self.mean_masked()
        lines.append(f"    mean masked realised {got_mean:.4f} vs reference "
                     f"{exp_mean:.4f} (delta {got_mean - exp_mean:+.4f})")
        return "\n".join(lines)


def sampler_d_histogram(n_draws: int = 20000, seed: int = 0) -> DHistogram:
    """Draw from the TRAINING sampler and bin on ``d`` — the sampler audit.

    ``n_ctx`` is drawn as training draws it, uniform over ``[MIN_CONTEXT_PATCHES, MAX_CONTEXT_PATCHES]``, so
    the histogram compares directly with ``SAMPLER_REFERENCE``. Every masked patch counts: this is the
    sampler, not a protocol.
    """
    rng = np.random.default_rng(seed)
    hist = DHistogram(f"training sampler, {n_draws} draws, seed {seed}")
    for _ in range(int(n_draws)):
        n_ctx = int(rng.integers(MIN_CONTEXT_PATCHES, MAX_CONTEXT_PATCHES + 1))
        seq_len = n_ctx + PREDICTION_PATCHES
        spans = sample_mask_spans(seq_len, rng)
        _idx, valid, d, _anchor = _mask_slots(spans, seq_len)
        hist.add(d[valid])
    return hist


# Cohort census (§3.24) and context note (§3.25). Eligibility is PINNED to the 24 h footprint and does NOT
# follow MAX_CONTEXT_PATCHES: ``calibrate.py`` drops a short segment with a bare ``continue``, and segments
# are cut at every CGM gap over 30 min, so the survivors are the longest gap-free wears and a wider context
# would silently re-select the cohort.
ELIGIBILITY_HOURS = 24.0
ELIGIBILITY_CONTEXT_PATCHES = int(round(ELIGIBILITY_HOURS * _PATCHES_PER_HOUR))
ELIGIBILITY_STEPS = (ELIGIBILITY_CONTEXT_PATCHES + PREDICTION_PATCHES) * PATCH_SIZE


def usable_steps(n_steps: int) -> int:
    """Whole patches of a segment, in steps — the length the window loops use."""
    return (int(n_steps) // PATCH_SIZE) * PATCH_SIZE


def _window_count(n: int, footprint: int, stride: int) -> int:
    return 0 if n < footprint else 1 + (n - footprint) // stride


@dataclass(frozen=True)
class CohortCensus:
    """Kept and dropped segment / window counts for one cohort at one ``n_ctx``."""
    cohort: str
    n_ctx: int
    stride_patches: int
    n_segments: int
    kept_segments: int
    ineligible_segments: int          # shorter than the PINNED 24 h footprint
    lost_to_context_segments: int     # eligible at the pin, unservable at n_ctx
    pinned_windows: int               # windows of the FIXED set
    kept_windows: int                 # windows this arm can actually serve
    lost_to_context_windows: int

    @property
    def dropped_segments(self) -> int:
        return self.ineligible_segments + self.lost_to_context_segments

    @property
    def dropped_windows(self) -> int:
        return self.pinned_windows - self.kept_windows

    def format(self) -> str:
        hrs = self.n_ctx / _PATCHES_PER_HOUR
        return (
            f"  [{self.cohort}] n_ctx={self.n_ctx} patches ({hrs:.1f} h), "
            f"stride {self.stride_patches} patches\n"
            f"    segments kept {self.kept_segments} / {self.n_segments}  "
            f"dropped {self.dropped_segments} "
            f"({self.ineligible_segments} shorter than the pinned "
            f"{ELIGIBILITY_HOURS:.0f} h footprint, "
            f"{self.lost_to_context_segments} lost to n_ctx)\n"
            f"    windows  kept {self.kept_windows} / {self.pinned_windows} "
            f"pinned  dropped {self.dropped_windows} "
            f"({self.lost_to_context_windows} lost to n_ctx)"
        )


def census_segments(cohort: str, segment_steps: Iterable[int], n_ctx: int,
                    stride_patches: int = 8) -> CohortCensus:
    """Kept and dropped segments and windows for one cohort.

    ``segment_steps`` are per-segment lengths in steps, floored to whole patches as the window loops floor
    them; ``n_ctx`` is the width this evaluation RUNS at, ``stride_patches`` the gap between window starts.
    Eligibility is the PINNED 24 h footprint, so the window set is the same at every ``n_ctx``; what an
    ``n_ctx`` past the pin cannot serve is reported as lost to ``n_ctx``, never dropped silently.
    """
    stride = int(stride_patches) * PATCH_SIZE
    run_footprint = (int(n_ctx) + PREDICTION_PATCHES) * PATCH_SIZE
    n_seg = kept = ineligible = lost_seg = 0
    pinned_w = kept_w = 0
    for raw in segment_steps:
        n_seg += 1
        n = usable_steps(raw)
        if n < ELIGIBILITY_STEPS:
            ineligible += 1
            continue
        pinned_w += _window_count(n, ELIGIBILITY_STEPS, stride)
        if n < run_footprint:
            lost_seg += 1
            continue
        kept += 1
        kept_w += _window_count(n, max(run_footprint, ELIGIBILITY_STEPS), stride)
    return CohortCensus(
        cohort=cohort, n_ctx=int(n_ctx), stride_patches=int(stride_patches),
        n_segments=n_seg, kept_segments=kept, ineligible_segments=ineligible,
        lost_to_context_segments=lost_seg, pinned_windows=pinned_w,
        kept_windows=kept_w, lost_to_context_windows=pinned_w - kept_w)


def context_note(n_ctx: int) -> str:
    """One line naming the evaluated context width against the trained mixture.

    Headline numbers are measured at ``MAX_CONTEXT_PATCHES``, one of the widths drawn uniformly over
    ``[MIN_CONTEXT_PATCHES, MAX_CONTEXT_PATCHES]``: the mean trained context is the mixture's, not this one.
    """
    widths = MAX_CONTEXT_PATCHES - MIN_CONTEXT_PATCHES + 1
    mean_p = (MAX_CONTEXT_PATCHES + MIN_CONTEXT_PATCHES) / 2.0
    return (
        f"  n_ctx = {int(n_ctx)} patches ({int(n_ctx) / _PATCHES_PER_HOUR:.1f} h); "
        f"training draws n_ctx ~ U{{{MIN_CONTEXT_PATCHES}..{MAX_CONTEXT_PATCHES}}}, "
        f"mean {mean_p:.1f} patches ({mean_p / _PATCHES_PER_HOUR:.1f} h) — this "
        f"width is 1 of {widths} sampled widths"
    )


@dataclass
class RunReport:
    """Everything an evaluation must print: n_ctx, the d histogram, the census.

    Reported for EVERY run.  The ``d`` histogram is per protocol and never
    pooled across them; the census is per cohort.
    """
    label: str
    n_ctx: int
    _hists: dict[str, DHistogram] = field(default_factory=dict)
    _census: list[CohortCensus] = field(default_factory=list)

    def observe(self, masked_set: MaskedSet) -> MaskedSet:
        """Record one window's scored ``d`` values under its protocol."""
        name = masked_set.protocol.name
        if name not in self._hists:
            self._hists[name] = DHistogram(f"{self.label} · {name} protocol")
        self._hists[name].add_masked_set(masked_set)
        return masked_set

    def census(self, cohort: str, segment_steps: Iterable[int],
               stride_patches: int = 8) -> CohortCensus:
        c = census_segments(cohort, segment_steps, self.n_ctx, stride_patches)
        self._census.append(c)
        return c

    def format(self) -> str:
        lines = [f"=== PROTOCOL REPORT · {self.label} ===", context_note(self.n_ctx)]
        for c in self._census:
            lines.append(c.format())
        for name in sorted(self._hists):
            lines.append(self._hists[name].format())
        if not self._hists:
            lines.append("  (no masked sets observed)")
        else:
            # A protocol's histogram is FIXED by the protocol and must NOT match SAMPLER_REFERENCE; only
            # ``sampler_d_histogram`` is comparable to it.
            lines.append("  (protocol histograms are fixed by the protocol and are "
                         "NOT comparable to SAMPLER_REFERENCE; the training "
                         "sampler's own audit is protocols.sampler_d_histogram)")
        return "\n".join(lines)

    def emit(self) -> None:
        print(self.format(), flush=True)


class InfillScores:
    """The point-error side of the ``infill_*`` columns, per ``d``; interpolation is the only baseline.

    Squared and absolute errors accumulate per ``d`` for the model and the baseline, so every emitted column
    names its ``d`` and no pooled scalar is reachable from here.
    The FAN side belongs to ``metrics.scoring``, the single definition of CRPS, Winkler, coverage-with-
    sharpness and joint coverage; this class only collects the fan, truth and ``d`` for it. The alarm curve is
    a FORECAST figure — infill has no alarm decision time, so ``forecast_lead_minutes`` does not apply.
    """

    def __init__(self):
        self._n: Counter = Counter()
        self._se: Counter = Counter()
        self._ae: Counter = Counter()
        self._base_se: Counter = Counter()
        self._base_ae: Counter = Counter()
        self._q: list[np.ndarray] = []
        self._true: list[np.ndarray] = []
        self._d: list[np.ndarray] = []
        self._group: list[np.ndarray] = []
        self._n_groups = 0

    def add(self, d_patches: np.ndarray, pred: np.ndarray, true: np.ndarray,
            baseline: np.ndarray, bands: np.ndarray | None = None) -> None:
        """Record one window's scored patches.

        ``d_patches`` ``(P,)`` in row order; ``pred`` / ``true`` / ``baseline`` ``(P, PATCH_SIZE)`` mg/dL;
        ``bands`` optional ``(P, PATCH_SIZE, N_QUANTILES)`` decoded fan, mg/dL.
        """
        d_patches = np.asarray(d_patches, dtype=np.int64)
        pred = np.asarray(pred, dtype=np.float64).reshape(len(d_patches), PATCH_SIZE)
        true = np.asarray(true, dtype=np.float64).reshape(len(d_patches), PATCH_SIZE)
        base = np.asarray(baseline, dtype=np.float64).reshape(len(d_patches), PATCH_SIZE)
        err, berr = pred - true, base - true
        for d in np.unique(d_patches):
            m = d_patches == d
            d = int(d)
            self._n[d] += int(m.sum()) * PATCH_SIZE
            self._se[d] += float(np.sum(err[m] ** 2))
            self._ae[d] += float(np.sum(np.abs(err[m])))
            self._base_se[d] += float(np.sum(berr[m] ** 2))
            self._base_ae[d] += float(np.sum(np.abs(berr[m])))
        if bands is not None:
            self._q.append(np.asarray(bands, dtype=np.float64))
            self._true.append(true)
            self._d.append(d_patches)
            self._group.append(np.full(len(d_patches), self._n_groups, dtype=np.int64))
        self._n_groups += 1

    def fan(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """``(q, true, d, group)`` over every scored patch, in ``scoring.score_fan``'s shapes.

        ``q`` ``(N, PATCH_SIZE, N_QUANTILES)`` mg/dL, ``true`` ``(N, PATCH_SIZE)`` mg/dL, ``d`` ``(N,)``,
        ``group`` ``(N,)`` the window index. Empty arrays when no fan was collected.
        """
        if not self._q:
            return (np.zeros((0, PATCH_SIZE, len(QUANTILE_LEVELS))),
                    np.zeros((0, PATCH_SIZE)), np.zeros(0, dtype=np.int64),
                    np.zeros(0, dtype=np.int64))
        return (np.concatenate(self._q), np.concatenate(self._true),
                np.concatenate(self._d), np.concatenate(self._group))

    def columns(self) -> dict[str, float]:
        """The ``infill_*`` point-error columns, every one named with its ``d``."""
        out: dict[str, float] = {}
        for d in sorted(self._n):
            n = self._n[d]
            if n == 0:
                continue
            rmse = math.sqrt(self._se[d] / n)
            base_rmse = math.sqrt(self._base_se[d] / n)
            out[column(INFILL, 'n', d)] = float(n)
            out[column(INFILL, 'rmse', d)] = rmse
            out[column(INFILL, 'mae', d)] = self._ae[d] / n
            out[column(INFILL, 'interp_rmse', d)] = base_rmse
            out[column(INFILL, 'interp_mae', d)] = self._base_ae[d] / n
            out[column(INFILL, 'skill_interp', d)] = (
                1.0 - rmse / base_rmse if base_rmse > 0 else float('nan'))
        return out

    def format(self) -> str:
        cols = self.columns()
        if not cols:
            return "  infill: no scored patches"
        lines = ["  infill_* columns (scored against LINEAR INTERPOLATION only; "
                 "fan figures come from metrics.scoring over .fan())"]
        for k in sorted(cols):
            lines.append(f"    {k:<28} {cols[k]:12.4f}")
        return "\n".join(lines)


def score_infill_trajectory(model, stats, feats: np.ndarray, cgm: np.ndarray,
                            scores: InfillScores, rng: np.random.Generator,
                            n_ctx: int = MAX_CONTEXT_PATCHES,
                            stride_patches: int = 8,
                            max_windows: int | None = None,
                            announce: tuple[int, ...] = (0, 1, 2),
                            report: RunReport | None = None,
                            device=None) -> int:
    """Run the INFILL protocol over one trajectory into ``scores`` -> the number of windows scored.

    Each window draws its own interior spans, forwards once, and scores ONLY those: the mandatory trailing
    span rides unscored, since a one-sided row does not belong in the infill namespace. Truth is observed CGM
    the model was not shown; the baseline is linear interpolation, never persistence.
    """
    from inference import predict                          # local: keeps torch off import
    from metrics.core.features import context_window
    from metrics.core.calibrate import _future_overrides
    from config import CHANNEL_TO_FEAT

    # A set short of CHANNEL_TO_FEAT leaves the dropped slot at normalize(0), a legal "no event", so the
    # protocol would score a regime training never saw.
    assert tuple(announce) == tuple(CHANNEL_TO_FEAT), (
        f"announced set {tuple(announce)} != announceable set "
        f"{tuple(CHANNEL_TO_FEAT)}")

    cgm = np.asarray(cgm, dtype=np.float64)
    footprint = (int(n_ctx) + PREDICTION_PATCHES) * PATCH_SIZE
    n = usable_steps(len(cgm))
    stride = int(stride_patches) * PATCH_SIZE
    scored = 0
    for pred_start in range(int(n_ctx) * PATCH_SIZE, n - SPAN_STEPS + 1, stride):
        if max_windows is not None and scored >= max_windows:
            break
        window_start = pred_start - int(n_ctx) * PATCH_SIZE
        if window_start + footprint > n:
            break
        ms = infill_masked_set(int(n_ctx), rng)
        ctx = context_window(feats, pred_start, int(n_ctx))
        out = predict(model, ctx, normalization_stats=stats, device=device,
                      overrides=_future_overrides(feats, pred_start, announce),
                      mask_spans=ms.spans)
        rows = ms.scored_rows()
        pred = out['median_bg'].detach().cpu().numpy().reshape(-1, PATCH_SIZE)[rows]
        true = cgm[window_start + ms.scored_steps()]
        base = baseline_for(ms, cgm, window_start)
        bands = (out['bands'].detach().cpu().numpy()[rows]        # (P, S, K) mg/dL
                 if 'bands' in out else None)
        scores.add(ms.scored_d(), pred, true, base, bands)
        if report is not None:
            report.observe(ms)
        scored += 1
    return scored


def main() -> None:
    """Self-report: the sampler audit and both protocols' realised d histograms."""
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--draws', type=int, default=20000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n-ctx', type=int, default=MAX_CONTEXT_PATCHES)
    args = ap.parse_args()

    print("=== SAMPLER AUDIT (training placement, uniform) ===")
    hist = sampler_d_histogram(args.draws, args.seed)
    print(hist.format())
    print(hist.compare_to_sampler_reference())

    print("\n=== FIXED PROTOCOLS ===")
    print(f"  forecast: reachable d {reachable_d(FORECAST)}, baseline "
          f"{FORECAST.baseline}, columns = the existing names")
    print(f"  infill:   reachable d {reachable_d(INFILL)}, baseline "
          f"{INFILL.baseline}, columns = {INFILL.prefix}* per d "
          f"(interior budget {INFILL_BUDGET_PATCHES} patches)")

    rep = RunReport(label=f"protocols self-report (n_ctx={args.n_ctx})",
                    n_ctx=args.n_ctx)
    rng = np.random.default_rng(args.seed)
    for _ in range(args.draws):
        rep.observe(forecast_masked_set(args.n_ctx))
        rep.observe(infill_masked_set(args.n_ctx, rng))
    print()
    rep.emit()
    print("\n  thresholds:", "UNSET" if not THRESHOLDS else THRESHOLDS)


if __name__ == '__main__':
    main()
