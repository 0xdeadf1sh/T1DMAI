# T1DMAI — Agent Instructions

## Quick start

```bash
pip install -r requirements.txt         # torch, numpy, blosc2, matplotlib, pygame, pytest
python normalization.py                 # must run once before training (or --from-cache DIR)
python train.py                         # train
python -m pytest tests/ -v -s           # test (-s required: DUMP lines catch silent numerical bugs)
```

`T1DMSIM` is a symlink to the sibling checkout and `import config` needs it — config reads
`SIMULATOR_WARMUP_HOURS` and the two exercise constants from `T1DMSIM.simulator` rather than
restating them.

**Config is a single plain file.** `config.py` is a plain file, not a symlink, and there is no
JSON config tier. Every training value lives there — `BATCH_SIZE = 512`, `NUM_WORKERS = 8`, the
Muon/AdamW LRs, `LR_MIN_RATIO`, `PATIENT_UNIFORM_SAMPLE_PROB = 0.0`, `PREDICTION_HORIZON_HOURS`,
and the mask-sampler constants. Edit `config.py` directly, or `resize_model.py` for the
architecture knobs.

The Kovatchev risk transform bakes the clinical hypo>hyper asymmetry into the loss geometry itself
(equal risk-distance is a larger danger at low BG), so there is no focal or composite hypo-weighting
term. Checkpoint selection is `val_loss_total` alone; the counterfactual probe is diagnostic and
never feeds the loss or selection.

## Three spaces and the only two bridges

The whole pipeline lives in exactly three spaces, crossed by exactly two bridge pairs. Mixing them is the highest-severity bug class — track which space every tensor is in.

| space | sole-legal representation |
|---|---|
| (a) normalized z-space | model **inputs** — the four `CHANNEL_NAMES` signals via `normalization_stats`; the bg input is `z(f(bg))`, Kovatchev risk THEN z-score. Input feat 4 (`bg_masked`) is a bit and belongs to no space: it carries no statistics. |
| (b) mg/dL physical | `anchor_bg`, the true-BG target, **all** clinical metrics, every array crossing `metrics/scoring.py`'s and `mondrian.py`'s boundary, the GUI |
| (c) risk space | model bg **input** (`f(bg)`, before the z-score) and head **outputs** (quantiles, median), the loss terms, the f-transformed target |

- `normalize`/`denormalize` (in `normalization.py`) is the **only** (a)↔(b) crossing. For the bg channel that "physical" side is risk space: `normalize` applies `kovatchev_f_np` to bg BEFORE the z-score (via `RISK_SPACE_CHANNELS`, which always contains `bg_absolute`), and `denormalize` inverts z THEN `f_inv`. Inputs AND outputs are therefore risk-space — the model never sees raw-mg/dL bg.
- `utils.kovatchev_f` / `utils.kovatchev_f_inv` is the **only** (b)↔(c) crossing. (Kovatchev risk transform, anchored to the `[40, 400]` mg/dL device range: `_KOVATCHEV_SCALE`, `_KOVATCHEV_POWER`, `_KOVATCHEV_OFFSET` at the top of `utils.py`, solved so `f(40) = −√10` and `f(400) = +√10`, i.e. the risk `10·f²` saturates at 100 at both rails; verify the constants against a reference in a unit test.)
- `f` on the bg INPUT happens inside `normalize`/`_forward_transform` (once per channel gather); `f` on the TARGET happens exactly once (top of `risk_total_loss`); `f(anchor_bg)` inside `assemble_quantiles` is a separate legitimate call on a different tensor. No op crosses any other way.
- **`f`-guard split.** The `[40, 400]` Kovatchev anchors and the physical clamp `[BG_CLAMP_MIN, BG_CLAMP_MAX] = [10, 400]` are separate quantities. The clamp is wider below the anchors, so the realised risk range `[f(10), f(400)] = [−6.8198, +3.1623]` is asymmetric — that is the anchoring working, not a defect; do not "correct" it by moving an anchor. `kovatchev_f(g)` carries a **hard** `assert (g >= BG_CLAMP_MIN - 1e-3).all()` — the UNITS TRIPWIRE; reserve it for controlled callers that must never carry z-space (`f(anchor_bg)`, re-f of an inverted value), so a stray z-scored value trips it loudly: every legal `z` satisfies `z < BG_CLAMP_MIN - 1e-3` by a wide margin, whichever pool fit the stats — the guarantee is that bound, not any particular numeric z-range. `kovatchev_f` itself does **not** clamp — it asserts the floor and warns above the ceiling. The physical `g.clamp(BG_CLAMP_MIN, BG_CLAMP_MAX)` before `f` lives in `kovatchev_f_target(g)` (TARGET path only, warn if it clamps beyond a small tolerance — a rare backstop, since the target is the raw post-noise `bg_observed`, provably cache mg/dL) and in `kovatchev_f_np(g)`, the NumPy INPUT-path sibling used by the bg-input transform inside `normalize` and by the stat fit. `kovatchev_f_inv(r)` scrubs non-finite input, clamps the **risk input** to `[f(BG_CLAMP_MIN), f(BG_CLAMP_MAX)]`, forms the base, `exp`s, then clamps the output to `[BG_CLAMP_MIN, BG_CLAMP_MAX]` — killing both the negative-base NaN and the fp32 exp overflow. `kovatchev_f_inv_np` mirrors it exactly.
- `f` is **never** differentiated: targets and `anchor_bg` are constants; `anchor_bg` is `.detach()`'d before `f` inside `model.forward`. `utils` imports `BG_CLAMP_MIN/MAX` from `T1DMSIM.simulator` — never restate them.
- `metrics/scoring.py` re-runs the same tripwire (`_assert_mgdl`) at every public entry point, so a risk-space fan handed to a scoring rule trips loudly instead of producing a plausible number.
- **No input smoother — raw post-noise signals (a-space precondition).** There is no causal smoother anywhere in the pipeline, and scipy is not a dependency. Every signal channel (bg, carb, insulin, exercise) enters normalization as the RAW simulator-final post-noise series: bg is only clamped to the physical `[BG_CLAMP_MIN, BG_CLAMP_MAX]` (via `kovatchev_f_np`'s internal clamp — the clamp, not the `[40, 400]` anchors), and the sparse channels are floored at 0 by the log1p `max(x, 0)`. The SAME raw bg is the model input, the forecast target, AND the anchor — there is no input/target asymmetry. Deployment realism is intrinsic: the live CGM stream is consumed as-is, and train↔inference share one distribution since the autoregressive roll re-feeds the model's own raw output.

## The masked-BG objective

**There is no prediction zone.** A window is `T` patches, each VISIBLE or MASKED. A masked patch
withholds bg (feat 0); carb, insulin and exercise keep their true (training) or announced
(inference) values there. The head emits a quantile fan for every masked patch. A masked span
ending at patch `T-1` is a FORECAST, one starting at patch 0 a BACKCAST, anything else INFILL —
three cases of one objective, not three modes. `PREDICTION_PATCHES` is the span of the fixed
forecast *protocol* and the reference length the per-span median basis scales against; it is not a
region of a training sample.

- **The sampler** (`data.sample_mask_spans`), per sample: `n_spans ~ U{1..MASK_MAX_SPANS}` (3);
  each `L_i ~ U(MASK_SPAN_LENGTHS)` independently; `sum(L) > MAX_MASKED_PATCHES` resamples the
  **whole length vector**, never one element (per-element redrawing yields a different length
  distribution and so a different `d` histogram); placement is stars-and-bars over the `n_spans + 1`
  gaps, **except** that with probability `MASK_RIGHT_EDGE_QUOTA` (0.35) the LAST span is pinned
  flush against patch `T-1` and the rest composed over the prefix, which holds the same slack. That
  is the one departure from uniform placement: `n_spans` and the length law are drawn identically in
  both branches, so only the `d` histogram moves. There is no curriculum, no annealing and no
  rejection on placement. At quota 0 the branch draw is short-circuited, so the rng stream is the
  pre-quota one exactly. `d_balance.d_distribution` enumerates both branches, and
  `metrics/protocols.py`'s `SAMPLER_REFERENCE` is produced from it.
- **Two masked spans never abut.** One mandatory visible separator is charged up front. The
  separator is what makes the anchor, the per-span median basis and the DILATE length bucket well
  defined per span; two spans with nothing between them are one longer span, and `utils._span_layout`
  identifies spans by adjacency in `mask_idx` precisely because the sampler guarantees this.
- **Head slots.** `MAX_MASKED_PATCHES` is two things: the sampler's cap on `sum(L)`, and `M`, the
  head's fixed slot count. The head always emits `M` slots, so a sample with fewer masked patches
  pads the surplus; a padded slot gathers patch 0, which makes its output a real number against a
  real target. **Every loss and metric path MUST discard padded slots by the `(B, M)` `valid`
  flag**, or it trains against patch 0's BG behind a plausible neighbouring anchor.
- **The anchor is per slot, ONE-SIDED and LEFT-PREFERRING** (`data._anchor_step_for_span`): the last
  step of the left neighbour patch, or the first step of the right neighbour when the span starts at
  patch 0 (the only no-left-neighbour case there is). Every slot of a contiguous span shares one
  value. It reaches the model as `(B, M)` mg/dL — not one value broadcast across the window.
- **`d` is the distance in patches to the nearest visible evidence on EITHER side**
  (`data._mask_slots`), and it is the only axis a masked-BG metric bins on — never span length,
  which confounds one-sided and two-sided cases at equal difficulty. `d` and the
  anchor disagree by construction: the anchor ignores the near side, so a two-sided span's last slot
  can sit at `d = 1` while anchoring `L` patches to the left. `_mask_slots` states the exact
  fraction; do not re-derive it.
- **Nothing pooled over `d` is a selection metric.** The sampler concentrates supervision at small
  `d`, so a pooled masked-BG scalar averages over a mask distribution rather than a difficulty: it
  improves when the mixture softens and moves between protocols that share no mixture.
  `metrics/scoring.py` stamps every pooled figure with `POOLED_NOT_COMPARABLE`; `metrics/protocols.py`
  carries the enumerated percentages. Compare pooled against pooled only within one fixed protocol.

## Architecture gotchas

- **Capacity.** `D_MODEL = 32`, `N_LAYERS = 2`, `N_HEADS = 2`, `FFN_DIM = 4×D_MODEL`,
  `BG_HEAD_HIDDEN = 1×D_MODEL` — 38,241 parameters plus one 18-element `step_basis` buffer.
  `ARCH_VERSION = 'risk-v4'`. Don't bake those numbers into other code or comments: `resize_model.py`
  rewrites them, preserving `HEAD_DIM = D_MODEL // N_HEADS` and the symbolic `FFN_DIM = k·D_MODEL` /
  `BG_HEAD_HIDDEN = k·D_MODEL` relations.
- **Patch = 6 timesteps = 30 min.** Patches are the attention/loss unit; the BG head emits
  `PATCH_SIZE` timesteps per slot.
- **Combined insulin = basal + bolus.** The simulator provides them separately; always sum.
- **5 input features — the FROZEN index map.** `[bg_absolute, carb_intake, insulin_combined,
  exercise_equiv, bg_masked]`, `N_INPUT_FEATURES = 5`, `PATCH_DIM = PATCH_SIZE × N_INPUT_FEATURES =
  30`, **step-major**, so a feature's columns are the stride `[:, f::N_INPUT_FEATURES]`. There are no
  temporal sin/cos features and no `TEMPORAL_FEAT_START`.
  - **feat 0** `bg_absolute` is the SOLE risk-space input: `z(f(bg))`, handled entirely inside
    `normalize`/`_forward_transform` via `RISK_SPACE_CHANNELS`. carb/insulin/exercise keep log1p+z.
  - **feat 3** `exercise_equiv` is T1DMSIM's total exercise as a carbohydrate-EQUIVALENT
    glucose-disposal curve in **g/step**, fed at that scale — never Kovatchev-transformed, never
    rescaled to an intensity.
  - **feat 4** `bg_masked` is a per-PATCH **bit** written into all `PATCH_SIZE` of that patch's
    step-major columns. It carries no normalization statistics — no mean, no std, no log1p — so
    `CHANNEL_NAMES` has FOUR entries while `N_INPUT_FEATURES` is 5. `data.BG_MASKED_FEAT =
    len(CHANNEL_NAMES)` derives the index; never assert the two counts equal. The masked set is
    ANNOUNCED rather than inferred because masking is not positional and `z = 0` in a withheld bg
    slot decodes to an ordinary reading (~142 mg/dL on the balanced pool), not a sentinel. The bit
    must never sit outside the step-major block — `PATCH_DIM` and the stride idiom both depend on it.
  - `NON_MASKABLE_FEATS = (0,)`; `MASKABLE_FEATS = (1, 2, 3)`; feat 4 is in neither, being written
    from the sampled mask rather than from a signal. `CHANNEL_TO_FEAT = {0: 1, 1: 2, 2: 3}` is the
    single output-channel → input-feat mapping, consumed by BOTH `data.py`'s announced-dose write and
    `inference.py`'s override-write — never two independent `+offset` literals.
  - Feats 1–3 are **plan** channels: nothing but what the patient announced is ever written into
    them, masked patches included. There is no conditioned/unconditioned dichotomy and no reveal.
- **Forward signature (FROZEN): `forward(patches, attn_mask, anchor_bg, mask_idx, return_time=False)
  -> (q_tau, median)`.**
  - `patches` `(B, T, PATCH_DIM)`; `T <= MAX_SEQ_LEN` and is never asserted equal to it — the collate
    left-pads to the BATCH maximum, so `T` varies batch to batch.
  - `attn_mask` `(T, T)` or `(B, T, T)` **bool**, True = attend.
  - `anchor_bg` `(B, M)` mg/dL; the units tripwire reads **all** `M` slots, padded ones included, so
    a padded slot must still carry a legal mg/dL value.
  - `mask_idx` `(B, M)` **int64** — the patch index each head slot reads; padded slots gather 0.
  - `q_tau` `(B, M, PATCH_SIZE, N_QUANTILES)` risk space, ascending τ; `median` `(B, M, PATCH_SIZE)`
    risk space, `== q_tau[..., QUANTILE_LEVELS.index(0.5)]`. `return_time=True` appends `time_pred`
    `(B, M, TIME_PROBE_N_BINS)`, or `None` when `TIME_PROBE_ENABLED` is False; `q_tau` and `median`
    are computed identically either way. There is no `return_trend` / `return_alarm` kwarg.
- **The head reads its `M` slots by GATHER, not by a trailing slice.** `x.gather(1, mask_idx…)` — the
  masked set is arbitrary, so a slice is correct only for a forecast. Read the output the same way:
  slot `j` is patch `mask_idx[:, j]`, and any per-slot target (hour of day, true BG) must follow
  `mask_idx`, not a fixed offset from the context end.
- **The attention mask is bool and reaches SDPA as one.** `utils.create_attention_mask_from_visible(
  visible, is_pad)` is the general form, built in four load-bearing lines: visible→visible allowed
  (bidirectional among evidence), visible row→masked col **blocked** (evidence never reads a
  prediction), masked row→any real col allowed, pad rows and columns blocked except the diagonal (an
  all-False row is a softmax NaN). `create_attention_mask(n_context, n_prediction)` is the
  right-edge shim over it. Nothing is memoized — no cheap key identifies a masked set, and a memo on
  `(n_context, n_prediction)` hands one sample's mask to another with no shape error. `model.forward`
  gives a per-sample `(B, T, T)` mask its head axis with `unsqueeze(1)` → `(B, 1, T, T)`; passing it
  straight aligns `B` onto the head axis. No additive float mask is materialized on the eager path
  and nothing is built per layer. Print a small example before trusting it.
- **Position is RoPE alone.** `ROPE_BASE = 1000`; the cos/sin tables are built once per forward at
  the model level and shared across layers. QK-norm is KEPT — per-head RMSNorm on Q and K
  (`q_norm`/`k_norm`), applied BEFORE RoPE so the normalized norms are not undone by the rotation.
  There is no additive per-head distance bias on the logits.
- **Context window is variable 24–48 h** (`MIN_CONTEXT_PATCHES = 48`, `MAX_CONTEXT_PATCHES = 96` at 2 patches/hour, since `PATCH_SIZE = 6` makes one patch 30 min). Each training sample draws `n_ctx` uniformly in `[MIN, MAX]`; `collate_fn` left-pads to the batch maximum. The simulator's own ACF analysis in `T1DMSIM/diff/README.md` §0.5 measures pooled-CGM ACF₀.₂ at 5.3 h for the simulator and 2.4–4.8 h across the three real cohorts, so the `MIN` sits well above the autocorrelation rather than being derived from it. The 24 h floor spans one full basal dose cycle — both analogues are injected on a fixed 24 h interval with no jitter (they differ in *action* duration, glargine 26 h / degludec 42 h, not in dosing) — and preserves GT context through the 8 h night long-horizon rolling validation (`inference.predict_rolling` slides the window once it exceeds `MAX_CONTEXT_PATCHES`). At inference time the model accepts any `n_ctx ≥ MIN_CONTEXT_PATCHES`.
- **No patient embedding.** The model is patient-agnostic — no learned patient identity vector and no `patient_seeds` argument to `model.forward()`. Patient identity is implicit in the context window. `compute_patient_seed` exists only as a deterministic key for picking simulator runs in `data.py`.
- **Unified day + night training.** One model covers both. A window's masked spans may sit at any patch-aligned position in the trajectory — there is no day/night band restriction and no time-of-day input feature; day vs night dynamics are learned from the glucose/carb/insulin/exercise trajectory shape alone. `PREDICTION_PATCHES = PREDICTION_HORIZON_HOURS × _PATCHES_PER_HOUR` is derived in `config.py`; change the hours, not the patch count.
- **No dynamics output channels.** There is no `N_OUTPUT_CHANNELS`, no per-channel head, no MDN, no IS/HGO/carb/insulin output, no `bg_delta` anywhere. The model emits a single BG quantile head.
- **Single BG quantile head in Kovatchev risk space.** `model.bg_head` is a 3-layer MLP
  (`Linear(D_MODEL, BG_HEAD_HIDDEN), SiLU, Linear, SiLU, Linear(BG_HEAD_HIDDEN,
  BG_HEAD_STEP_BASIS_DIM·(1 + 2·N_SPREADS))`, `N_SPREADS = 3`) run on each gathered masked-slot
  hidden state (post `final_norm`).
  - **Smooth-basis step expansion (structural anti-oscillation).** The head emits
    `K = BG_HEAD_STEP_BASIS_DIM` (3) coefficients per (slot, channel) — not `PATCH_SIZE` independent
    per-step values — expanded across the within-patch timesteps by the fixed orthonormal
    `step_basis (S, K)` buffer (`BG_HEAD_STEP_BASIS_TYPE = 'dct'`, or `'poly'`) via
    `einsum('sk,bmkc->bmsc', step_basis, coeff)`. With `K < PATCH_SIZE` the period-2 within-patch
    mode is unrepresentable, so an intra-patch median zigzag cannot be emitted by construction;
    `K = PATCH_SIZE` recovers a fully-free per-step head.
  - **Assembly.** `head_raw (B, M, S, 1 + 2·N_SPREADS)` goes to
    `utils.assemble_quantiles(head_raw, anchor_bg, mask_idx, valid, carry_spread)`: col 0 = median
    delta; cols 1..3 = the `τ>.5` spreads (nearest→far .75/.9/.95); cols 4..6 = the `τ<.5` spreads
    (.25/.1/.05). `anchor = f(anchor_bg).detach()` **per slot**; `spread = softplus(raw) +
    BG_QUANTILE_SPREAD_MIN` (prevents σ-collapse); `q(τ>.5) = m + carry_spread + cumsum(d+)`,
    `q(τ<.5) = m − carry_spread − cumsum(d−)`; the fan is ascending by construction.
    `BG_HEAD_INIT_SCALE = 1e-2` with zero bias ⇒ small coeffs ⇒ `median ≈ f(anchor_bg)` at init (the
    initial forecast is flat from each slot's own anchor). `QUANTILE_LEVELS = (.05,.1,.25,.5,.75,.9,.95)`,
    `N_QUANTILES = 7`. Output is RISK space; inference owns `f_inv → mg/dL`.
  - **`carry_spread` is DEAD at runtime** (risk space, default 0.0). It seeds the cumulative-spread
    base on both sides of the median, and nothing outside `tests/test_utils.py` passes it: the sole
    non-test call site is `model.forward`, `assemble_quantiles(head_raw, anchor_bg.detach(),
    mask_idx)`, which takes the default for `valid` and `carry_spread` alike. The rolling widening
    that needs this algebra lives in `inference.predict_rolling` (see *Rolling prediction BG fill*),
    which cannot reach the argument at all — the assembly runs inside `model.forward`, so by the
    time a caller holds a fan the fan is already built. The two pieces of algebra are duplicated of
    necessity: change the `± carry_spread` here and the rolling shift silently stops matching it.
  - `assemble_quantiles` is the SINGLE chokepoint for the **median and the native fan**: training,
    `inference.predict` and `predict_rolling` all reach it through `model.forward`, so a median-mode
    change propagates to all three identically. It is not the chokepoint for the rolling band carry,
    which is applied after the fact on the returned `q_tau`; and the export cuts the graph upstream
    of it, at `head_raw`.
- **The median basis is PER SPAN, and a fixed `G` is a defect.** `BG_HEAD_MEDIAN_MODE = 'global'`
  projects each span's median delta onto a low-frequency DCT-II subspace spanning that span's
  `n = L·PATCH_SIZE` steps, with
  `G_L = max(1, ceil(BG_HEAD_MEDIAN_GLOBAL_DIM · L / PREDICTION_PATCHES))` (`utils.global_median_dim`),
  clamped to `min(G_L, n)`. A projection is an L2 **contraction**, so the per-patch offset is bounded
  and non-monotone in `p` and cannot drift or amplify. A FIXED `G` is not an approximation: at
  `L = 1` the projection would have as many columns as the span has steps — the identity — so the
  anti-drift contraction is ABSENT rather than weakened, and every fan assert still passes. What
  `G_L` holds roughly constant is the fraction of the span the basis can bend; the cutoff period
  `2n/G_L` varies with `L` and is what to report. `utils._span_layout` is what groups the `M` slots
  into spans (adjacency in `mask_idx`, `valid` respected); passing `mask_idx = None` low-passes the
  whole `M` axis as one span. The two alternative modes are `'cumulative'` (each patch continues
  from the previous patch's endpoint, C0 at every seam) and `'independent'` (flat
  `m = anchor + delta`).
- **No channel cross-attention.** Each transformer block is `temporal_attn(norm1) → FFN(norm2)` — **2 residual writes**, residual init rescale `base_std / sqrt(2·N_LAYERS)`.
- **fp32 everywhere.** No bf16 autocast anywhere — forward and loss are both fp32. RMSNorm and SwiGLU run native fp32. No gradient checkpointing — every block runs its forward once and keeps activations.

## Normalization pipeline (highest bug surface)

- **4 normalized channels.** `CHANNEL_NAMES = [bg_absolute, carb_intake, insulin_combined, exercise_equiv]`, and `N_CHANNELS == 4` is asserted at import. It is deliberately NOT tied to `config.N_INPUT_FEATURES = 5`: feat 4 is a mask bit, not a measured signal, and nothing in this module normalizes or denormalizes it.
- **Sparse channels** — `SPARSE_LOG1P_CHANNELS = {carb_intake, insulin_combined, exercise_equiv}` — use log1p: `normalized = (log1p(max(x,0)) - mean) / std`. Denormalize: `expm1(normalized * std + mean)` clamped ≥ 0. `exercise_equiv` takes carb's encoding exactly: it is g/step, so the trained scale is g/step.
- **BG channel** (`bg_absolute`) is in **Kovatchev risk space**: `normalized = (kovatchev_f_np(bg) - mean) / std` — the risk transform BEFORE the z-score. Denormalize: `kovatchev_f_inv(normalized * std + mean)`. `RISK_SPACE_CHANNELS = frozenset({"bg_absolute"})` is unconditional and is the single source of truth for which channel crosses (b)→(c) at normalize time. There is no raw-mg/dL input branch.
- **Inputs are RAW simulator post-noise channels** — `data['bg_observed']` (CGM sensor noise applied, not the clean `data['bg']`), `total_carb`, `total_insulin`, `total_exercise` (the sparse three carry the simulator's multiplicative AR(1) σ≈2% absorption noise) — consumed directly, with NO smoothing (bg clamped to the physical range; the sparse channels floored at 0). The same raw bg is the model input, the forecast label and the anchor.
- `SPARSE_LOG1P_CHANNELS` and `RISK_SPACE_CHANNELS` are the single sources of truth. Every normalize/denormalize call site must consult them.
- **Stats fit in risk space (bg) / log1p space (sparse).** The fit runs each channel through `normalization._forward_transform` — no smoothing — then accumulates per-channel Welford stats, so the saved `mean`/`std` match exactly what `data.py` feeds the model. `_forward_transform` is a second copy of the branch `normalize` and `data._build_sample` inline, and the guarantee holds only while all three agree: change one, change them together. There is a **single** stats file, `NORM_STATS_FILE = "normalization_stats.json"`, carrying four keys.
- **`load_normalization_stats` validates on the way in** and raises on a missing `CHANNEL_NAMES` entry, a non-finite mean/std, or `std <= 0`. Nothing downstream can distinguish a malformed file from a good one: a missing channel would reach `normalize` through a `.get` default and train an untrained channel, and `std = 0.0` divides by `0 + 1e-8` and scales its channel by ~1e8. Both train to completion behind a plausible validation table.
- **The file is per pool.** `bg_absolute.mean` is `0.296001733517332` for `cache_balanced_cf` and `-0.021693615541362746` for `cache_hypo_cf`, so it must be reissued whenever `--cache-path` changes: `python normalization.py --from-cache <pool>` (exact against the pool's bytes) or a byte-copy of that pool's own `normalization_stats.json`. **Nothing checks it.** No non-test code path opens `<cache>/normalization_stats.json`; `train.py` loads whatever sits in the CWD; `inference.predict` (and `predict_what_if`, which delegates to it) and `predict_origin_hour` fall back to it silently. `predict_rolling` is the one entry point that refuses — it raises `ValueError` when `normalization_stats` is `None`. A four-key file from the wrong pool passes validation and yields a fully formed sample in a z-space that does not exist.
- `T1DMSIM/cache_balanced` and `T1DMSIM/cache_hypo` carry three-key stats files with no `exercise_equiv` key; `load_normalization_stats` raises on them rather than training an untrained channel.
- **Regenerating stats is a precondition.** Trimming `CHANNEL_NAMES`, moving a channel between `SPARSE_LOG1P_CHANNELS` and `RISK_SPACE_CHANNELS`, or changing the clamp puts the file on disk in the wrong space. The pool's stats are regenerated with the pool by `T1DMSIM/cache_simulator.py` alongside `meta.json`.
- **Transform sites — alter any channel's transform and audit every one.** `normalization.py` defines the `normalize` / `denormalize` pair. `denormalize` has exactly ONE non-test caller: `train.py`'s infill-window reconstruction, which takes the bg z-stack to mg/dL and then overwrites the masked slots with the raw targets. **Five sites re-implement the per-channel transform inline** against the same two frozensets rather than calling the pair — `realdata/run_eval.py._denorm_channel` (inverse); `data.py._build_sample`, `realdata/features.py.build_feature_stack` and `gui.py`'s `_normalize_channel_array` / `_normalize_channel` (forward). `utils.last_bg_mgdl_from_context` is the bg-only inverse (z-unscale → `kovatchev_f_inv_np`) every anchor read crosses, on the inference path and in the export self-check. `inference.py` calls `normalize` forward, owns the risk→mg/dL step through `kovatchev_f_inv`, and inlines the bg z-score for the rolling re-feed; `gui.py` and `realdata/report.py` get their mg/dL through `inference.py` and denormalize nothing themselves, and `realdata/figures.py` is handed mg/dL arrays by `report.py` — it touches neither transform nor `inference`.

## Training invariants

- **fp32 everywhere.** Forward and loss are both fp32 — no autocast, no bf16, no gradient checkpointing. The isfinite / `_halve_optimizer_state` / `_maybe_restore_from_ema` resilience is kept verbatim (soft-DTW in fp32 can still Inf).
- **NaN propagation, not crash.** A non-finite intermediate must REACH train.py's isfinite / `_maybe_restore_from_ema` guard rather than abort the step inside a deep assert. (a) `dilate.py` keeps only pure SHAPE asserts, so a non-finite median/cost flows out to the loss. (b) `utils.kovatchev_f_inv` scrubs a non-finite risk input before the clamp so it can never silently emit NaN mg/dL. (c) `utils.ModelEMA.update` skips blending non-finite incoming weights per tensor, so a single NaN cannot permanently poison the shadow. (d) train.py's resilience guard wraps forward + loss + backward together.
- **Live vs EMA weights**: training runs on live weights; validation runs under the EMA shadow (`ModelEMA.apply_to(model)`). The shadow updates only on accepted steps. **The two Kendall-Gal log-variance parameters are EMA-EXCLUDED** structurally — they live on a separate `KendallGalWeighting` module that is never passed to `ModelEMA` (train.py wraps only `model`); there is no name filter in `ModelEMA.__init__`, whose signature is `(model, decay)`.
- **Loss = `risk_loss.risk_total_loss(q_tau, median, true_bg_mgdl, weighting, valid, mask_idx)`** (soft-DTW DP in `dilate.py`). The target is f-transformed exactly once, at the top: `y_risk = kovatchev_f_target(true_bg_mgdl)`, shared by the pinball and DILATE terms. `valid` and `mask_idx` default to `None`, which means the dense right-edge case — pass them on every real call.
  - **`L_Q` (pinball)** over all `(slot, step, τ)`: `mean rho_τ(y_risk, q_tau)`, `rho_τ(a,b) = (a-b)·(τ − 1[a<b])`, masked by `valid`. There is no per-`d` reweighting: the sampler's right-edge quota corrects the mixture by PLACEMENT, and weights on top of it buy nothing. τ=0.5 is kept as the pointwise level anchor.
  - **`L_D` (DILATE on the median only), computed ONCE PER MASKED SPAN**, not once per sample. Spans are bucketed by length `L`, each bucket stacked to `(n_b, L·PATCH_SIZE)` **C-contiguous patch-major** via `_to_patch_major` for one `dilate_loss` call. Gathering only a span's slots is also what makes a padded slot's gradient exactly zero here — no grad path to it exists at all.
    - **An empty bucket is never dispatched.** `dilate_loss` reduces over the batch axis with `.mean()`, so a `(0, H)` input returns NaN with no exception and no shape assert, and a fixed protocol can leave a bucket empty in *every* batch. That NaN flows through the running totals and past `val_total < best_val_loss` — False for NaN against `inf` — ending the run with no best checkpoint. A span-count-weighted mean does not rescue it: `0.0 * nan = nan`.
    - **Buckets combine by a span-count-weighted mean of the per-bucket scalars**, never by concatenation and never unweighted. DILATE is not scale-free in `H = L·S`: the shape term grows with `H` while the normalised TDI does not, so `DILATE_ALPHA` weights a different mixture in each bucket and `log_sigma_D` absorbs it silently. The per-bucket `loss_D_L{L}` and the span-length histogram `n_spans_L{L}` are logged beside the combined value — two runs are comparable only at an equal span-length mixture.
    - `L_D = DILATE_ALPHA·shape + (1−DILATE_ALPHA)·TDI`, passed **uncentred**. The **shape** term is a DIVERGENCE soft-DTW `sDTW(m,y) − 0.5·sDTW(m,m) − 0.5·sDTW(y,y)`. soft-DTW is **vectorized batched** (anti-diagonal, not a per-sample Python loop) with a max-subtraction-stabilized logsumexp softmin; cost = squared diff in risk space. **TDI** = `<A,Ω>` is the directional derivative of the soft-DTW value along `Ω`, by a one-sided finite difference `[sDTW(C+ε·Ω)−sDTW(C)]/ε` (`DILATE_TDI_FD_EPS`) — one extra `SoftDTWBatch` forward reusing the shape `sDTW(C)`; the median gradient is the exact TDI gradient to `O(ε)`. `SoftDTWBatch` is the only custom `torch.autograd.Function`. `DILATE_GAMMA = 1.0` is a **softmin-softness knob, not an overflow guard**: the stabilized softmin is overflow-free in fp32 down to γ=1e-3, and soft-DTW is 1-homogeneous in `(cost, γ)`. `DILATE_ALPHA = 0.5`.
  - **Kendall-Gal combine** (numerically safe form): `L_KG = 0.5·exp(−2·log_sigma_Q)·L_Q + log_sigma_Q + 0.5·exp(−2·log_sigma_D)·L_D + log_sigma_D`. `log_sigma_Q`/`log_sigma_D = nn.Parameter(zeros(()))` (init `KENDALL_LOGVAR_INIT`), clamped `[-7, 7]`, ndim==0 ⇒ AdamW group (never Muon), EMA-excluded. `L_KG` is the full training total — there is no smoothness, curvature or seam term.
- **Raw target — no input/target asymmetry.** The target is the raw post-noise `bg_observed` at the masked slots, `(B, M, PATCH_SIZE)` mg/dL, NOT f-transformed in the batch, and it is the SAME raw bg the model receives as input.
- **The anchor is read the same way at train and inference.** `data._build_sample` reads it off the raw mg/dL array; `utils.last_bg_mgdl_from_context(context, stats, patch_idx, step_idx)` reconstructs the same physical value from the normalized context by a z-unscale → `kovatchev_f_inv_np` round trip, all `M` cells in one host transfer. **Only VISIBLE cells may be indexed**: feat 0 of a masked patch is a legal-looking `z` that decodes to an ordinary mg/dL, so a wrong index yields a plausible anchor rather than an error. Causality is intrinsic — the anchor reads a visible neighbour, never a masked patch.
- **No carb-noise augmentation.** There is no `CARB_NOISE_AUG_*` and no `data._jitter_carb_norm`; the carb input is the post-noise simulator value, unjittered.
- **Validation runs THREE forwards per batch**, and they are not interchangeable:
  1. the **OBJECTIVE** forward, on the sample's own masked set — `val_loss_total` and therefore checkpoint selection are read off this one, so the selection scalar stays the validation value of the training objective;
  2. the **FORECAST protocol** forward, masking the trailing `PREDICTION_PATCHES` — the whole horizon-keyed clinical suite is read off this one. Those names are defined against a right-edge zone, and a training mask lands there only on the quota's share of windows, so scoring them over the objective forward's slots reads a different patch on most rows;
  3. the **INFILL protocol** forward, masking sampled interior spans — fills the `infill_*` columns, scored against **linear interpolation** between the bracketing visible readings.
- **The two protocols live in `metrics/protocols.py`** and nowhere else. Forecast is scored against **persistence**; infill against **linear interpolation and never persistence** — persistence is a forecasting baseline, and against a two-sided task it is a strawman. Forecast supplies exactly one masked patch at each of `d = 1..4` per window, which is what makes per-`d` calibration well populated there; `@30/@60/@90/@120` min IS `d = 1..4` one-sided. `column()` refuses to name an infill column without a `d`. Forecast columns keep the names `realdata.metrics.compute_suite` defines; `protocols.py` restates none of them.
- **The five proper scoring rules live in `metrics/scoring.py`** and nowhere else: `crps_by_d`, `winkler_by_d`, `coverage_sharpness_by_d` (coverage and the width that bought it, never apart), `joint_coverage_by_d` (simultaneous horizon coverage, distinct from the per-step marginal), and `alarm_operating_curve` (hypo detection rate vs false alarms/day, carrying the **median lead time in minutes** — a detection rate bought at a two-minute lead is not a usable alarm, and neither rate can show that alone). Every rule bins on `d`.
- **Band-edge hypo/hyper detectors.** The headline clinical `hypo_recall`/`hypo_precision` and `hyper_recall`/`hyper_precision` key off the **band edges**, not the median. Derive the edges once: `lo_idx = QUANTILE_LEVELS.index(HYPO_ALARM_QUANTILE_TAU)` (0.25) and `hi_idx = QUANTILE_LEVELS.index(HYPER_ALARM_QUANTILE_TAU)` (0.75) — never a bare literal; then `pred_lo = kovatchev_f_inv(q_tau[..., lo_idx])`, `pred_hi = kovatchev_f_inv(q_tau[..., hi_idx])` (both asserted physical). `pred_hypo = pred_lo < BG_HYPO_THRESHOLD`, `pred_hyper = pred_hi > BG_HYPER_THRESHOLD` — the clinically conservative call on each side. `true_hypo`/`true_hyper` stay off the TRUE bg. **Recall is strict** (`TP/#true`). **Precision forgives near-boundary false alarms** within `EXCURSION_PRECISION_TOLERANCE_MGDL`, so CGM noise near a threshold doesn't deflate it; precision rows carry a `±k` suffix, recall rows don't.
- **Diagnostics that are not selection metrics.** `sign_balance@{30,60,90,120}` (fraction of true BG strictly below the median, target 0.5 — a directional-bias / mean-collapse witness); `inner50_cov@{30,60,90,120}` (empirical coverage of `[τ.25, τ.75]`, target 0.5); the central-90% marginal per-`(h,τ)` coverage; `median_roughness` / `median_roughness_far`; the excursion-amplitude block `exc_*`; the counterfactual `cf_*` rows; the in-training conformal probe `conf_*`. All uncoloured, none feeds the loss or selection.
- **Counterfactual probe.** `train._run_counterfactual_probe` perturbs the masked-span doses against a baseline forecast over the SAME context and reports whether the dose response is physiologically correct: `cf_carb_dbg`/`cf_carb_dir` (a `CF_CARB_BOLUS_G` carb bolus must RAISE BG), `cf_insulin_dbg`/`cf_insulin_dir` (a `CF_INSULIN_BOLUS_U` bolus must LOWER it), `cf_*_monotonic` over a dose sweep, `cf_hypo_rescue` / `cf_hyper_rescue`. Every figure is a DIFFERENCE between the baseline and the perturbed forecast over one shared context, so a short run shifts both together and the sign survives. `CF_EXERCISE_G` is derived from T1DMSIM's two exercise constants, not restated.
- **Time-of-day probe (co-trains the trunk; PER-SLOT categorical).** A 2-layer SiLU MLP over EVERY gathered masked-slot hidden state (no mean-pool) emits `time_pred (B, M, TIME_PROBE_N_BINS)` circular hour-of-day logits, so every per-slot representation the BG head also reads is forced to encode the absolute clock. It is **built under a saved/restored RNG state and re-inited LAST in `_init_weights`**, so every forecast-weight RNG draw is byte-identical with or without it (regression-tested by `test_probe_construction_preserves_forecast_init_rng`). With `TIME_PROBE_DETACH = False` its loss co-trains the shared trunk; the forward VALUE of `q_tau`/`median` is unchanged either way, since the head never feeds them. `TIME_PROBE_N_BINS = round(24 / PREDICTION_HORIZON_HOURS)` and `TIME_PROBE_BIN_HOURS` tiles 24 h exactly. **Slot `j` is patch `mask_idx[:, j]`, so its hour target follows `mask_idx`, not a fixed offset from the context end.** Loss = per-slot CE against a wrapped-Gaussian circular soft label (`utils.time_of_day_bin_ce`, `TIME_PROBE_LABEL_SMOOTH_BINS`; `<=0` ⇒ one-hot), plus a separate teacher-forced **cross-window** penalty (`TIME_PROBE_CROSS_WINDOW_WEIGHT`) coupling consecutive INDEPENDENT-forward windows: `data.py` ships `batch['next_window']`, train.py runs a 2nd forward on it, and `utils.time_cross_window_consistency_loss` rotates window k's origin resultant by `2π·PREDICTION_HORIZON_HOURS/24` and matches window k+1's in the raw `(cos, sin)` plane (atan2-free). Both terms are scaled by `TIME_PROBE_LOSS_WEIGHT` in the TRAINING backward ONLY — never in `risk_total_loss`, `val_loss_total` or checkpoint selection. **Decode** via `utils.time_of_day_decode_bins(logits, n_bins) -> (hour, R)`: softmax, probability-weighted resultant, `hour = atan2(sin, cos)`, `R = hypot(cos, sin) ∈ [0,1]` as the confidence. Reported: `tod_mae_h` / `tod_acc_*` / `tod_conf`, the clock-reliability rows `tod_bias_h` / `tod_std_h` / `tod_p90_h` / `tod_gross_rate` / `tod_mae_hiconf`, and the jump witnesses `tod_jump_h` and `tod_xwin_jump_h`. These need the full residual distribution, so `_run_validation` accumulates per-sample arrays and finalizes once; circular stats via `utils.circular_bias_hours` / `circular_std_hours` (naive angle averaging breaks at the 24 h wrap). **Clock-face surfaces** are diagnostic-only: `inference.predict` / `predict_what_if` / `predict_rolling` take a trailing opt-in `return_time=False` whose default adds no key. The geometry lives in ONE tested numpy place — `utils.aggregate_origin_belief` and `utils.clock_wedge_geometry` — consumed by two thin no-math adapters, `gui_renderer.draw_clock_face` (pygame) and `clock_face.draw_clock_axis` (matplotlib).
- **Checkpoint selection (single "best" snapshot).** `t1dmai_best.pt` is saved on the minimum `val_loss_total` (= `risk_total_loss` on the objective forward); the periodic `t1dmai_step_{N}.pt` snapshots are independent. There is no clinical-composite selector and no resume / checkpoint-loading path. The checkpoint carries `arch_version`, `loss_schema` and the three sampler constants (`mask_span_lengths`, `max_masked_patches`, `mask_right_edge_quota`) as provenance; `finetune/finetune.py` refuses a checkpoint whose recorded sampler differs from the live one.
- **CSV/log keys.** `training_log.csv` carries `loss_total`, `loss_Q`, `loss_D` (with `loss_D_shape` / `loss_D_tdi`, the per-bucket `loss_D_L{L}` and `n_spans_L{L}`), and the two Kendall-Gal weights `log_sigma_Q` / `log_sigma_D`, at `LOG_INTERVAL` cadence with an EMA-smoothed total. `validation_log.csv` carries the protocol columns, the `infill_*` columns, the diagnostics above and the `cf_*` / `conf_*` blocks. Update the header AND the row writer together.
- **Conformal-calibration partition — ACTIVE.** A third `T1DMDataset` on a disjoint seed band (`master_seed + CALIBRATION_RESERVE_SEED_OFFSET`, disjoint from the train hashed seeds and from normalization's own offset band) feeds split-conformal recalibration of the BG bands. It touches neither the loss nor the headline validation metrics.
- **Conformal layer (`conformal.py`, pure numpy, mg/dL).** `fit_quantile_conformal(cal_q (N,S,K), cal_true (N,S), levels, median_idx) -> delta (S,K)` — per-(step, quantile) ASYMMETRIC split-conformal: `delta[s,k]` is the side-aware empirical τ-quantile of the residual `true − q_k` (`_conformal_offset`: `ceil((n+1)τ)` for an UPPER edge τ≥0.5, `floor((n+1)τ)` for a LOWER edge τ<0.5 — `ceil` on a lower edge is anti-conservative). `apply_quantile_conformal` adds `delta` and re-enforces the three LOAD-BEARING invariants (unit-tested): MEDIAN held FIXED, fan kept MONOTONE, all-zero delta = identity. The delta is mg/dL (downstream of `f_inv`) and must be RE-FIT per target distribution — validity rests on cal/test exchangeability.
- **Mondrian (region-binned) conformal — `mondrian.py`.** `conformal.py` fits one MARGINAL delta, which under-covers one regime and over-covers the other. `mondrian.fit_mondrian` re-fits it once per REGION bin, where the bin is a function of the WINDOW taken from where the forecast is HEADING (`forecast_destination` — the median line's mean over the final patch), not from the last observed BG. `REGION_EDGES = (110.0,)`: one edge inside the euglycaemic band. **A bin edge at a clinical threshold (70) is the one placement to avoid** — it splits the windows that decide the alarm across two separately-fit corrections and starves the low bin. Because the median is held fixed, a window's region is identical before and after correction, so the binning is not circular. **Below `n = 39` a bin takes the MARGINAL delta** and `fit_mondrian` records that it did: at τ=0.05 the order statistic `floor((n+1)τ)` only reaches index 2 at `n >= 39`, so below that the bin's own offset IS its most extreme residual. Every coverage figure is reported with `n`, the DISTINCT PATIENT count and the MEAN BAND WIDTH — coverage alone is not interpretable, since it is bought with width.
- **`calibrate_conformal.py` fits both protocols and ships one.** The forecast delta is written to `ckpt['conformal_delta']` — that is the band that ships. **`inference.predict` and `predict_rolling` never load it themselves**: each takes a `conformal_delta` argument defaulting to `None` (⇒ raw bands, bit-identical) and applies it only when the caller passes it. `realdata/report.py.load_model` is what lifts it off the checkpoint, onto `m.conformal_delta`, for its call sites to forward — and some paths pass `None` deliberately, so a band's calibration state is a property of the call, not of the checkpoint. The infill delta is stored under `conformal_delta_infill` with `shipped = False`: infill residuals are the easier ones, and folding them in buys an interval the forecast cannot honour.
- **Disjoint cache slabs — no train/val leak.** `data.py` partitions the cache index space into DISJOINT train/val/cal slabs via a `cache_partition` argument, so the val and cal seed bands cannot collapse back onto train rows through `patient_seed % pool_size`. The validation datasets train.py builds pass `cache_partition='val'`.
- **Rolling prediction BG fill.** `predict_rolling()` is **BG-autoregressive only**: `median → f_inv → mg/dL → normalize → bg_absolute slot 0`. carb (feat 1) / insulin (feat 2) / exercise (feat 3) come from `overrides_fn` if supplied, else from the **zero-RAW normalized baseline** — `normalize(0.0)` per channel, NOT `torch.zeros`, whose `z = 0` is a phantom event that corrupts a no-dose roll. The caller MUST pass `normalization_stats`: `predict_rolling` raises `ValueError` on `None` rather than falling back to the CWD file.
  - **A sparse channel's `z = 0` decodes to `expm1(mean)`, NOT to `mean`.** `z = 0` is the channel's own fitted log1p-space mean, and the sparse inverse is `expm1(z·std + mean)`. Under `cache_balanced_cf` that is **0.4679 g/step carb, 0.1503 U/step insulin, 0.0250 g/step exercise** — compute them, never read them off `normalization_stats.json`, whose `mean` fields are 0.3839 / 0.1400 / 0.0247. The gap is 18% on carb, 7% on insulin and 1% on exercise, so the channel with the smallest phantom is the one a quoted mean gets right, and the error survives inspection. The figures are pool-dependent; the `expm1(mean)` relation is not.
  - **The rolling band carry is `predict_rolling`'s own, not `assemble_quantiles`'s.** Each roll measures its NATIVE terminal-step half-width `0.5·(q_tau[-1,-1,-1] − q_tau[-1,-1,0])` on the PRE-carry fan; shifts that roll's fan by the running carry (`τ>.5` columns `+ carry`, `τ<.5` columns `− carry`, the median column untouched); derives `bands = f_inv(q_tau)` from the widened fan, with conformal on top of that; and only THEN adds the native half-width to the carry. So roll 0 is unwidened and the carry grows about linearly with the roll count. Measuring the half-width on the POST-carry fan instead compounds it geometrically (`carry → 2·carry + native`) and pins the long-horizon band to the physiological rails.
- **Long-horizon validation via rolling — conditioned on announced doses.** A single forward covers `PREDICTION_HORIZON_HOURS`, so the table reports `bg_rmse @{30,60,120}` (single-pass) plus rolling `@{180,360,480}` and night-only `night_bg_rmse @{180,360,480}m`, rolled out to `NIGHT_LONG_HORIZON_HOURS`. The roll is conditioned via `_make_long_horizon_overrides_fn` from `bg_formula_data['extended_{carb,insulin,exercise}_{norm,raw}']` — the nocturnal-hypo use case knows the programmed basal ahead of time. BG stays autoregressive; conditioning the doses is what tames the zero-basal-OOD runaway an unconditioned roll would suffer. Set `NIGHT_LONG_HORIZON_HOURS == PREDICTION_HORIZON_HOURS` to skip rolling.

## The export is a right-edge specialisation

`exporters/modified_forward.py` deliberately does NOT run the general masked forward. It differs in
exactly three ways, all required by the on-device contract and all documented in
`T1DMCOMMON/SPEC/inference.md`:

1. its only mask input is an external **additive float** struct mask, `NEG_FILL = -30000.0` where
   blocked (not `-inf`, so an fp16 NPU softmax stays finite; in fp32 `exp(-30000)` underflows to 0.0,
   making it bit-identical to `-inf`), built by `build_struct_mask` at a fixed `T = MAX_SEQ_LEN`;
2. the head reads the trailing `PREDICTION_PATCHES` patches as a **slice**, taking no `mask_idx`;
3. the graph is **cut at `head_raw`** `(B, P, S, 1 + 2·N_SPREADS)` in risk space — no `anchor_bg`, no
   `assemble_quantiles`. Everything downstream is Rust.

It emits two outputs in a fixed order, `head_raw` then `time_logits`. `exporters/descriptor.py` is
the SOLE pre/post source for the on-device core: the app reads the artifact plus the descriptor and
never parses the `.pt`. Three engine modules sit on the same modified forward —
`executorch_xnnpack.py` (fp32 CPU, the reference/authority), `executorch_vulkan.py` and
`litert_npu.py` — each self-checking on host that the modified forward matches the stock bool-mask
forward's `head_raw` and that the lowered artifact matches eager.

## Code style

- All tunable params as module-level UPPERCASE constants in `config.py`. Never duplicate them.
- Type hints on all function signatures. Docstrings with tensor shapes on all public functions.
- Assertions at function boundaries for shape validation.
- No comments unless asked.

## Documentation sync

Code changes that alter architecture, training, data pipeline, or user-facing interfaces must also update the corresponding markdown in the same change:

- `ARCHITECTURE.md` — model/loss/training specification
- `README.md` — public summary
- `docs/INFERENCE.md` — a **stub**: the model contract is specified once for the whole suite in `T1DMCOMMON/SPEC/inference.md`. Never restore a copy of it here; keep only what is local to this repository.

Markdown drift is a bug. Prefer editing sections over appending. Delete stale paragraphs rather than leaving them. `README.md`, `ARCHITECTURE.md`, `docs/` and `models/README.md` are PUBLIC — no second person, no prescriptive prose, no ranked action items, no speculation about future work.

## Key files

- `config.py` — all hyperparameters as uppercase constants (single source of truth)
- `d_balance.py` — the exact two-branch `d` histogram of the sampler; `metrics/protocols.py`'s `SAMPLER_REFERENCE` is produced from it
- `model.py` — T1DMAI model class; `make_step_basis`, `build_rope_cache`, `apply_rope`
- `data.py` — on-the-fly and cached sample generation; `sample_mask_spans`, `_anchor_step_for_span`, `_mask_slots`, `BG_MASKED_FEAT`
- `train.py` — training loop entry point
- `train_blind.py` — the unconditioned fork of `train.py` (all four signals withheld on masked patches via `data`'s `blind` flag, no counterfactual probe, unconditioned rolling validation, own `checkpoints_blind/` + `logs_blind/`); checkpoints stamp `masked_channel_policy`, which `finetune/finetune.py` and `calibrate_conformal.py` refuse to mix
- `inference.py` — prediction functions (standard, what-if, rolling), each with an opt-in `return_time`; `predict` takes an explicit `mask_spans` masked set (`None` selects the trailing forecast), what-if and rolling are right-edge by construction
- `normalization.py` — channel statistics; `CHANNEL_NAMES`, `SPARSE_LOG1P_CHANNELS`, `RISK_SPACE_CHANNELS`, `load_normalization_stats` (validating)
- `muon.py` — Muon optimizer implementation
- `utils.py` — seed hashing, the two attention-mask builders, `kovatchev_f` / `kovatchev_f_target` / `kovatchev_f_inv` (+ numpy siblings, the only b↔c bridges), `assemble_quantiles`, `global_median_dim` / `_span_layout` / `get_global_median_basis`, `last_bg_mgdl_from_context`, the clock-face core (`aggregate_origin_belief`, `clock_wedge_geometry`), `ModelEMA`
- `risk_loss.py` — `risk_total_loss` (pinball + per-span DILATE + Kendall-Gal combine; f-target applied once)
- `dilate.py` — vectorized batched soft-DTW (`SoftDTWBatch`); TDI is its directional derivative via a finite difference
- `cg_ega.py` — vectorized CG-EGA (Kovatchev 2004) clinical-accuracy metric
- `conformal.py` — marginal split-conformal recalibration (`fit_quantile_conformal` / `apply_quantile_conformal` / `band_coverage`); median-fixed, monotone, zero-delta identity
- `mondrian.py` — region-binned conformal: `forecast_destination`, `region_bin`, `fit_mondrian`, `fit_infill_conformal`, `apply_mondrian`, `bin_report`
- `calibrate_conformal.py` — post-training fit of the shipped forecast `conformal_delta` (and the unshipped infill one) on the reserved partition
- `metrics/scoring.py` — CRPS, Winkler, coverage-with-sharpness, joint horizon coverage, alarm operating curve with median lead; all binned on `d`, all mg/dL at the boundary
- `metrics/protocols.py` — the two fixed protocols (forecast vs persistence, infill vs linear interpolation), the `d` axis, column naming, the sampler `d` histogram and the cohort census
- `exporters/` — `modified_forward.py` (right-edge specialisation + struct mask + checkpoint load), `descriptor.py` (the on-device JSON contract), and the `executorch_xnnpack` / `executorch_vulkan` / `litert_npu` engine modules
- `finetune/` — `finetune.py` (leave-one-patient-out on a real cohort), `finetune_personal.py` (holds out a representative DAY of one record), `finetune_multi.py` (pooled multi-cohort); all reuse the pretraining machinery verbatim
- `model_health.py` — capacity / staleness audit of a checkpoint, keyed to `resize_model.py`'s knobs
- `resize_model.py` — rewrites `config.py` from manual architecture overrides; reports the resulting parameter count (computed, not targeted)
- `T1DMSIM/cache_simulator.py` (external, in the T1DMSIM repo) — pre-generates a compressed simulator pool consumed by `T1DMDataset(cache_path=...)`, and emits the four-channel `normalization_stats.json` alongside `meta.json`
- `realdata/report.py` — shared real-data report assembly; `load_model` builds `T1DMAI()` and loads the state dict, attaching `conformal_delta` without changing its return tuple
- `metrics/real/`, `metrics/augmented/`, `metrics/sim/` — the three evaluation reports (announced-event real CGM; the same with unlogged events reconstructed; fresh T1DMSIM patients at fixed seeds)
- `metrics/` (top level) — the real-data deep dive: `curves*.py`, `amp_var.py`, `q2_event_response.py`, `whatif.py`, `shift15.py`, `augexp.py`, `meal_stats.py`, `time_probe.py`, each writing its own JSON and its panels to `metrics/figures/`. The shared chrome — palette, cohort→hue map, mark and label conventions — lives in `metrics/figstyle.py`, the single copy: never inline a colour or an rcParam in a probe. **`metrics/rebuild_all.sh` does not run `augmented/build_report.py`, `augmented/make_comparison_figures.py`, `curves_aug.py`, `shift15.py` or `augexp.py`** — those are invoked by hand, so reading `metrics/augmented/README.md`, `shift15.json` or `augexp.json` after a rebuild reads output built against whatever checkpoint was live when someone last ran them. Re-run them alongside it, or state which checkpoint each number came from.
- `make_card.py`, `make_figures.py`, `make_readme_figures.py` — model-card, training-figure and README-figure generation
- `gui.py`, `gui_renderer.py`, `gui_controls.py`, `gui_state.py`, `clock_face.py` — the pygame GUI and the two clock-face host adapters
- `models/` — per-capacity training logs, figures and metric reports, plus `compare.py`; it holds no checkpoints
- `tests/` — pytest suite

## Testing

- Run `python -m pytest tests/ -v -s` after every implementation phase.
- The `-s` flag is critical — tests print diagnostic `[DUMP]` lines that catch silent numerical issues.
- Test with `num_workers=0` to avoid multiprocessing issues during debugging.
- Verify the attention mask visually before trusting it.
- Check for NaN after every new component is added.
- `tests/test_mask_sampler.py`, `test_masked_objective.py`, `test_no_mask_conditioning.py` and `test_scoring.py` / `test_mondrian.py` are the gates on the masked objective; `tests/test_bitident.py` pins the forward against `tests/bitident_ref.json`.

## Simulator

- Located at `./T1DMSIM/` (a symlink to the sibling checkout, read-only). Import: `from T1DMSIM.simulator import T1DMSimulator`.
- Simulator values are in "amount per step" units (grams/step for carbs, units/step for insulin, carbohydrate-equivalent grams/step for exercise).
- The model works in normalized space. Track which representation you are in at every point.

## Running tests

Do **not** wrap pytest (or any test command) in `timeout`. Some tests in this
repo run long simulator warmups or training-step smoke checks and a hard
timeout will mask real convergence/throughput regressions as spurious failures.
If a test appears to hang, investigate the test itself rather than killing it
with `timeout`.

Use the Bash tool's own `timeout` parameter (in milliseconds) only when you
genuinely need a wall-clock limit on a long-running command, never to
short-circuit a test.
