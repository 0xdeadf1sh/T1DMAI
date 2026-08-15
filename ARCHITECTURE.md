# T1DMAI architecture

> [!CAUTION]
> **Research and educational use only — not a medical device.** T1DMAI is a
> research artifact with no clinical validation and no regulatory clearance. It
> must not be used for medical, diagnostic, dosing, or treatment decisions. See
> the [README](README.md) for the full disclaimer.

This document specifies the model and how it is trained: the tensors, the blocks,
the heads, the loss, the optimiser, and what validation measures.

It is **not** the inference contract. How a consumer loads a checkpoint and turns
the head's raw output back into mg/dL — the three spaces, the exact Kovatchev
constants, the quantile-assembly algebra, the decode recipe — is specified once
for the whole suite in
[`T1DMCOMMON/SPEC/inference.md`](https://github.com/0xdeadf1sh/T1DMCOMMON), and
[`docs/INFERENCE.md`](docs/INFERENCE.md) maps it onto this repository. Where the
two touch, that specification is authoritative and this document defers to it.

Every numeric dimension lives in `config.py`, which holds the current values.
Formulas here stay correct across a `resize_model.py` resize; literal values do
not.


## Contents

- [Overview](#overview)
- [Dimensions](#dimensions)
- [The masked objective](#the-masked-objective)
- [Inputs](#inputs)
- [Encoder](#encoder)
- [Heads](#heads)
- [Loss](#loss)
- [Optimisation](#optimisation)
- [Validation](#validation)
- [Conformal calibration](#conformal-calibration)
- [Report metric basis](#report-metric-basis)
- [Normalization statistics](#normalization-statistics)
- [Simulator cache](#simulator-cache)
- [Inference and export](#inference-and-export)
- [Parameter count](#parameter-count)


## Overview

An encoder-only transformer over patches of CGM glucose, carbohydrate
appearance, insulin action and exercise disposal. Every patch of a window is
either **visible** — its glucose observed — or **masked**, its glucose withheld
while the other three channels keep their true or announced values. The model
emits a fan of `N_QUANTILES` blood-glucose quantiles per 5-minute step of every
masked patch.

A masked span ending at the last patch is a **forecast**, one starting at patch 0
a **backcast**, anything between visible patches an **infill**: three cases of one
objective, not three modes. Training samples all of them. The deployed case is
the forecast, whose span is `PREDICTION_HORIZON_HOURS`.

One model covers every hour of the day; there is no day/night split and no
per-patient parameter.

The forecast is produced and optimised in **Kovatchev risk space**, not mg/dL.
The transform warps the glucose axis so that a fixed *mg/dL* error costs more
loss at low glucose than at high — a 20 mg/dL miss costs about four times the
risk-space error at 60 mg/dL that it does at 300.
That is the whole mechanism behind the model's hypoglycemia bias: no term in the
loss mentions hypoglycemia, and nothing is focally reweighted. Glucose enters
through the same transform, so input, output, target and loss share one space,
and only the reporting layer converts back to mg/dL.

The constants are re-anchored so that `f(40) = −√10` and `f(400) = +√10`. Those
anchors are not the clamp. `SPEC/invariants.md` §4 governs both, and distinguishes
this *model* risk space from the *clinical* LBGI/HBGI space the simulator uses.


## Dimensions

`resize_model.py` rewrites the settable constants in place and recomputes the
parameter count. It refuses to write unless `HEAD_DIM = D_MODEL // N_HEADS` lands
in `{16, 32, 64, 128}` — outside that set `scaled_dot_product_attention` falls
back to the math kernel and materialises the full `T × T` matrix — and unless
`PATCH_SIZE` divides 12, so a whole number of patches tiles the hour.

| Constant | Set by | Meaning |
| --- | --- | --- |
| `D_MODEL` | `--d-model` | Residual-stream width |
| `N_LAYERS` | `--layers` | Transformer blocks |
| `N_HEADS` | `--heads` | Attention heads; `HEAD_DIM = D_MODEL // N_HEADS` |
| `FFN_DIM` | `--ffn-mult k` | `k × D_MODEL`, SwiGLU inner width |
| `BG_HEAD_HIDDEN` | `--bg-head-hidden-mult k` | `k × D_MODEL`, glucose-head hidden width |
| `PATCH_SIZE` | `--patch-size` | Timesteps per patch; `PATCH_SIZE × 5 min` is the patch span |
| `MIN_CONTEXT_PATCHES` | `--min-context-patches` | Shortest sampled context |
| `MAX_CONTEXT_PATCHES` | `--max-context-patches` | Longest context, and the left-pad ceiling |

The mask sampler (see [The masked objective](#the-masked-objective)):

| Constant | Meaning |
| --- | --- |
| `MASK_SPAN_LENGTHS` | The pool each masked span's length is drawn from |
| `MAX_MASKED_PATCHES` | `M` — the sampler's cap on masked patches per sample, and the head's fixed slot count |
| `MASK_RIGHT_EDGE_QUOTA` | Share of windows whose last span is pinned flush against the final patch |

Derived, not settable:

| Constant | Definition |
| --- | --- |
| `N_INPUT_FEATURES` | 5 — `[bg_absolute, carb_intake, insulin_combined, exercise_equiv, bg_masked]`, of which the leading four are normalized signal channels |
| `PATCH_DIM` | `PATCH_SIZE × N_INPUT_FEATURES` |
| `PREDICTION_PATCHES` | `PREDICTION_HORIZON_HOURS × 60 / (PATCH_SIZE × 5)` |
| `MAX_SEQ_LEN` | `MAX_CONTEXT_PATCHES + PREDICTION_PATCHES` |
| `N_QUANTILES` | `len(QUANTILE_LEVELS)` — `SPEC/invariants.md` §6 fixes the levels and their ascending order for the whole suite |
| `N_SPREADS` | 3 — spreads per side; the head emits `1 + 2·N_SPREADS` values per step |

At the released defaults a patch is 30 minutes, the context runs 48–96 patches
(24–48 h), and the forecast protocol's span is 4 patches (2 h). The floor sits
well above every autocorrelation length `T1DMSIM/diff/README.md` §0.5 measures,
covers one full basal cycle, and leaves enough real context for the 8-hour
nocturnal roll.


## The masked objective

A sample's window is `T` patches, `T ≤ MAX_SEQ_LEN`. `data.sample_mask_spans`
draws its masked set:

```
n_spans   ~ U{1 .. MASK_MAX_SPANS}
each L_i  ~ U(MASK_SPAN_LENGTHS), independently
sum(L) > MAX_MASKED_PATCHES  ⇒  resample the WHOLE length vector
placement, with probability MASK_RIGHT_EDGE_QUOTA:
            the LAST span flush against patch T-1, the rest over the prefix
placement, otherwise:
            stars-and-bars over the n_spans + 1 gaps
```

The over-budget rejection redraws the whole length vector rather than one
element: per-element redrawing yields a different length distribution, and so a
different `d` histogram. Placement is rejected on nothing, and there is no
curriculum and no annealing. The right-edge quota is the one departure from
uniform placement, and it changes only where the last span lands: both branches
draw `n_spans` and the length vector identically, so the span-length histogram is
quota-independent and only the `d` histogram moves. Under uniform placement alone
a forecast — the case the model is deployed as — falls out as an accident of
about 3 % of windows, and the band it emits there loses coverage over training
while every selection scalar improves. One **mandatory visible separator** sits
between neighbouring spans,
so two masked spans never abut; the separator is what makes the anchor, the
per-span median basis and the DILATE length bucket well defined per span.

`data._mask_slots` expands the spans into the head's `M = MAX_MASKED_PATCHES`
fixed slots. A sample with fewer masked patches pads the surplus; a padded slot
gathers patch 0, so its output is a real number against a real target, and every
loss and metric path discards it by the `(B, M)` `valid` flag.

`d` — a masked patch's distance in patches to the nearest visible evidence on
**either** side — is the difficulty axis every masked-BG metric bins on. It is
never span length, which confounds one-sided and two-sided cases at equal
difficulty. Under the forecast protocol slot `j` sits at
`d = j + 1` one-sided, which is what makes the 30 / 60 / 90 / 120-minute columns
`d = 1..4`.

### The sampler's `d` histogram

`d_balance.d_distribution` enumerates both placement branches exactly, over the
window-length mixture the sampler draws from. It is an enumeration rather than a
measurement: at 10⁵ draws the per-position 1σ is several times the effect being
measured, so a correct sampler misreports the histogram in every replicate.
`metrics/protocols.py`'s `SAMPLER_REFERENCE` is produced from it and pinned to
the knobs it was enumerated under, and `sampler_reference_applies()` refuses the
comparison once one of those knobs moves.

The three sampler constants ride in every checkpoint's `training_config`, and a
loader compares them against the live config before accepting the weights: no
parameter shape depends on the sampler, so a strict state-dict load accepts
weights trained under any of them.


## Inputs

Five features per 5-minute step: four signal channels and one bit. There are no
time-of-day features — day and night are inferred from the trajectory alone.

| Feature | Units | Transform before the model |
| --- | --- | --- |
| `bg_absolute` | mg/dL | Clamp to `[BG_CLAMP_MIN, BG_CLAMP_MAX]`, Kovatchev `f`, z-score |
| `carb_intake` | g / step | Floor at 0, `log1p`, z-score |
| `insulin_combined` | U / step | Floor at 0, `log1p`, z-score |
| `exercise_equiv` | g / step, carbohydrate-equivalent | Floor at 0, `log1p`, z-score |
| `bg_masked` | bit, one value per patch | None — it is never normalized |

`log1p` is near-linear near zero, so the dense basal baseline passes through
almost unchanged while rare meal, bolus and exercise spikes are compressed out of
the channel's standard deviation. `normalization.py` holds the membership sets —
`RISK_SPACE_CHANNELS` for glucose, `SPARSE_LOG1P_CHANNELS` for the other three —
and every forward and inverse transform consults them, so the pipeline stays
invertible. `bg_masked` is in neither set: it carries no mean, no std and no
`log1p`, which is why `CHANNEL_NAMES` has four entries where
`N_INPUT_FEATURES` is five.

`exercise_equiv` is carbohydrate-equivalent glucose disposal in g/step — the
quantity the simulator subtracts from the appearance term — so it takes carb's
encoding exactly, never the Kovatchev transform and never a rescaling to an
intensity.

All four signal channels are the **raw post-noise** simulator signals. There is
no smoother anywhere, on the inputs or the target. The same raw glucose is the
model input, the forecast target and the anchor, so there is no input/target
asymmetry, and a live CGM stream needs no on-device filter to reproduce.

Insulin sensitivity and hepatic glucose output stay in the cache but never reach
the model. They are internal states a real CGM cannot observe, so withholding
them forces the model to forecast from signals deployment can supply.

`carb_intake`, `insulin_combined` and `exercise_equiv` are **rates**, not
ingestion, injection and session instants: grams entering the blood, units acting,
and grams of carbohydrate-equivalent disposal in each 5-minute step. Pretraining
takes them from the simulator directly. Real records rarely store carbohydrate and
insulin that way, so `realdata/features.py` reconstructs them by convolving logged
amounts with population-mean kernels, with the fidelity limits the README sets out.
A record whose events already carry their resolved series instead supplies them on
`Segment.carb_curve` / `Segment.insulin_curve`, which bypass the kernels; the
transforms in the table above are unchanged either way. `Segment.exercise` is
already a resolved g/step curve, so nothing on the input path convolves it, and
every real adapter fills it with zeros.

### The index map

Every patch carries all five features, step-major, so feature `f` is the stride
slice `[:, f::N_INPUT_FEATURES]`. On a **masked** patch:

- `bg_absolute` is **zeroed** — it is what the model predicts;
- `bg_masked` is 1.0 in all `PATCH_SIZE` step-major columns of feat 4;
- `carb_intake`, `insulin_combined` and `exercise_equiv` **still carry the
  carbohydrate-appearance, insulin-action and exercise-disposal curves, per
  5-minute step** — not the moment of eating, not the injection instant, not the
  start of a session, and not a delivery schedule (`SPEC/invariants.md` §5): the
  true values during training, the caller's announcement at inference.

The masked set is **announced, not inferred**. Masking is not a position rule,
and `z = 0` in a withheld glucose slot decodes to an ordinary reading rather than
a sentinel, so feat 4 is the only thing that tells the model which patches it
must predict. `data.collate_fn` and `inference._assert_mask_announced` each assert
that the bit reproduces the masked set — the one the attention mask was built from
on the training path, the one the head gathers by `mask_idx` on the inference path
— and that the bit is constant across a patch's columns.

The model is therefore always conditioned on a declared plan. There is no
conditioned/unconditioned split, which is why what-if forecasting is a property
of the forward pass rather than a separate mode — announcing a different plan
just writes different values into those three slots.

`config.CHANNEL_TO_FEAT` is the single mapping from an announceable output
channel to its input feature slot. Both the data pipeline and the inference
override path read it, so no second offset literal exists.

### The anchor

One anchor per head slot, in mg/dL. It is **one-sided and left-preferring**: the
last step of the span's left neighbour patch, or — when the span starts at patch
0 and there is no left neighbour — the first step of the right neighbour. Every
slot of a contiguous span carries the same value, and `assemble_quantiles` anchors
that slot's median at `f(anchor_bg)`. A right-edge span reduces to the last
visible glucose before the forecast, so no separate origin rule exists.

Training and inference reach the value by two paths that agree in value but not
in mechanism: `data._build_sample` reads the raw mg/dL trajectory at the anchor
step, while inference and export reconstruct the same physical value from the
context cell via `utils.last_bg_mgdl_from_context` — un-z-score, then the inverse
risk transform — matching to a sub-ulp round-trip difference. Only a visible cell
may be indexed: feat 0 of a masked patch is a legal-looking `z` that decodes to a
plausible glucose, so a wrong index yields a wrong anchor rather than an error.

The anchor is not the distance the metrics bin on. Ignoring the near side leaves
the last slot of a two-sided span anchored from the far end while sitting one
patch from visible evidence on the other, which is a large minority of
supervision under the sampler (`data._mask_slots` carries the enumerated
fraction). It costs no information — a masked row attends in both directions —
and only makes the head's offset parameterisation work harder.


## Encoder

### Patch embedding

`PATCH_SIZE` consecutive timesteps become one token: `PATCH_DIM` raw values
through a single `Linear(PATCH_DIM, D_MODEL)` with bias. The bias matters because
a masked patch's glucose slots are a hard zero — a legal reading rather than a
sentinel — and a learned offset lets the projection use the `bg_masked` bit to
tell the two apart.

There is no patient embedding. `forward` takes no patient argument; identity is
whatever the context window implies.

### Position

One signal, relative: **RoPE** on Q and K, at base frequency `ROPE_BASE`. The
cosine and sine tables depend only on sequence length and head dimension, so they
are built once per forward pass and shared across every block. Nothing else
biases a logit by the distance between its two positions, and every head carries
the same positional signal.

### Block

Pre-norm, two residual writes per block, so output projections initialise at
`base_std / sqrt(2 · N_LAYERS)`.

```
x = x + TemporalSelfAttention(RMSNorm(x))
x = x + SwiGLU(RMSNorm(x))
```

Attention is standard multi-head with per-head RMSNorm applied to Q and K before
RoPE, which bounds the logits and stops gradient spikes through a deep stack.
SwiGLU is `(SiLU(x W1) ⊙ x W3) W2`, no biases. RMSNorm is
`x / sqrt(mean(x²) + 1e-6) · gamma`, no bias and no mean subtraction. Everything
runs natively in fp32.

A final RMSNorm follows the last block.

### Attention mask

The mask follows the visible/masked/padding labelling, not a position rule:

| | to visible | to masked |
| --- | --- | --- |
| **from visible** | attend | **blocked** |
| **from masked** | attend | attend |

Masked patches are decoded jointly, so they attend to each other in both
directions. The one blocked region is what prevents a future leak: nothing the
model must predict can influence the evidence's representation.

Shorter contexts are left-padded to the batch maximum, never to `MAX_SEQ_LEN`.
Padding rows and columns are blocked outright; the diagonal is then forced True
at every position, padding included, so a padding row attends only to itself and
its softmax is never fully masked. Padding outputs are never read — the heads
gather by index, and a padded slot's output is dropped by `valid`.

`utils.create_attention_mask_from_visible` builds it in four lines, all four
load-bearing, and memoizes nothing: the masked set varies per sample, and no
cheap key identifies it, so a memo would hand one sample's mask to another with
no shape error. The mask reaches `scaled_dot_product_attention` as a **bool**
(True = attend); a per-sample `(B, T, T)` mask gains the head axis in
`T1DMAI.forward` before it gets there, since broadcasting aligns from the right
and would otherwise put the batch axis onto the head axis. Nothing additive is
materialised per layer.


## Heads

Both heads read the `M` masked-patch hidden states after the final norm, gathered
by `mask_idx` — one `D_MODEL` vector per slot. The gather is not a trailing
slice: the masked set may sit anywhere in the sequence.

### Blood-glucose quantile head

A 3-layer SiLU MLP emitting `BG_HEAD_STEP_BASIS_DIM` coefficients per channel per
slot, where a channel is the median offset or one of the `2 · N_SPREADS`
spreads. A fixed orthonormal basis then expands those coefficients across the
patch's timesteps:

```
coeff    = bg_head(pred).view(B, M, K, 1 + 2·N_SPREADS)
head_raw = einsum('sk,bmkc->bmsc', step_basis, coeff)     # (B, M, S, C)
```

`step_basis` is `(PATCH_SIZE, K)` — the low-frequency DCT-II modes, or orthonormal
polynomials under `BG_HEAD_STEP_BASIS_TYPE = 'poly'`. Emitting fewer coefficients
than the patch has steps makes the highest-frequency within-patch mode
unrepresentable, so the head **cannot** produce an intra-patch zigzag. Setting
`K = PATCH_SIZE` recovers a fully free per-step head. `step_basis` is a
registered buffer: it travels with the state dict and with `.to(device)`, and it
holds no parameters.

The final layer initialises at `std = BG_HEAD_INIT_SCALE`, so at step 0 the
coefficients are near zero, the median offset is near zero, and every slot's
forecast is a flat persistence line at its own anchor.

### Quantile assembly

`utils.assemble_quantiles` turns `head_raw` into the ascending fan. It is the
single chokepoint — training, `predict` and `predict_rolling` all pass through
it, so a config change propagates identically everywhere. The exact algebra is in
`SPEC/inference.md` §8.1; what matters here is why it has the shape it does.

**Assembly is per span.** `utils._span_layout` groups the `M` slots into
contiguous spans from `mask_idx`, exploiting the sampler's mandatory separator:
adjacency in `mask_idx` identifies a span exactly. Nothing accumulates or
low-passes across the visible patches between two spans, and a padded slot falls
out as its own singleton.

**The median is projected, not integrated.** A span's per-patch median offsets
are flattened patch-major over that span's own `L · PATCH_SIZE` steps and
projected onto a fixed low-frequency DCT-II subspace of `G_L` columns, where
`G_L = max(1, ceil(BG_HEAD_MEDIAN_GLOBAL_DIM · L / PREDICTION_PATCHES))`
(`utils.global_median_dim`). A projection is an L2 contraction, so the offset is
bounded and cannot accumulate across patches. Column 0 is the constant mode,
which preserves the persistence level. The low-pass also removes the seam
sawtooth, so no separate seam penalty is needed, and exact seam continuity is
deliberately not enforced.

`G_L` scales with the span rather than being fixed. A fixed `G` is a defect
rather than an approximation: at `L = 1` the projection would carry as many
columns as the span has steps — the identity — so the contraction would be
absent, not weakened, while every assertion on the fan still passed. What `G_L`
holds roughly constant is the fraction of a span the basis can bend, not the
cutoff period: that is `2·L·PATCH_SIZE / G_L` steps, and it varies with `L`.

`BG_HEAD_MEDIAN_MODE` selects the assembly. `'global'` is the released default and
the one described above; `'cumulative'` continues each patch from the previous
patch's endpoint, an unconstrained integrator with no bound on the accumulated
offset, and `'independent'` applies the raw per-patch offset. Both alternatives
remain reachable and are kept for ablation.

**The spreads are monotone by construction.** Each passes through a softplus and
a floor of `BG_QUANTILE_SPREAD_MIN`, then accumulates outward from the median, so
the fan is strictly ordered with a guaranteed minimum gap and `q_tau[..., i]`
matches `QUANTILE_LEVELS[i]` index for index. The spread algebra is identical
under every median mode.

`carry_spread` seeds the accumulation on both sides, but no runtime caller passes
it — it defaults to zero everywhere. Because the assembly sits inside
`model.forward`, `predict_rolling` accumulates its own risk-space half-width and
applies the identical shift post-forward on the returned fan. Band width
therefore carries across roll boundaries and the envelope grows monotonically
rather than resetting at each seam.

### Time-of-day probe

A 2-layer SiLU MLP that classifies each masked patch's absolute hour into
`TIME_PROBE_N_BINS = round(24 / PREDICTION_HORIZON_HOURS)` circular bins — twelve
two-hour bins at the released defaults. It runs on every gathered slot, with no
pooling. There is no clock input, so the hour has to be inferred from the
trajectory. The bin count is a label-resolution choice, not a horizon: the
expression is kept because it tiles 24 h exactly, with the forecast protocol's
span entering as the reference length.

The target is a wrapped-Gaussian soft label over neighbouring bins, of width
`TIME_PROBE_LABEL_SMOOTH_BINS`, and the loss is cross-entropy against it. Slot
`j` is patch `mask_idx[:, j]`, so each slot's target hour follows `mask_idx`
rather than a fixed offset from the context edge. A second term couples
consecutive windows: `data.py` ships window `k + 1` — the same trajectory shifted
forward one horizon, teacher-forced, at the same context length — and a penalty
ties the two origin-phase estimates to exactly one horizon apart, measured in the
`(cos, sin)` plane so the gradient is stable across the midnight wrap. Within a
single window the slots share one forward pass and are already consistent, so no
within-window term is needed.

With `TIME_PROBE_DETACH = False`, the released setting, the probe's gradient
reaches the shared trunk. That is the point of it: the per-slot representations
the glucose head reads are pushed to encode circadian phase, which makes the
forecast time-aware. The forward *value* of `q_tau` and `median` is unaffected
either way — the probe never feeds them — and `L_tod` never enters the validation
loss or checkpoint selection.

Decoding softmaxes the bins, forms the resultant vector, and reads the hour from
its angle and a confidence `R ∈ [0, 1]` from its length. `R` near 1 is a sharp
belief, near 0 an ambiguous one. All of the circular geometry lives in `utils` —
the decode helpers in torch, the clock-face wedge geometry in pure numpy — so the
pygame and matplotlib renderers share one implementation.

`TIME_PROBE_ENABLED = False` leaves the head unbuilt and the forward
bit-identical to a model without it.


## Loss

`risk_loss.risk_total_loss(q_tau, median, true_bg_mgdl, weighting, valid,
mask_idx, d)`. The mg/dL target crosses into risk space exactly once, at the top,
and both terms share the result. `valid` and `mask_idx` are what make the
supervised set the masked patches rather than the head's slot axis; their
defaults describe a single dense right-edge span.

### Pinball

The check loss over every supervised step and every quantile level:

```
rho_τ(a, b) = (a − b) · (τ − 1[a < b])
L_Q         = mean over (valid slot, step, τ) of rho_τ(y_risk, q_tau)
```

It calibrates each level to its coverage. τ = 0.5 is retained deliberately as the
pointwise level anchor, dividing labour with the term below: pinball pins the
level, DILATE shapes the trajectory.

The reduction is per masked patch. A padded slot is removed from the
**denominator**, not merely zeroed in the numerator: zeroing alone still divides
by `B·M·S·Q` and rescales `L_Q` against `L_D` by the padded fraction, a level
`log_σ_Q` then absorbs while every loss curve looks unchanged.

### DILATE

Shape and time distortion on the median only, evaluated **once per masked span**.
Spans are bucketed by length `L`, each bucket stacked to `(n_b, L·PATCH_SIZE)` in
patch-major order so the time axis stays monotone, and one `dilate_loss` call
runs per bucket. Both sequences go in uncentred, so the gradient carries a level
component alongside the shape.

```
L_D = DILATE_ALPHA · shape + (1 − DILATE_ALPHA) · TDI
```

**Shape** is a soft-DTW *divergence*, `sDTW(m, y) − ½ sDTW(m, m) − ½ sDTW(y, y)`.
The self-terms are necessary because soft-DTW of a sequence with itself is not
zero; subtracting them makes the term and its gradient vanish at a perfect match.
The target self-term is a constant and is detached.

**TDI** penalises pulling the alignment off the diagonal. It equals the soft
alignment contracted with the off-diagonal distance `Ω[i,j] = ((i−j)/H)²`, which
is the directional derivative of the soft-DTW value along `Ω`. Evaluating it as a
one-sided finite difference costs one extra soft-DTW forward and rides on the
existing backward, so no alignment matrix is materialised and no second-order
autograd is involved. The median gradient is the exact TDI gradient to
`O(DILATE_TDI_FD_EPS)`.

Buckets combine by a **span-count-weighted mean of the per-bucket scalars**,
never by concatenation. DILATE is not scale-free in `H = L · PATCH_SIZE`: the
shape term grows with `H` while the normalised TDI does not track it, so `alpha`
weights a different mixture in each bucket and `log_σ_D` absorbs the difference
silently. The per-bucket `loss_D_L{L}` and the span-length histogram
`n_spans_L{L}` are logged alongside the combined value, so two runs are comparable
at an equal span-length mixture. An empty bucket is never dispatched: `dilate_loss`
reduces with `.mean()` and would return NaN, which survives every downstream
comparison and ends a run with no best checkpoint.

The dynamic program is a batched anti-diagonal recursion in `dilate.py`, with a
max-subtracted log-sum-exp softmin. `DILATE_GAMMA` is a softness knob, not an
overflow guard: a single cell peaks at
`(f(BG_CLAMP_MAX) − f(BG_CLAMP_MIN))² = 99.6416`, the stabilised softmin is
overflow-free in fp32 down to `γ = 1e-3`, and soft-DTW is 1-homogeneous in
`(cost, γ)`. Smaller γ approaches a hard minimum with a peakier gradient.

### Kendall-Gal weighting

The two terms are fused by learned homoscedastic-uncertainty weights rather than
a fixed ratio:

```
total = ½·exp(−2·log_σ_Q)·L_Q + log_σ_Q
      + ½·exp(−2·log_σ_D)·L_D + log_σ_D
```

Each precision-weighted term is paired with its `+ log_σ` regulariser, which is
what stops a weight collapsing to zero. The two scalars init at
`KENDALL_LOGVAR_INIT` and clamp to `[-7, 7]`.

They live on a small `KendallGalWeighting` module, deliberately **not** on the
model, which excludes them from the weight EMA structurally. They join AdamW in
their own group at `weight_decay = 0` — a log-variance must not decay toward
zero — and serialise under `weighting_state_dict`. Gradient clipping covers model
and weighting parameters together.

This is the entire training objective for the forecast. The time-probe loss is
added to the backward separately and never enters it.


## Optimisation

### Parameter split

Muon takes every parameter with `ndim ≥ 2`, the patch-embedding matrix included.
AdamW takes the 1-D parameters — norm gains and biases — plus the two
Kendall-Gal scalars in their own zero-decay group. Muon orthogonalises the
momentum buffer with a few Newton-Schulz iterations, which equalises an
anisotropic gradient spectrum; the operation is only meaningful for matrices,
which is why the 1-D parameters go elsewhere.

### Schedule-aware weight decay

A matrix feeding a normalization has its gradient orthogonal to its own weights,
so decoupled decay drives it to a steady state that scales as
`sqrt(2·λ / γ_t)`. Under a cosine schedule `γ_t → 0`, so that target diverges and
the gradient norm climbs over the final steps. AdamC (Defazio 2025) scales the
decay coefficient by `γ_t / γ_max`, pinning the steady state at a
schedule-independent value.

Since `γ_max` is the peak learning rate, the correction factor is exactly the
schedule ratio already computed for the LR, so `_update_lr` sets
`weight_decay = base × ratio` on the corrected group each step. At peak LR it is
bit-identical to plain decoupled decay, so the tuned regime is untouched; the
correction only softens decay as the LR decays.

Muon runs as two groups. The corrected one holds the trunk matrices and the head
*hidden* layers. The uncorrected one holds the two output projections
(`bg_head[-1]`, `time_head[-1]`), following the paper's exclusion of the output
layer; the AdamW group is uncorrected for the same reason. `--no-wd-correction`
restores constant decay throughout.

### Weight EMA

A shadow copy of the model's float state, updated after every accepted step as
`θ_ema ← decay·θ_ema + (1 − decay)·θ`, swapped in around validation and swapped
out again before training continues. Training always runs on live weights.

The motivation is narrow: threshold-crossing metrics like hypo recall are very
sensitive to small parameter shifts, and Muon's per-batch jitter was large enough
in that space that consecutive validations disagreed while the loss was falling
smoothly. Averaging over roughly a thousand steps removes that without touching
the training trajectory. `EMA_DECAY = 0` disables it.

### Non-finite steps

A non-finite intermediate must reach the training loop's guard, which wraps
forward, loss and backward together, rather than abort inside a deep assert.
Three sites cooperate: `dilate.py` keeps only shape assertions, so a non-finite
cost flows out; `kovatchev_f_inv` scrubs non-finite risk inputs before its clamp
so it can never emit a non-finite mg/dL; and the EMA skips blending non-finite
weights, so one transient NaN cannot poison the shadow permanently. The guard
then halves the optimizer state or restores from the EMA and continues.

### Checkpoints

`t1dmai_best.pt` is written whenever `val_loss_total` reaches a new minimum.
Periodic `t1dmai_step_{N}.pt` snapshots are independent of it. Each checkpoint
carries its architecture in `training_config` — including the three mask-sampler
constants — plus the normalization statistics, the EMA shadow, the
weighting module, and `arch_version` / `loss_schema` provenance tags. Clinical
metrics stay read-only diagnostics; there is no separate clinically-selected
checkpoint, because the risk-space loss geometry and the band-edge detectors
already carry the clinical bias structurally.


## Validation

Runs every `VALIDATION_INTERVAL` steps on a fixed held-out pool of
`VALIDATION_N_PATIENTS` patients, under EMA weights. Two artifacts: a row
appended to `logs/validation_log.csv`, and a table printed to stdout with columns
`Metric | Value | Prev`, coloured against published thresholds with a trend arrow
against the previous run. A section that loses every row loses its header.

A row of a per-horizon, per-region or per-`d` family whose bin came out empty
renders `—` and keeps its place; a standalone row whose value is unavailable is
omitted. An empty bin is a measurement, and neither way of hiding it is honest: a
vanished row reads as a metric nobody computes, and `0` reads as a rate of zero.

The CSV is the record and the table is a reading surface: every metric reaches
the CSV, while the table carries the families worth a glance at a 1000-step
cadence — the calibration section above all, since `val_loss_total` is the
objective on each sample's own mask and improves monotonically whether or not the
deployed one-sided band holds its stated level.

Because the model is always conditioned, there is one value per metric — no
second unconditional pass, and no `uncond_*` columns.

### The two protocols

Training places masks uniformly, so a metric averaged over the training mask
distribution is dominated by the easy regime and improves for free. Validation
therefore scores exactly two fixed protocols, defined once in
`metrics/protocols.py`:

| Protocol | Mask | Columns | Baseline |
| --- | --- | --- | --- |
| forecast | the trailing `PREDICTION_PATCHES` patches | the existing names, per horizon | persistence |
| infill | sampled interior spans | `infill_*`, per `d` | linear interpolation |

The forecast protocol supplies exactly one masked patch at each of `d = 1..4` per
window, which is what keeps its columns element-for-element comparable with the
historical tables and its per-`d` calibration populated. It is rebuilt as its own
forward: the trailing zone is masked and announced whatever the sample's own mask
was, and a row whose context-edge patch that mask already covered is **dropped**,
since the anchor would otherwise be read off a withheld reading. `fc_n` reports
how many rows survived; mask placement is independent of glucose, so the
survivors are an unbiased subsample.

Infill is scored against linear interpolation between the bracketing visible
readings and never against persistence: persistence is a forecasting baseline,
and against a two-sided span it is a strawman. Its interior spans are drawn at a
fixed seed, so a moving `infill_*` column is the model rather than the draw.

Both protocols report per `d`. Pooling over `d` is refused —
`metrics.protocols.column` will not name an infill column without one — because
the sampler concentrates supervision at small `d`, so a pooled masked-BG scalar
improves when the mixture softens and is not comparable between protocols. Every
pooled figure `metrics/scoring.py` emits carries that warning with it, and
nothing pooled is a selection metric.

`protocols.RunReport` bundles what every report run states about its own
conditions: the realised `d` histogram against the sampler's exactly-enumerated
one, so a departure means the sampler changed; the `n_ctx` the figures were
measured at; and the kept and dropped segment and window counts per cohort, so
two context widths cannot silently evaluate different window sets.

### Scoring rules

`metrics/scoring.py` holds five rules, each binned on `d`, each taking and
returning mg/dL:

- **CRPS**, a strictly proper score over the whole fan;
- **Winkler**, the interval score per nominal central level;
- **coverage with sharpness** — the width that bought the coverage is reported
  beside it, never apart, since any band widens to any coverage;
- **joint coverage**, every step of the path inside the band at once, labelled
  apart from the per-step marginal it is bounded above by;
- the **hypo alarm operating curve**, sweeping the band-edge τ and reporting
  detection rate, false alarms per day and **median lead time in minutes**
  together — a detection rate bought at a two-minute lead is not a usable alarm,
  and neither rate shows that alone.

The module defines no pass/fail threshold on any of them, and
`metrics.protocols.threshold` raises rather than return a guess.

### Table sections

| Section | What it carries |
| --- | --- |
| Training & internal losses | `val_loss_total` and `val_loss_Q`, the selection scalar and its pinball half |
| BG forecast (RMSE / MAE) | Single-pass horizons, 30 / 60 / 120 min, both error measures |
| BG forecast — night only @ 180+ | The rolled long horizons, 180 / 360 / 480 min, night-filtered per sample, each with the count it was scored over, plus the roll's mean context and the two short-window skip counts against their own denominators |
| Quantile calibration | Marginal 90 % band coverage and inner-50 % coverage with their widths, `sign_balance` at the far horizon, the one-sided against two-sided pair at `d = 1` — both arms `coverage_sharpness_by_d` over their own protocol's fan, so only sidedness separates them — and joint coverage of the whole path |
| Relative error & derivative tracking | MARD per horizon, the rate-of-change correlation, and median roughness pooled and at the far patch |
| Amplitude & excursion shape | The mean-collapse detectors: per-patch and per-excursion amplitude ratio, gain, correlation, and the overshoot/undershoot split |
| Conformal probe @ excursion peaks | Raw against region-binned coverage, each with the width that bought it, and both hypo-escape rates |
| Clinical error grid (Clarke) | All five zone shares, A+B, and all five again per horizon |
| Clinical error grid (DTS) | All five pooled zone shares and all five per horizon, uncoloured |
| Clinical accuracy (CG-EGA) | Per-region accurate, benign and erroneous fractions |
| Longitudinal excursions & TIR | Pooled hypo/hyper detection, time-in-range error, and the same detection per 30-minute bucket |
| Nocturnal | Nocturnal hypo AND hyper recall/precision |
| Night-onset excursion call | Per-NIGHT binary hypo and hyper recall/precision, with the nights scored, the nights skipped, and each side's true/called counts |
| Counterfactual dose-response | The two dose directions and insulin monotonicity (`train.py` only — the blind fork removes the probe) |
| Time-of-day probe | The whole probe: MAE pooled and at high confidence, the three accuracies, bias / std / p90 / gross-error rate, confidence, and the within- and cross-window jump witnesses |

A **pooled** grid share is over every scored step of the forecast protocol, so it
mixes a 5-minute-ahead error with a 2-hour-ahead one and moves when the horizon
mixture moves. A `@{h}` share is the single step at that horizon, which is what a
published grid figure can be read against. Both grids report both.

`validation_log.csv` carries every metric on every row, this table included and
the four families it does not print besides: the proper scoring rules per `d`, the
hypo alarm operating curve, the infill protocol's columns, and the nocturnal
duplicates of Clarke, MARD and CG-EGA (which repeat their all-sample counterparts
on a subset).

A row's colour band and its trend arrow are set separately, because on one row
they disagree. A marginal coverage has its nominal inside its band, so movement
toward the band's midpoint is improvement. Joint coverage does not: it is bounded
above by the smallest marginal in scope and rises toward that bound, so it is
trended as higher-is-better while still being coloured against the band where a
joint figure is acceptable.

The two point error grids are reported side by side and are not
interchangeable. Clarke's zones describe the treatment error a wrong reading
would cause; the DTS grid's are contours of a risk function elicited from 206
clinicians, asymmetric between over- and under-estimation, so the two disagree by
construction on the same pair. The DTS rows are uncoloured because no acceptance
threshold is published for that grid, and no zone A+B figure is derived: its
source states that presenting one is inappropriate and that zone A alone is the
measure.

Level metrics all read `pred_bg = f_inv(median)`, asserted inside the physical
range on entry. The exception is excursion detection, below.

The CSV is wider than the table. The pooled rolled horizons and the DILATE
shape/TDI split are computed and logged but not displayed; the two log-σ appear in
the per-step training log rather than the validation CSV.

### Band-edge detection

Hypo and hyper detection key off the **band edges**, not the median. The hypo
alarm reads the `HYPO_ALARM_QUANTILE_TAU` lower edge and fires when it drops
below `BG_HYPO_THRESHOLD`; the hyper alarm reads the `HYPER_ALARM_QUANTILE_TAU`
upper edge and fires when it rises above `BG_HYPER_THRESHOLD`. Both taus are
selected from `QUANTILE_LEVELS` by value, never by a bare index. Reading an
envelope rather than a point estimate is the clinically conservative call.

Recall is strict. Precision forgives a false alarm whose edge sits within
`EXCURSION_PRECISION_TOLERANCE_MGDL` of the true value, so sensor noise near a
threshold does not deflate it; those rows carry a `±k` suffix. Setting the
tolerance to zero recovers the plain confusion matrix.

Per-horizon targets decline with horizon, because near-term detection is largely
fixed by insulin on board while detection at two hours is information-limited. A
flat bar made every long-horizon cell read red regardless of quality. The
schedule lives in `train.py` as display-only constants; it colours rows and never
touches the loss, the CSV or checkpoint selection.

### Rate-of-change rows

`roc_corr`, `roc_rmse` and `trend_gain_beta` are computed on the **per-patch**
30-minute ΔBG, not the per-5-minute one, because the observed CGM's 5-minute
difference is dominated by sensor noise. `trend_amp_ratio`, the ratio of
predicted to true ΔBG standard deviation, isolates the amplitude axis that the
regression slope conflates with direction; it falls toward 0 when the forecast
collapses to a flat line. `bg_curve_corr` is anchor-relative — it correlates
`pred − anchor` against `true − anchor` — so it scores curve shape rather than
trivial level agreement.

The excursion-amplitude block is the peak-localised companion: it measures
forecast against true amplitude at each window's true peak, where damping is most
visible, along with over- and under-shoot fractions.

### Probes

**Counterfactual.** Perturbs the doses announced over a right-edge forecast span
against a baseline forecast and checks the response is physiologically right:
carbohydrate must raise glucose, insulin must lower it, a dose sweep must move it
monotonically, and the opposing dose must clear a baseline excursion. This is
what makes the always-on what-if path trustworthy. Diagnostic only.

**Conformal coverage.** Fits on a deterministic 60 % of the collected validation
windows and measures excursion-peak coverage on the disjoint 40 %, reporting raw
against calibrated with the mean band width that bought each. The fit is
region-binned, and the marginal fit is measured on the same windows in the same
call so the two fits are never compared across runs. The validation sample is
small, so the figures are directional; the deployable correction is fit after
training by `calibrate_conformal.py`.

**Night onset.** Answers the bedtime question — will there be an excursion before
morning? The forecast origin is forced to `NOCTURNAL_START_HOUR`, the forecast is
rolled across the whole night conditioned on the announced overnight doses, and a
per-night binary call is scored off the band edges: of the nights that truly go
low or high, how many are flagged, and of the flagged nights, how many truly do.


## Conformal calibration

The quantile bands are well-shaped but over-confident at excursions, and
asymmetrically so — realized glucose escapes the lower edge more than the upper.
Split-conformal recalibration corrects coverage after the fact without disturbing
the point forecast. The layer is pure numpy, runs entirely in mg/dL downstream of
the inverse transform, and never enters the loss or the model.
`SPEC/inference.md` §8.4 specifies the apply and the fit, including the side-aware
order statistic and the exchangeability requirement; this repository implements
them in `conformal.py` and does not restate them.

`conformal.py` fits one correction over the whole calibration set — a marginal
one, valid on average. `mondrian.py` re-fits it once per **region bin**, so
coverage is restored conditional on the regime rather than only on average. The
region is a property of the window, read from where the forecast is heading: the
median line's mean over the final patch. `REGION_EDGES` places its single edge in
the euglycaemic band and deliberately not at a clinical threshold, which would
split the windows that decide the alarm across two separately-fit corrections and
starve the low bin. Because the median is held fixed, a window's bin is the same
before and after correction, so the binning needs no second pass.

A bin whose calibration count falls below `MIN_N_OWN_FIT` takes the marginal
delta, and the fit records that it did: below that count the extreme level's own
offset **is** its most extreme residual, an estimate that moves with a single
outlier. Every bin reports its `n`, its distinct-patient count and its mean band
width beside its coverage.

Forecast and infill residuals are not exchangeable, so the split is by remit:
the forecast protocol's fit is what ships in `ckpt['conformal_delta']`, and
infill gets its own coarse fit, stored apart and marked `shipped = False`.

`calibrate_conformal.py` fits the shipped correction on the disjoint reserved
partition and stores it in the checkpoint; that one is valid for the simulator
path only. The real-data harness re-fits **per cohort** from each cohort's own
calibration split, and the export path ships none.


## Report metric basis

The real and simulator report suites score two bases, and they are not
interchangeable.

**Median line** — the point forecast `f_inv(median)`. This is what published
forecasters report, so it is the basis for every peer comparison, for the
fine-tuning harness's held-out summary, and for checkpoint-selection scalars.

**Band-scored** — the truth projected onto the inner band,
`clip(true, q[METRIC_BAND_TAU_LO], q[METRIC_BAND_TAU_HI])`. A level error becomes
the distance to the nearer edge, and zero wherever the truth lies inside the
band. This describes band geometry. It is not comparable to a point forecast: it
is bounded above by the median-line error and tends to zero as the band widens,
which also means it cannot order checkpoints.

Every emitted record names its basis. A degenerate band reproduces the
median-line numbers exactly. Persistence carries no band, so the skill baseline
stays a point forecast under both. Two band diagnostics sit beside the headline
block per horizon: realized coverage of the band, and its mean width in mg/dL.


## Normalization statistics

Four channels, one file — `normalization_stats.json`, a `{mean, std}` pair each
for `bg_absolute`, `carb_intake`, `insulin_combined` and `exercise_equiv`.
Glucose statistics live in **risk space** and the other three in **log1p space**,
so a consumer must apply the same forward transform before normalizing and the
same inverse after denormalizing. `bg_masked` has no entry, being a bit rather
than a signal.

The statistics can come from `python normalization.py`, which simulates a pool of
independent patients and accumulates with Welford's algorithm, or from
`T1DMSIM/cache_simulator.py`, which emits them beside `meta.json` when a cache is
built. Both produce the same four pairs in the same space.

The pool must match the distribution the model actually trains on — the same
skill mix and the same post-warmup window. Fitting on a different distribution
silently mis-scales every input. `normalization.py` draws seeds
`master_seed + 1_000_000 + i` while training hashes its patient seeds over the
full 63-bit space, so the two are not disjoint *ranges* — a collision is merely
astronomically unlikely, which is what keeps the fit clear of training samples.

The stored pairs are a function of the Kovatchev constants, of
`RISK_SPACE_CHANNELS` and of the membership of `SPARSE_LOG1P_CHANNELS`: change any
of the three and the file on disk describes a space the transforms no longer
produce.


## Simulator cache

The simulator is a Python loop and is the dominant per-batch cost, so on a fast
GPU it starves the device. `T1DMSIM/cache_simulator.py` pre-generates a pool of
post-warmup trajectories once; `T1DMDataset(cache_path=...)` then reads rows
instead of simulating. No pool ships with this repository; each is built locally,
and the build writes a `DATASET.md` into the pool directory recording that pool's
geometry, glycemic mix and per-channel storage. The two pools T1DMSIM publishes
are the earlier 666-step geometry, which the loader rejects.

**Layout.** One file per channel under `<out_dir>/`, plus a small raw `icr.npy`,
a `normalization_stats.json`, and a `meta.json` written last as the completion
sentinel. `meta.json` names the `cache_format`, which selects the reader:
`'blosc2-ndarray-v1'` for the compressed `.b2nd` layout, `'npy-memmap-v1'` for one
uncompressed `.npy` memmap per channel. Both carry the same fields and the same
per-row semantics. A cache with no `cache_format` key is rejected.

**Build.** Workers stage rows into per-channel `.npy` memmaps inside
`<out_dir>.partial/`, keeping parent RAM near zero; a single-threaded pass then
transcodes each channel to compressed `.b2nd` and deletes its staging file. The
transcode is deliberately separate from the fan-out, because out-of-order writes
into a compressed array would force a decompress–modify–recompress per row. The
directory is renamed into place only after every channel has flushed, so a
crashed build is never loadable.

**Validation on load.** `T1DMDataset.__init__` checks the cache format, the
channel list, the warmup hours, the simulated hours, `dt_minutes` and the
uniform-sample probability against the runtime config, and raises rather than
train on divergent data. The per-channel shape check against
`pool_size × n_timesteps` happens later, in `_load_cache`, on the first row read
inside a DataLoader worker — the open is deliberately lazy so open cache handles
are not pickled across the fork. The generation parameters under
`params` — hypoglycemia oversampling, the rail filter, the seed salt — are **not**
checked at all, so two pools with different glycemic mixes are both accepted.

**Compression.** Each `.b2nd` is chunked `(rows_per_chunk, T)` with byte-shuffle
and zstd. Byte-shuffle groups each float's high-entropy mantissa bytes apart from
its low-entropy exponent bytes, which compresses far better than raw IEEE-754
layout. On a million-row pool at the 1242-step geometry this runs about 1.3–1.7×
on the dense physiologic channels and 27–539× on the near-constant ones, for
roughly 2.4× over the pool as a whole; each pool's `DATASET.md` carries its own
measured per-channel ratios. Smaller chunks waste fewer decompressed bytes per
single-row read but give zstd a smaller window; at the default 32 rows per chunk a
chunk is `32 × n_timesteps × 4 B`, about 159 KB per channel at 1242 steps.

**Resident memory.** The two layouts are read differently, and only one of them is
mapped. Under blosc2 each channel is opened for ordinary file I/O and a read
decompresses one chunk per channel into a fresh array, so nothing accumulates in
the process; the compressed bytes stay in kernel page cache, shared between
DataLoader workers and reclaimed under pressure. Mapping a `.b2nd` instead makes
each touched chunk's compressed pages resident in every worker that read it, with
no way to release them — blosc2 exposes no mapping to `madvise` — so random access
over a large pool drives the resident set toward the pool's whole compressed
footprint.

Under `npy-memmap` the channel is mapped and a read faults *uncompressed* pages of
the touched row. Random access over a large pool touches ever-new rows, so the
resident set climbs toward the full on-disk footprint — on a unified-memory device
that presents as rising GPU memory and can starve the allocator.
`CACHE_MADVISE_DONTNEED` bounds it: each read copies its row out and issues
`madvise(MADV_DONTNEED)` on the pages it faulted. The reader also issues
`madvise(MADV_RANDOM)` at open, without which the kernel's 128 KB readahead pulls
far more than the row needs and leaves the remainder resident. The flag is a no-op
under blosc2 and on the fly.

**Reuse is benign.** The index maps to a row by `patient_seed % slab_size` within
the partition's disjoint slab, and each draw takes a fresh random window from
that row, so a reused row yields a different training window. Values are stored
float32, below the simulator's own noise floor. When the pool is smaller than the
number of samples the run will draw, the dataset prints a one-line reuse factor
at construction.


## Inference and export

All three inference modes live in `inference.py`, and `SPEC/inference.md` §9
specifies the recipes they implement.

**Standard.** `predict` runs one forward over one masked set. The default set is
the trailing `PREDICTION_PATCHES` patches — a forecast — and `mask_spans` names
any other: a span at patch 0 is a backcast, one between visible patches an
infill. The masked patches carry a zeroed glucose slot, an announced `bg_masked`
bit, and either the announced plan or the zero-dose baseline in the three plan
slots; the anchors and their patch indices are built by the same
`data._mask_slots` the training path uses. `predict` accepts an optional
conformal correction, applied to the bands with the median untouched.

The zero-dose baseline is `normalize(0)` per channel, **not** a literal `z = 0`.
A literal zero decodes through the sparse log1p inverse to a phantom dose or a
phantom exercise session, which would corrupt a no-dose forecast.

**What-if.** The same forward pass with different values in the three plan slots.
A baseline is simply another call.

**Rolling.** Re-feed is glucose-only: the median goes through the inverse
transform to mg/dL, then back through normalization into the new context patches'
glucose slot, whose `bg_masked` bit is cleared — those patches are visible now.
The plan slots come from the caller's callback, or the zero-dose baseline. The
context slides forward and drops its oldest patches at `MAX_CONTEXT_PATCHES`. The
accumulated band half-width carries into the next roll, so the envelope grows
monotonically rather than resetting at each seam. `normalization_stats` is
required; without it the re-normalization cannot be computed.

**Export.** The exported graph is the **right-edge specialisation** of the masked
objective, and `SPEC/inference.md` §3.1 documents it as such:
`exporters/modified_forward.py` reads the trailing `PREDICTION_PATCHES` patches as
a slice instead of gathering by `mask_idx`, takes its mask as an external additive
float struct at the fixed `T = MAX_SEQ_LEN` with `NEG_FILL = -30000.0`, and cuts
the graph at `head_raw` so everything downstream of it — the anchor, the assembly,
the decode — is the consumer's. `NEG_FILL` rather than `-inf` keeps an fp16 NPU
softmax finite, and underflows to the same zero in fp32.


## Parameter count

The count is computed from the architecture, never targeted. Per block it is
dominated by the FFN at `3 · D_MODEL · FFN_DIM` and attention at `4 · D_MODEL²`;
the norms are negligible. Multiply by `N_LAYERS` and add the patch embedding and
the two heads. `MAX_MASKED_PATCHES` sizes no weight — the head is applied per slot
with shared weights — and `step_basis` is a buffer, so neither enters the count.

`resize_model.py` with no flags instantiates the model on the `meta` device and
prints the current architecture and its exact count.
