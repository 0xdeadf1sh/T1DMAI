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

Every numeric dimension lives in `config.py`. Formulas here stay correct across a
`resize_model.py` resize; literal values do not, so read them from the config.


## Contents

- [Overview](#overview)
- [Dimensions](#dimensions)
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
- [Inference modes](#inference-modes)
- [Parameter count](#parameter-count)


## Overview

An encoder-only transformer. It reads 8–24 hours of observed history — CGM
glucose, logged carbohydrate, logged insulin — and emits the next
`PREDICTION_HORIZON_HOURS` as seven blood-glucose quantiles per 5-minute step.
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

The constants are re-anchored to the `[40, 400]` mg/dL device range, which puts
zero risk near 128 mg/dL. `SPEC/invariants.md` §4 governs them and distinguishes
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
| `MAX_CONTEXT_PATCHES` | `--max-context-patches` | Longest context, and the left-pad width |

Derived, not settable:

| Constant | Definition |
| --- | --- |
| `N_INPUT_FEATURES` | 3 — `[bg_absolute, carb_intake, insulin_combined]` |
| `PATCH_DIM` | `PATCH_SIZE × N_INPUT_FEATURES` |
| `PREDICTION_PATCHES` | `PREDICTION_HORIZON_HOURS × 60 / (PATCH_SIZE × 5)` |
| `MAX_SEQ_LEN` | `MAX_CONTEXT_PATCHES + PREDICTION_PATCHES` |
| `N_QUANTILES` | 7 — `QUANTILE_LEVELS = (.05, .1, .25, .5, .75, .9, .95)` |
| `N_SPREADS` | 3 — spreads per side; the head emits `1 + 2·N_SPREADS` values per step |

At the released defaults a patch is 30 minutes, the context runs 16–48 patches
(8–24 h), and the horizon is 4 patches (2 h). The 8-hour floor follows the
autocorrelation analysis in `config.py`; the 24-hour ceiling covers one full
basal cycle and leaves enough real context for the 8-hour nocturnal roll.


## Inputs

Three features per 5-minute step. There are no time-of-day features and no mask
bits — day and night are inferred from the trajectory alone.

| Feature | Units | Transform before the model |
| --- | --- | --- |
| `bg_absolute` | mg/dL | Clamp to `[BG_CLAMP_MIN, BG_CLAMP_MAX]`, Kovatchev `f`, z-score |
| `carb_intake` | g / step | Floor at 0, `log1p`, z-score |
| `insulin_combined` | U / step | Floor at 0, `log1p`, z-score |

`log1p` is near-linear near zero, so the dense basal baseline passes through
almost unchanged while rare meal and bolus spikes are compressed out of the
channel's standard deviation. `normalization.py` holds the membership sets —
`RISK_SPACE_CHANNELS` for glucose, `SPARSE_LOG1P_CHANNELS` for the other two —
and every forward and inverse transform consults them, so the pipeline stays
invertible.

All three are the **raw post-noise** simulator signals. There is no smoother
anywhere, on the inputs or the target. The same raw glucose is the model input,
the forecast target and the anchor, so there is no input/target asymmetry, and a
live CGM stream needs no on-device filter to reproduce.

Insulin sensitivity, hepatic glucose output and exercise stay in the cache but
never reach the model. They are internal states a real CGM cannot observe, so
withholding them forces the model to forecast from signals deployment can supply.

`carb_intake` and `insulin_combined` are **absorption and action rates**, not
ingestion and injection instants: grams entering the blood and units acting in each
5-minute step. Pretraining takes them from the simulator directly. Real records
rarely store them, so `realdata/features.py` reconstructs them by convolving logged
amounts with population-mean kernels, with the fidelity limits the README sets out.
A record whose events already carry their resolved series instead supplies them on
`Segment.carb_curve` / `Segment.insulin_curve`, which bypass the kernels; the
transforms in the table above are unchanged either way.

### The index map

Context patches carry all three features. In the prediction zone:

- `bg_absolute` is **always zeroed** — it is what the model predicts.
- `carb_intake` and `insulin_combined` **always carry the future carbohydrate-
  appearance and insulin-action curves, per 5-minute step** — not the moment of
  eating, not the injection instant, and not a delivery schedule
  (`SPEC/invariants.md` §5): the true values during training, the caller's
  announcement at inference.

The model is therefore always conditioned on a declared plan. There is no masked
regime and no conditioned/unconditioned split, which is why what-if forecasting
is a property of the forward pass rather than a separate mode — announcing a
different plan just writes different values into those two slots.

`CHANNEL_TO_FEAT = {0: 1, 1: 2}` is the single mapping from an announceable
output channel to its input slot. Both the data pipeline and the inference
override path read it, so no second offset literal exists.

### The anchor

`last_bg` is the last context glucose in mg/dL, at `pred_start − 1`. Training and
inference reach it by two paths that agree in value but not in mechanism:
training reads the raw mg/dL trajectory directly in `data.py`, while inference and
export reconstruct the same physical value from the rightmost context cell via
`utils.last_bg_mgdl_from_context` — un-z-score, then the inverse risk transform —
matching to a sub-ulp round-trip difference. Neither path indexes at or past the
prediction origin, so the anchor cannot leak the horizon's trend.


## Encoder

### Patch embedding

`PATCH_SIZE` consecutive timesteps become one token: `PATCH_DIM` raw values
through a single `Linear(PATCH_DIM, D_MODEL)` with bias. The bias matters because
the prediction zone's glucose slot is a hard zero, and a learned offset lets the
model tell that apart from a genuine reading of zero.

There is no patient embedding. `forward` takes no patient argument; identity is
whatever the context window implies.

### Position

Two signals, both relative:

- **RoPE** on Q and K, at base frequency `ROPE_BASE`. The cosine and sine tables
  depend only on sequence length and head dimension, so they are built once per
  forward pass and shared across every block.
- **ALiBi**, a per-head learnable slope added to the logits as `−|i − j| · |s_h|`.
  Slopes initialise to the geometric series `2^(−8(h+1)/N_HEADS)`, giving each
  head a different reach — sharply local at one end, nearly flat at the other.
  The `.abs()` keeps the bias a penalty; a negative slope would reward distance.

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

Four regions, not a causal mask:

| | to context | to horizon |
| --- | --- | --- |
| **from context** | attend | **blocked** |
| **from horizon** | attend | attend |

The horizon is decoded jointly, so horizon tokens attend to each other in both
directions. The one blocked region is what prevents a future leak: nothing the
model must predict can influence the context representation.

Shorter contexts are left-padded to `MAX_CONTEXT_PATCHES`. Padding rows and
columns are blocked outright; the collate function then forces the attention
diagonal True at every position, padding included, so a padding row attends only
to itself and its softmax is never fully masked. Padding outputs are never read —
the heads slice the last `PREDICTION_PATCHES` tokens, and padding sits at the far
left.


## Heads

Both heads read the last `PREDICTION_PATCHES` tokens after the final norm — one
`D_MODEL` vector per horizon patch.

### Blood-glucose quantile head

A 3-layer SiLU MLP emitting `BG_HEAD_STEP_BASIS_DIM` coefficients per channel per
patch, where a channel is the median offset or one of the `2 · N_SPREADS`
spreads. A fixed orthonormal basis then expands those coefficients across the
patch's timesteps:

```
coeff    = bg_head(pred).view(B, P, K, 1 + 2·N_SPREADS)
head_raw = einsum('sk,bpkc->bpsc', step_basis, coeff)     # (B, P, S, C)
```

`step_basis` is `(PATCH_SIZE, K)` — the low-frequency DCT-II modes, or orthonormal
polynomials under `BG_HEAD_STEP_BASIS_TYPE = 'poly'`. Emitting fewer coefficients
than the patch has steps makes the highest-frequency within-patch mode
unrepresentable, so the head **cannot** produce an intra-patch zigzag. Setting
`K = PATCH_SIZE` recovers a fully free per-step head.

The final layer initialises at `std = BG_HEAD_INIT_SCALE`, so at step 0 the
coefficients are near zero, the median offset is near zero, and the forecast is a
flat persistence line at the last observed glucose.

### Quantile assembly

`utils.assemble_quantiles` turns `head_raw` into the ascending fan. It is the
single chokepoint — training, `predict` and `predict_rolling` all pass through
it, so a config change propagates identically everywhere. The exact algebra is in
`SPEC/inference.md` §8.1; what matters here is why it has the shape it does.

**The median is projected, not integrated.** The per-patch median offsets are
flattened patch-major over the whole horizon and projected onto a fixed
low-frequency DCT-II subspace of `BG_HEAD_MEDIAN_GLOBAL_DIM` columns. A projection
is an L2 contraction, so the offset is bounded and cannot accumulate across
patches. That is the point: an earlier design carried each patch forward from the
previous patch's endpoint, which is an unconstrained integrator, and it drifted —
overshooting a single pass and biasing the level over an 8-hour roll. The
low-pass also removes the seam sawtooth, so no separate seam penalty is needed.
Column 0 is the constant mode, which preserves the persistence level. Exact
seam continuity is deliberately not enforced; enforcing it is what caused the
drift.

`BG_HEAD_MEDIAN_MODE` selects the assembly. `'global'` is the released default and
the one described above; `'cumulative'` and `'independent'` are kept only for
ablation.

**The spreads are monotone by construction.** Each passes through a softplus and
a floor of `BG_QUANTILE_SPREAD_MIN`, then accumulates outward from the median, so
the fan is strictly ordered with a guaranteed minimum gap and `q_tau[..., i]`
matches `QUANTILE_LEVELS[i]` index for index.

`carry_spread` seeds the accumulation on both sides, but no runtime caller passes
it — it defaults to zero everywhere. Because the assembly sits inside
`model.forward`, `predict_rolling` accumulates its own risk-space half-width and
applies the identical shift post-forward on the returned fan. Band width
therefore carries across roll boundaries and the envelope grows monotonically
rather than resetting at each seam.

### Time-of-day probe

A 2-layer SiLU MLP that classifies each horizon patch's absolute hour into
`TIME_PROBE_N_BINS = round(24 / PREDICTION_HORIZON_HOURS)` circular bins — twelve
two-hour bins at the released horizon. It runs on every horizon patch, with no
pooling. There is no clock input, so the hour has to be inferred from the
trajectory.

The target is a wrapped-Gaussian soft label over neighbouring bins, of width
`TIME_PROBE_LABEL_SMOOTH_BINS`, and the loss is cross-entropy against it. A second
term couples consecutive windows: `data.py` ships window `k + 1` — the same
trajectory shifted forward one horizon, teacher-forced, at the same context
length so the mask is reused — and a penalty ties the two origin-phase estimates
to exactly one horizon apart, measured in the `(cos, sin)` plane so the gradient
is stable across the midnight wrap. Within a single window the patches share one
forward pass and are already consistent, so no within-window term is needed.

With `TIME_PROBE_DETACH = False`, the released setting, the probe's gradient
reaches the shared trunk. That is the point of it: the per-patch representations
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

`risk_loss.risk_total_loss(q_tau, median, true_bg_mgdl, weighting)`. The mg/dL
target crosses into risk space exactly once, at the top, and both terms share the
result.

### Pinball

The check loss over every horizon step and every quantile level:

```
rho_τ(a, b) = (a − b) · (τ − 1[a < b])
L_Q         = mean over (step, τ) of rho_τ(y_risk, q_tau)
```

It calibrates each level to its coverage. τ = 0.5 is retained deliberately as the
pointwise level anchor, dividing labour with the term below: pinball pins the
level, DILATE shapes the trajectory.

### DILATE

Shape and time distortion on the median only, after reshaping median and target
to `(B, P·S)` in patch-major order so the time axis stays monotone. Both go in
uncentred, so the gradient carries a level component alongside the shape.

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

The dynamic program is a batched anti-diagonal recursion in `dilate.py`, with a
max-subtracted log-sum-exp softmin. `DILATE_GAMMA` is a softness knob, not an
overflow guard: a single cell peaks at `(f(400) − f(40))² = 40`, the stabilised
softmin is overflow-free in fp32 down to `γ = 1e-3`, and soft-DTW is
1-homogeneous in `(cost, γ)`. Smaller γ approaches a hard minimum with a peakier
gradient.

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
AdamW takes the 1-D parameters — norm gains, biases, ALiBi slopes — plus the two
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
carries its architecture in `training_config`, the normalization statistics, the
EMA shadow, the weighting module, and `arch_version` / `loss_schema` provenance
tags. Clinical metrics stay read-only diagnostics; there is no separate
clinically-selected checkpoint, because the risk-space loss geometry and the
band-edge detectors already carry the clinical bias structurally.


## Validation

Runs every `VALIDATION_INTERVAL` steps on a fixed held-out patient pool, under
EMA weights. Two artifacts: a row appended to `logs/validation_log.csv`, and a
table printed to stdout with columns `Metric | Value | Prev | Target`, coloured
against published thresholds with a trend arrow against the previous run. Rows
whose value is unavailable are omitted, and a section that loses every row loses
its header.

Because the model is always conditioned, there is one value per metric — no
second unconditional pass, and no `uncond_*` columns.

| Section | What it carries |
| --- | --- |
| Training & internal losses | `val_loss_total`, the pinball and DILATE terms, the two log-σ |
| BG forecast (RMSE) | Single-pass horizons, 30 / 60 / 120 min |
| BG forecast — night only @ 180+ | The rolled long horizons, 180 / 360 / 480 min, night-filtered per sample |
| Quantile calibration | Marginal 90 % band coverage, inner-50 % coverage, `sign_balance` |
| Relative error & derivative tracking | MARD, rate-of-change and trend rows, excursion amplitude, the conformal probe |
| Clinical error grid (Clarke) | Per-horizon zone A, plus pooled A+B and D |
| Clinical accuracy (CG-EGA) | Per-region accurate and erroneous fractions |
| Longitudinal excursions & TIR | Pooled hypo/hyper detection, time in range |
| Excursions by horizon | The same detection over disjoint 30-minute buckets |
| Nocturnal | Night-only metrics, and the per-night onset call |
| Counterfactual dose-response | Sign, monotonicity and rescue probes |
| Time-of-day probe | Accuracy, reliability and the two no-jump witnesses |

Level metrics all read `pred_bg = f_inv(median)`, asserted inside the physical
range on entry. The exception is excursion detection, below.

The CSV is wider than the table. The pooled rolled horizons, Clarke zone E and the
DILATE shape/TDI split are computed and logged but not displayed; the two log-σ
appear in the table and in the per-step training log rather than the validation
CSV.

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
`pred − last_bg` against `true − last_bg` — so it scores curve shape rather than
trivial level agreement.

The excursion-amplitude block is the peak-localised companion: it measures
forecast against true amplitude at each window's true peak, where damping is most
visible, along with over- and under-shoot fractions.

### Probes

**Counterfactual.** Perturbs the prediction-zone doses against a baseline
forecast and checks the response is physiologically right: carbohydrate must
raise glucose, insulin must lower it, a dose sweep must move it monotonically,
and a baseline excursion should clear once the opposing dose is added. This is
what makes the always-on what-if path trustworthy. Diagnostic only.

**Conformal coverage.** Fits per-step, per-level corrections on a deterministic
60 % of the collected validation windows and measures excursion-peak coverage on
the disjoint 40 %, reporting raw against calibrated. It is a witness that
recalibration restores the coverage the raw fan loses at excursions. The
validation sample is small, so read it directionally; the deployable correction
is fit after training by `calibrate_conformal.py`.

**Night onset.** Answers the bedtime question — will there be an excursion before
morning? The prediction origin is forced to `NOCTURNAL_START_HOUR`, the forecast
is rolled across the whole night conditioned on the announced overnight doses,
and a per-night binary call is scored off the band edges: of the nights that
truly go low or high, how many are flagged, and of the flagged nights, how many
truly do.


## Conformal calibration

The quantile bands are well-shaped but over-confident at excursions, and
asymmetrically so — realized glucose escapes the lower edge more than the upper.
Split-conformal recalibration corrects coverage after the fact without disturbing
the point forecast. The layer is pure numpy in `conformal.py` and runs entirely
in mg/dL, downstream of the inverse transform, never inside the loss or the model.

`fit_quantile_conformal` fits a per-step, per-level additive correction from
calibration residuals. Because each level is fitted from its own residuals, the
correction is asymmetric by construction, which is what lets it correct the hypo
side harder. The order statistic is side-aware: an upper edge uses
`ceil((n+1)·τ)`, a lower edge `floor((n+1)·τ)`. Using `ceil` on a lower edge sits
it one order statistic too high and lets glucose escape below it more than
nominal — anti-conservative on exactly the clinically load-bearing edge, and
worst at small calibration sets.

`apply_quantile_conformal` enforces three invariants, each unit-tested: the
median column is untouched, the fan stays monotone outward from it, and an
all-zero correction is the identity.

Validity rests on calibration and test being exchangeable, so a correction is
only valid for the distribution it was fitted on. `calibrate_conformal.py` fits
the simulator correction on the disjoint reserved partition and stores it in the
checkpoint; that one is valid for the simulator path only. The real-data harness
re-fits **per cohort** from each cohort's own calibration split.


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

Three channels, one file — `normalization_stats.json`, a `{mean, std}` pair each
for `bg_absolute`, `carb_intake` and `insulin_combined`. Glucose statistics live
in **risk space** and the other two in **log1p space**, so a consumer must apply
the same forward transform before normalizing and the same inverse after
denormalizing.

The statistics can come from `python normalization.py`, which simulates a pool of
independent patients and accumulates with Welford's algorithm, or from
`T1DMSIM/cache_simulator.py`, which emits them beside `meta.json` when a cache is
built. Both produce the same three pairs in the same space.

The pool must match the distribution the model actually trains on — the same
skill mix and the same post-warmup window. Fitting on a different distribution
silently mis-scales every input. `normalization.py` draws seeds
`master_seed + 1_000_000 + i` while training hashes its patient seeds over the
full 63-bit space, so the two are not disjoint *ranges* — a collision is merely
astronomically unlikely, which is what keeps the fit clear of training samples.

Regenerate the file after any change to the Kovatchev constants, to
`RISK_SPACE_CHANNELS`, or to the membership of `SPARSE_LOG1P_CHANNELS`.


## Simulator cache

The simulator is a Python loop and is the dominant per-batch cost, so on a fast
GPU it starves the device. `T1DMSIM/cache_simulator.py` pre-generates a pool of
post-warmup trajectories once; `T1DMDataset(cache_path=...)` then reads rows
instead of simulating. The README lists the published pools.

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
inside a DataLoader worker — the open is deliberately lazy so memory-mapped
handles are not pickled across the fork. The generation parameters under
`params` — hypoglycemia oversampling, the rail filter, the seed salt — are **not**
checked at all, so two pools with different glycemic mixes are both accepted.

**Compression.** Each `.b2nd` is chunked `(rows_per_chunk, T)` with byte-shuffle
and zstd. Byte-shuffle groups each float's high-entropy mantissa bytes apart from
its low-entropy exponent bytes, which compresses far better than raw IEEE-754
layout. On the published pools this runs about 1.3–1.6× on the dense physiologic
channels and 17–633× on the near-constant ones — exercise about 19×, hour-of-day
about 86×, the day index about 633× — for roughly 2.3× over the pool as a whole.
Smaller chunks waste fewer decompressed bytes per single-row read but give zstd
a smaller window; the default keeps a chunk near 85 KB per channel.

**Resident memory.** Both layouts open memory-mapped, so DataLoader workers share
the kernel page cache. They differ sharply in what stays resident. Under blosc2
the shared pages hold *compressed* bytes and each read decompresses one chunk per
channel into a fresh array, so the resident set is bounded by the compressed
footprint. Under `npy-memmap` a read faults *uncompressed* pages of the touched
row, and random access over a large pool touches ever-new rows, so the page cache
climbs toward the full on-disk footprint — on a unified-memory device that
presents as rising GPU memory and can starve the allocator. `CACHE_MADVISE_DONTNEED`
bounds it: each read copies its row out and issues `madvise(MADV_DONTNEED)` on the
pages it faulted. The reader also issues `madvise(MADV_RANDOM)` at open, without
which the kernel's 128 KB readahead pulls far more than the row needs and leaves
the remainder resident. The flag is a no-op under blosc2 and on the fly.

**Reuse is benign.** The index maps to a row by `patient_seed % slab_size` within
the partition's disjoint slab, and each draw takes a fresh random window from
that row, so a reused row yields a different training window. Values are stored
float32, below the simulator's own noise floor. When the pool is smaller than the
number of samples the run will draw, the dataset prints a one-line reuse factor
at construction.


## Inference modes

All three live in `inference.py`. The decode recipe is specified in
`SPEC/inference.md` §9.

**Standard.** Fill the context with observed history; zero the prediction zone's
glucose slot and fill its dose slots with the announced future, or with the
zero-dose baseline if none is announced; recover `last_bg`; one forward pass;
invert the median and the fan to mg/dL. `predict` accepts an optional conformal
correction, applied to the bands with the median untouched.

The zero-dose baseline is `normalize(0)` per channel, **not** a literal `z = 0`.
A literal zero decodes through the sparse log1p inverse to a phantom dose, which
would corrupt a no-dose forecast.

**What-if.** The same forward pass with different values in the two dose slots. A
baseline is simply another call.

**Rolling.** Re-feed is glucose-only: the median goes through the inverse
transform to mg/dL, then back through normalization into the new context patches'
glucose slot. Dose slots come from the caller's callback, or the zero-dose
baseline. The context slides forward and drops its oldest patches at
`MAX_CONTEXT_PATCHES`. The accumulated band half-width carries into the next
roll, so the envelope grows monotonically rather than resetting at each seam.
`normalization_stats` is required; without it the re-normalization cannot be
computed.


## Parameter count

The count is computed from the architecture, never targeted. Per block it is
dominated by the FFN at `3 · D_MODEL · FFN_DIM` and attention at `4 · D_MODEL²`;
norms and ALiBi slopes are negligible. Multiply by `N_LAYERS` and add the patch
embedding and the two heads.

`resize_model.py` with no flags instantiates the model on the `meta` device and
prints the current architecture and its exact count.
