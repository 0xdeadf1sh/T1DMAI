"""All tunable constants. CLI flags override these; train.py prints the resolved set."""

# Owned by T1DMSIM, never restated here; ``import config`` needs the symlink.
# SIMULATOR_WARMUP_HOURS: hours dropped so trajectory hour 0 sits at basal steady state.
from T1DMSIM.simulator import (
    SIMULATOR_WARMUP_HOURS,
    EXERCISE_CARB_EQUIV_PER_MIN as _EX_G_PER_MIN,
    EXERCISE_DURATION_MEAN_MIN as _EX_DUR_MEAN_MIN,
)

# resize_model.py rewrites these, preserving HEAD_DIM = D_MODEL // N_HEADS, the symbolic
# FFN_DIM / BG_HEAD_HIDDEN = k * D_MODEL and HEAD_DIM in {16, 32, 64, 128} — so don't bake
# their values into other code. PATCH_SIZE must divide 12, and MIN/MAX_CONTEXT_PATCHES
# count patches, so changing it rescales their wall-clock span.
D_MODEL = 32
N_LAYERS = 32
N_HEADS = 1
HEAD_DIM = D_MODEL // N_HEADS
FFN_DIM = 1 * D_MODEL
PATCH_SIZE = 6                   # 6 × 5 min = 30 min per patch

# Feature order (FROZEN), step-major within a patch: 0 bg_absolute as z(f(bg)) in risk
# space | 1 carb | 2 insulin | 3 exercise_equiv in g/step | 4 bg_masked bit. Feats 1-3 are
# plan channels, always carrying announced values, masked patches included.
N_INPUT_FEATURES = 5
# feat 4 carries no normalization statistics and must stay inside the step-major block:
# PATCH_DIM and the [:, f::N_INPUT_FEATURES] stride both depend on it. The masked set is
# ANNOUNCED, not inferred — z = 0 in a withheld bg slot decodes to an ordinary reading.
PATCH_DIM = PATCH_SIZE * N_INPUT_FEATURES

# Output-channel space {0: carb, 1: insulin, 2: exercise}. feat 4 is in neither tuple: it
# is written from the sampled mask rather than from a signal.
NON_MASKABLE_FEATS = (0,)        # bg input slot zeroed at every masked patch
MASKABLE_FEATS = (1, 2, 3)
# The single output-channel -> input-feat mapping; data.py and inference.py both read it.
CHANNEL_TO_FEAT = {0: 1, 1: 2, 2: 3}

# Context 84–168 h. The floor sits far above the simulator's 5.3 h pooled-CGM ACF₀.₂
# (T1DMSIM/diff/README.md §0.5), spans several 24 h basal dose cycles, and leaves GT
# context through the 8 h night rolling validation. Each sample draws n_ctx uniform in
# [MIN, MAX]; collate_fn left-pads to the batch maximum.
MAX_CONTEXT_PATCHES = 336        # patches, not hours: hours = / _PATCHES_PER_HOUR
MIN_CONTEXT_PATCHES = 168

# Span of the fixed FORECAST protocol: a dense right-edge run of PREDICTION_PATCHES masked
# patches at any patch-aligned position in the day. Also the reference length the per-span
# time probe scales against. Time of day is never an input feature.
PREDICTION_HORIZON_HOURS = 2
_PATCHES_PER_HOUR = 60 // (PATCH_SIZE * 5)
PREDICTION_PATCHES = PREDICTION_HORIZON_HOURS * _PATCHES_PER_HOUR
MAX_SEQ_LEN = MAX_CONTEXT_PATCHES + PREDICTION_PATCHES

# Sampler (data.sample_mask_spans), per sample: n_spans ~ U{1..MASK_MAX_SPANS}, each
# L_i ~ U(MASK_SPAN_LENGTHS); sum(L) > MAX_MASKED_PATCHES resamples the WHOLE length
# vector, since per-element redrawing gives a different length law and so a different d
# histogram. Placement is stars-and-bars over the n_spans + 1 gaps, with one mandatory
# visible separator: two abutting spans are one longer span, and the separator is what
# makes the anchor, the spline's node sequence and the DILATE bucket well defined.
MASK_MAX_SPANS = 3
MASK_SPAN_LENGTHS = (1, 2, 3, 4, 5, 6, 7, 8)
# Sampler cap on sum(L), and M, the head's slot count. Surplus slots pad and gather patch
# 0, so every loss and metric path must discard them by their ``valid`` flag.
MAX_MASKED_PATCHES = 12
# Share of windows whose LAST span is pinned flush right. At 0 the deployed right-edge
# case is ~3% of windows and its 90% band decays with training: cov90@30 is 0.9149 here
# against the control's 0.8805 at step 12000, falling to 0.8547 by 100k. Plateau starts at
# 0.15 (0.9053) and 0.60 buys nothing (0.9087) while interpolation accuracy pays
# monotonically. Moving this invalidates metrics.protocols.SAMPLER_REFERENCE, which reads it.
MASK_RIGHT_EDGE_QUOTA = 0.50

# Validation rolls predict_rolling out to this on nocturnal samples so the table can report
# bg_rmse past a single forward. Equal to PREDICTION_HORIZON_HOURS skips rolling.
NIGHT_LONG_HORIZON_HOURS = 8
NIGHT_LONG_HORIZON_PATCHES = NIGHT_LONG_HORIZON_HOURS * _PATCHES_PER_HOUR

# q_tau[..., i] is QUANTILE_LEVELS[i] index for index. The head's raw width is
# 1 + 2*N_SPREADS: col 0 = median delta, cols 1..3 = the τ>.5 spreads .75/.9/.95
# nearest->far, cols 4..6 = the τ<.5 spreads .25/.1/.05. Quantiles are assembled around
# each slot's OWN anchor f(anchor_bg) — see utils.assemble_quantiles.
QUANTILE_LEVELS = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)   # ascending τ
N_QUANTILES = 7
N_SPREADS = 3                    # spreads per side of the median
BG_HEAD_HIDDEN = 1 * D_MODEL     # resize_model.py --bg-head-hidden-mult
BG_HEAD_INIT_SCALE = 1e-2        # small ⇒ median ≈ f(anchor_bg) at init ⇒ starts at persistence
BG_QUANTILE_SPREAD_MIN = 1e-3    # additive floor per softplus spread; anti σ-collapse

# Per-slot circular hour-of-day classifier over every gathered masked-slot hidden state (no
# mean-pool), trained by cross-entropy against a wrapped-Gaussian soft label. With
# TIME_PROBE_DETACH False the hidden states stay attached, so the probe shapes the shared
# trunk; the forward VALUE of q_tau/median is the same either way. Its loss enters the
# TRAINING backward only — never risk_total_loss, val_loss_total or checkpoint selection.
TIME_PROBE_ENABLED = True        # False ⇒ head not built
TIME_PROBE_HIDDEN = 1 * D_MODEL  # 2-layer SiLU probe MLP
TIME_PROBE_DETACH = False        # True ⇒ read-only diagnostic, no trunk gradient
TIME_PROBE_LOSS_WEIGHT = 1.0     # scales the probe CE + cross-window term, backward only
TIME_PROBE_INIT_SCALE = 1e-2     # probe final-layer weight init std
# A label-resolution choice for circadian phase, not a consequence of any horizon: the
# expression is kept because it tiles 24 h exactly with PREDICTION_HORIZON_HOURS entering
# as the reference span.
TIME_PROBE_N_BINS = max(1, round(24.0 / PREDICTION_HORIZON_HOURS))   # 12 bins, 2 h each
TIME_PROBE_BIN_HOURS = 24.0 / TIME_PROBE_N_BINS                      # exact tiling of 24 h
TIME_PROBE_LABEL_SMOOTH_BINS = 0.75                                  # soft-label std in bins; <=0 ⇒ one-hot
# Teacher-forced penalty coupling consecutive INDEPENDENT-forward windows so the rolling
# clock advances by one horizon across each seam; 0.0 skips the 2nd forward entirely. There
# is no within-window term: masked patches share one forward and are already consistent
# (tod_jump_h ≈ 0.12 h either way).
TIME_PROBE_CROSS_WINDOW_WEIGHT = 1.0
TIME_PROBE_CROSS_WINDOW_FRACTION = 1.0  # 2nd forward on the first ceil(frac*B) rows; validation always uses all

ROPE_BASE = 1000

MASTER_SEED = 42
DETERMINISTIC = False            # True: TF32 off, cuDNN deterministic; SDPA backward still not bit-exact
TOTAL_STEPS = 10000
BATCH_SIZE = 64
NUM_WORKERS = 8

# npy-memmap cache only. Each random row read faults pages that never repeat, so the page
# cache climbs until it eats free RAM (on unified-memory GPUs it presents as rising VRAM).
# T1DMDataset copies each row out then madvise(MADV_DONTNEED)s the range — which alone does
# NOT bound growth, since the 128 KB per-fault readahead (~0.5 GB/step) outlives the row's
# ~2 pages — so it also madvise(MADV_RANDOM)s each channel mapping at open; together they
# drop growth ~160×. blosc2 has the same failure and no cure (it exposes no mapping to
# madvise), so _load_cache reads a .b2nd through ordinary file I/O instead.
CACHE_MADVISE_DONTNEED = True
WARMUP_STEPS = 2000              # steps of linear LR warmup
LR_MIN_RATIO = 0.001             # cosine decay floor as a fraction of peak LR

# Must stay 0.0: every cache T1DMSIM writes carries patient_uniform_sample_prob = 0.0 and
# the loader equality-checks this against it. Above 0, a sample's patient would be drawn
# with skills uniform over [SKILL_MIN, SKILL_MAX] instead of the normal sampler's.
PATIENT_UNIFORM_SAMPLE_PROB = 0.0

MUON_LR = 0.02                   # 2D weight matrices
MUON_MOMENTUM = 0.95
MUON_NS_ITERATIONS = 5           # quintic Newton-Schulz steps
MUON_WEIGHT_DECAY = 0.05         # decoupled
ADAM_LR = 0.003                  # embeddings and 1D parameters
ADAM_BETAS = (0.9, 0.95)
ADAM_WEIGHT_DECAY = 0.05
ADAM_EPS = 1e-8

# AdamC (arXiv 2506.02285): a weight matrix feeding a norm has its gradient orthogonal to
# its weights, so decoupled decay drives it to ||g||/||x|| = sqrt(2*lambda/gamma_t), which
# diverges as the cosine sends gamma_t -> 0. Scaling the decay by gamma_t/gamma_max — here
# exactly the schedule ratio — fixes the steady state at sqrt(2*lambda/gamma_max). Applied
# only to the Muon-owned normalized matrices, never the output projections (bg_head[-1],
# time_head[-1]) or the 1D AdamW group. Bit-identical to plain decoupled decay at peak LR.
WEIGHT_DECAY_SCHEDULE_CORRECTION = True

GRADIENT_CLIP_NORM = 1.0

# Two terms fused by learned Kendall-Gal homoscedastic weighting (risk_loss.py): L_Q, the
# pinball loss over the 7 QUANTILE_LEVELS, and (1-MSE_ALPHA)·L_D + MSE_ALPHA·L_M — DILATE on
# the median mixed with the median's risk-space MSE. Both log-σ live on a separate module —
# their own AdamW group, never Muon, EMA-excluded.
DILATE_ALPHA = 0.5               # alpha*shape + (1-alpha)*TDI
# softmin softness, NOT an overflow guard: the max-subtracted softmin is overflow-free down
# to γ=1e-3 and soft-DTW is 1-homogeneous in (cost, γ), so the single-cell cost peak
# (f(BG_CLAMP_MAX) - f(BG_CLAMP_MIN))² = 99.6416 forces no particular γ.
DILATE_GAMMA = 1.0
DILATE_TDI_FD_EPS = 0.05         # FD step for TDI = d/dε sDTW(C+εΩ)|0; median grad exact to O(ε)
MSE_ALPHA = 0.0                 # 0 = DILATE only (MSE skipped); 1 = MSE only (soft-DTW skipped)
KENDALL_LOGVAR_INIT = 0.0        # init for log_sigma_Q / log_sigma_D; clamped [-7, 7]

# Provenance only: stamped into every checkpoint, the summary JSON, the resolved-config
# dump and the export descriptor, and compared by nothing at load time.
ARCH_VERSION = 'risk-v5'
LOSS_SCHEMA = 'kendall-pinball-dilate-mse-v4'

# mg/dL cutoffs for a hypo / hyper excursion and for the CG-EGA / Clarke / TIR glycemic
# regions. TIR keeps the fixed clinical 70-180 band (BG_TARGET_LO/HI in train.py) so it
# stays comparable across runs.
BG_HYPO_THRESHOLD = 70.0
BG_HYPER_THRESHOLD = 180.0

# The clinical alarm reads a BAND EDGE, not the median: hypo fires when the τ=HYPO lower
# edge dips below BG_HYPO_THRESHOLD, hyper when the τ=HYPER upper edge rises above
# BG_HYPER_THRESHOLD — the conservative call on each side. Both are selectable: any
# QUANTILE_LEVELS entry on the correct side, indexed as QUANTILE_LEVELS.index(τ), never a
# bare literal. MARD / RMSE / Clarke / CG-EGA / coverage stay median-based.
HYPO_ALARM_QUANTILE_TAU = 0.25
HYPER_ALARM_QUANTILE_TAU = 0.75
assert HYPO_ALARM_QUANTILE_TAU in QUANTILE_LEVELS and HYPO_ALARM_QUANTILE_TAU < 0.5, \
    "HYPO_ALARM_QUANTILE_TAU must be a lower-half level in QUANTILE_LEVELS"
assert HYPER_ALARM_QUANTILE_TAU in QUANTILE_LEVELS and HYPER_ALARM_QUANTILE_TAU > 0.5, \
    "HYPER_ALARM_QUANTILE_TAU must be an upper-half level in QUANTILE_LEVELS"

# The band the real/sim suite (metrics/core/suite.py) scores against — a knob distinct from
# the alarm taus, though numerically equal today. pred_eff = clip(true, q[LO], q[HI]): zero
# error inside the band, distance to the nearer edge outside. A degenerate band reproduces
# the median-line numbers exactly. Index via QUANTILE_LEVELS.index(τ), never a literal.
METRIC_BAND_TAU_LO = 0.25
METRIC_BAND_TAU_HI = 0.75
assert METRIC_BAND_TAU_LO in QUANTILE_LEVELS and METRIC_BAND_TAU_LO < 0.5, \
    "METRIC_BAND_TAU_LO must be a lower-half level in QUANTILE_LEVELS"
assert METRIC_BAND_TAU_HI in QUANTILE_LEVELS and METRIC_BAND_TAU_HI > 0.5, \
    "METRIC_BAND_TAU_HI must be an upper-half level in QUANTILE_LEVELS"

# mg/dL forgiveness band, PRECISION only: a predicted excursion whose band edge is within
# this of the true value is not counted as a false alarm. Recall stays strict. 0.0 disables.
EXCURSION_PRECISION_TOLERANCE_MGDL = 10.0

# Counterfactual probe magnitudes, in RAW units, injected as a bolus at the first masked
# patch and then re-normalized through the log1p stats.
CF_CARB_BOLUS_G = 40.0
CF_INSULIN_BOLUS_U = 2.0
# Grams of carbohydrate-EQUIVALENT glucose disposal per exercise session — never minutes,
# never an intensity. Derived from T1DMSIM's own session model, not restated.
CF_EXERCISE_G = _EX_DUR_MEAN_MIN * _EX_G_PER_MIN

# Hours counted as night; a sample's forecast-origin hour is tested against this range.
# Cross-midnight wraps when END < START (22–06 is 22:00 to 06:00).
NOCTURNAL_START_HOUR = 22.0
NOCTURNAL_END_HOUR = 6.0

# θ_ema = decay · θ_ema + (1 - decay) · θ on each accepted step; validation runs under the
# shadow, which stabilizes threshold-crossing metrics. 0.0 disables it (no shadow kept).
# 0.999 decays over ~1000 steps, 0.9999 over ~10000.
EMA_DECAY = 0.999

CHECKPOINT_INTERVAL = 1000       # steps
VALIDATION_INTERVAL = 1000       # steps
LOG_INTERVAL = 100               # steps
# Every coverage row is a proportion over this many windows: at 100 its 95% interval is
# 5-11 points — wider than any effect the table tracks, and wide enough to have hidden a
# 30-minute band covering 0.83 while the row read 0.905. At 1000 it is ~1.9 points. The
# cost is the per-sample rolling long-horizon loop, which runs on every window.
VALIDATION_N_PATIENTS = 1000

# The two PER-SAMPLE probes — the long-horizon roll and the counterfactual dose response —
# cost ~77% of a validation and are read for direction, so they stop here while every
# metric scored in the batched window loop stays at VALIDATION_N_PATIENTS. Each figure they
# emit travels with its own denominator (cf_n, bg_rmse_{h}_n, night_bg_rmse_{h}_n). Set
# equal to VALIDATION_N_PATIENTS to probe every window.
VALIDATION_PROBE_N_PATIENTS = 250

# The stats pass mirrors data generation — same skill mix, same ON_THE_FLY_SIM_HOURS window
# (data.py), NOT a long 720 h run whose pooled spread is wider than the windows the model
# sees. The patient count is high because that window is short; warmup-dominated per-patient
# cost keeps the runtime modest.
NORM_N_PATIENTS = 10000

# One unconditional file, four channels: bg_absolute in Kovatchev risk space (z(f(bg)), the
# transform BEFORE the z-score), carb / insulin / exercise in log1p + z. See
# normalization.RISK_SPACE_CHANNELS.
NORM_STATS_FILE = "normalization_stats.json"

# Split-conformal calibration partition: seeds master_seed + OFFSET + i, disjoint from both
# the train hashed seeds and normalization's +1_000_000 band, so it feeds neither the loss
# nor the headline validation. calibrate_conformal.py fits the per-(step, quantile) delta
# over it and stores it in the checkpoint under ``conformal_delta``.
CALIBRATION_RESERVE_SEED_OFFSET = 2_000_000
# The conformal fit's own coverage is a random variable in the calibration size: over 60
# random splits, test coverage has sd 0.036 at n_cal=64 (p5-p95 0.838-0.946), 0.020 at 256,
# 0.015 at 512, 0.011 at 1024. Below n = 39 a mondrian region bin cannot form its own
# tau=0.05 order statistic and takes the marginal delta (mondrian.fit_mondrian). This
# ceiling has to sit above whatever calibrate_conformal.py is run with.
CALIBRATION_RESERVE_N_PATIENTS = 2000

# The long-prediction horizon is fit to the drawing: it rolls far enough to cover the
# furthest painted dose (rounded up to a whole PREDICTION_HORIZON_HOURS roll), floored at
# GUI_LONG_PREDICTION_HOURS and capped at GUI_MAX_PREDICTION_HOURS.
GUI_LONG_PREDICTION_HOURS = 8    # hours
GUI_MAX_PREDICTION_HOURS = 12    # hours
# Smooths the freehand stroke SHAPE only, before it becomes an announced override; the
# override compiler then normalizes the raw painted dose with no further smoothing, since
# the model trains on raw post-noise inputs.
GUI_PENCIL_SMOOTH_STEPS = 9      # centered-Gaussian window, in 5-min steps
