"""All tunable constants. CLI flags override these; train.py prints the resolved set."""

# Owned by T1DMSIM (SIMULATOR_WARMUP_HOURS: hours to basal steady state); needs the symlink.
from T1DMSIM.simulator import (
    SIMULATOR_WARMUP_HOURS,
    EXERCISE_CARB_EQUIV_PER_MIN as _EX_G_PER_MIN,
    EXERCISE_DURATION_MEAN_MIN as _EX_DUR_MEAN_MIN,
)

# resize_model.py rewrites these; don't bake values elsewhere. PATCH_SIZE must divide 12.
D_MODEL = 32
N_LAYERS = 32
N_HEADS = 1
HEAD_DIM = D_MODEL // N_HEADS
FFN_DIM = 1 * D_MODEL
PATCH_SIZE = 6                   # 6 × 5 min = 30 min per patch

# FROZEN order, step-major: 0 bg z(f(bg)) | 1 carb | 2 insulin | 3 exercise g/step | 4 bg_masked.
N_INPUT_FEATURES = 5
# feat 4 has no norm stats, must stay in the step-major block; mask is ANNOUNCED, not inferred.
PATCH_DIM = PATCH_SIZE * N_INPUT_FEATURES

# Output-channel space {0: carb, 1: insulin, 2: exercise}; feat 4 is in neither tuple.
NON_MASKABLE_FEATS = (0,)        # bg input slot zeroed at every masked patch
MASKABLE_FEATS = (1, 2, 3)
# The single output-channel -> input-feat mapping; data.py and inference.py both read it.
CHANNEL_TO_FEAT = {0: 1, 1: 2, 2: 3}

# Context 84-168 h; floor is far above the sim's 5.3 h ACF (T1DMSIM/diff/README.md §0.5).
MAX_CONTEXT_PATCHES = 336        # patches, not hours: hours = / _PATCHES_PER_HOUR
MIN_CONTEXT_PATCHES = 168

# Fixed FORECAST protocol span: a dense right-edge run of PREDICTION_PATCHES masked patches.
PREDICTION_HORIZON_HOURS = 2
_PATCHES_PER_HOUR = 60 // (PATCH_SIZE * 5)
PREDICTION_PATCHES = PREDICTION_HORIZON_HOURS * _PATCHES_PER_HOUR
MAX_SEQ_LEN = MAX_CONTEXT_PATCHES + PREDICTION_PATCHES

# Sampler: n_spans ~ U{1..MASK_MAX_SPANS}; sum(L)>MAX_MASKED_PATCHES resamples the WHOLE vector.
MASK_MAX_SPANS = 3
MASK_SPAN_LENGTHS = (1, 2, 3, 4, 5, 6, 7, 8)
# Sampler cap on sum(L), and M, the head's slot count; surplus slots pad, discard via ``valid``.
MAX_MASKED_PATCHES = 12
# Share of windows whose LAST span is pinned flush right; invalidates SAMPLER_REFERENCE if moved.

# cov90@30: 0.9149 vs control 0.8805 at step 12000; plateaus at 0.15, no gain past 0.60.
MASK_RIGHT_EDGE_QUOTA = 0.50

# Rolls predict_rolling to this on nocturnal samples; == PREDICTION_HORIZON_HOURS skips rolling.
NIGHT_LONG_HORIZON_HOURS = 8
NIGHT_LONG_HORIZON_PATCHES = NIGHT_LONG_HORIZON_HOURS * _PATCHES_PER_HOUR

# q_tau[..., i] indexes QUANTILE_LEVELS[i], assembled per slot around f(anchor_bg).

# head raw cols: 0 median delta | 1-3 tau>.5 spreads .75/.9/.95 | 4-6 tau<.5 .25/.1/.05.
QUANTILE_LEVELS = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)   # ascending τ
N_QUANTILES = 7
N_SPREADS = 3                    # spreads per side of the median
BG_HEAD_HIDDEN = 1 * D_MODEL     # resize_model.py --bg-head-hidden-mult
BG_HEAD_INIT_SCALE = 1e-2        # small ⇒ median ≈ f(anchor_bg) at init ⇒ starts at persistence
BG_QUANTILE_SPREAD_MIN = 1e-3    # additive floor per softplus spread; anti σ-collapse

# Per-slot hour-of-day probe (no mean-pool); its loss never enters risk_total_loss or selection.
TIME_PROBE_ENABLED = True        # False ⇒ head not built
TIME_PROBE_HIDDEN = 1 * D_MODEL  # 2-layer SiLU probe MLP
TIME_PROBE_DETACH = False        # True ⇒ read-only diagnostic, no trunk gradient
TIME_PROBE_LOSS_WEIGHT = 1.0     # scales the probe CE + cross-window term, backward only
TIME_PROBE_INIT_SCALE = 1e-2     # probe final-layer weight init std
# Label-resolution choice, not derived from any horizon; tiles 24 h exactly.
TIME_PROBE_N_BINS = max(1, round(24.0 / PREDICTION_HORIZON_HOURS))   # 12 bins, 2 h each
TIME_PROBE_BIN_HOURS = 24.0 / TIME_PROBE_N_BINS                      # exact tiling of 24 h
TIME_PROBE_LABEL_SMOOTH_BINS = 0.75                                  # soft-label std; <=0 ⇒ one-hot
# Teacher-forced penalty coupling consecutive windows; 0.0 skips the 2nd forward entirely.
TIME_PROBE_CROSS_WINDOW_WEIGHT = 1.0
TIME_PROBE_CROSS_WINDOW_FRACTION = 1.0  # 2nd forward on first ceil(frac*B) rows; val uses all

ROPE_BASE = 1000

MASTER_SEED = 42
DETERMINISTIC = False            # True: TF32 off, cuDNN deterministic; SDPA backward not bit-exact
TOTAL_STEPS = 10000
BATCH_SIZE = 512
NUM_WORKERS = 20

# npy-memmap cache: random reads fault pages that never repeat, growing page cache/VRAM.

# MADV_DONTNEED + MADV_RANDOM together cut growth ~160x; blosc2 has no mapping to madvise.
CACHE_MADVISE_DONTNEED = True
WARMUP_STEPS = 2000              # steps of linear LR warmup
LR_MIN_RATIO = 0.01              # cosine decay floor as a fraction of peak LR

# Must stay 0.0: loader equality-checks it against the cache's patient_uniform_sample_prob.
PATIENT_UNIFORM_SAMPLE_PROB = 0.0

MUON_LR = 0.02                   # 2D weight matrices
MUON_MOMENTUM = 0.95
MUON_NS_ITERATIONS = 5           # quintic Newton-Schulz steps
MUON_WEIGHT_DECAY = 0.05         # decoupled
ADAM_LR = 0.003                  # embeddings and 1D parameters
ADAM_BETAS = (0.9, 0.95)
ADAM_WEIGHT_DECAY = 0.05
ADAM_EPS = 1e-8

# AdamC (arXiv 2506.02285): scales decay by gamma_t/gamma_max so steady state doesn't diverge.

# Applied only to Muon-owned matrices, never output projections or the 1D AdamW group.
WEIGHT_DECAY_SCHEDULE_CORRECTION = True

GRADIENT_CLIP_NORM = 1.0

# Kendall-Gal fuses L_Q with (1-MSE_ALPHA)*L_D + MSE_ALPHA*L_M; log-sigmas EMA-excluded.
DILATE_ALPHA = 0.5               # alpha*shape + (1-alpha)*TDI
# softmin softness, not an overflow guard: overflow-free down to gamma=1e-3, 1-homogeneous.

# cost peak (f(BG_CLAMP_MAX)-f(BG_CLAMP_MIN))^2 = 99.6416 forces no particular gamma.
DILATE_GAMMA = 1.0
DILATE_TDI_FD_EPS = 0.05         # FD step for TDI = d/dε sDTW(C+εΩ)|0; median grad exact to O(ε)
MSE_ALPHA = 0.0                 # 0 = DILATE only (MSE skipped); 1 = MSE only (soft-DTW skipped)
KENDALL_LOGVAR_INIT = 0.0        # init for log_sigma_Q / log_sigma_D; clamped [-7, 7]

# Provenance only: stamped into checkpoint/JSON/descriptor, compared by nothing at load time.
ARCH_VERSION = 'risk-v5'
LOSS_SCHEMA = 'kendall-pinball-dilate-mse-v4'

# mg/dL hypo/hyper cutoffs, also the CG-EGA/Clarke/TIR regions; TIR mirrors train.py's 70-180.
BG_HYPO_THRESHOLD = 70.0
BG_HYPER_THRESHOLD = 180.0

# Alarm reads a BAND EDGE, not the median; index via QUANTILE_LEVELS.index(tau), never a literal.
HYPO_ALARM_QUANTILE_TAU = 0.25
HYPER_ALARM_QUANTILE_TAU = 0.75
assert HYPO_ALARM_QUANTILE_TAU in QUANTILE_LEVELS and HYPO_ALARM_QUANTILE_TAU < 0.5, \
    "HYPO_ALARM_QUANTILE_TAU must be a lower-half level in QUANTILE_LEVELS"
assert HYPER_ALARM_QUANTILE_TAU in QUANTILE_LEVELS and HYPER_ALARM_QUANTILE_TAU > 0.5, \
    "HYPER_ALARM_QUANTILE_TAU must be an upper-half level in QUANTILE_LEVELS"

# metrics/core/suite.py's band, distinct from the alarm taus; pred_eff = clip(true, q_lo, q_hi).
METRIC_BAND_TAU_LO = 0.25
METRIC_BAND_TAU_HI = 0.75
assert METRIC_BAND_TAU_LO in QUANTILE_LEVELS and METRIC_BAND_TAU_LO < 0.5, \
    "METRIC_BAND_TAU_LO must be a lower-half level in QUANTILE_LEVELS"
assert METRIC_BAND_TAU_HI in QUANTILE_LEVELS and METRIC_BAND_TAU_HI > 0.5, \
    "METRIC_BAND_TAU_HI must be an upper-half level in QUANTILE_LEVELS"

# mg/dL forgiveness band, PRECISION only (recall stays strict); 0.0 disables.
EXCURSION_PRECISION_TOLERANCE_MGDL = 10.0

# Counterfactual probe magnitudes, RAW units, injected at the first masked patch.
CF_CARB_BOLUS_G = 40.0
CF_INSULIN_BOLUS_U = 2.0
# Grams carb-EQUIVALENT disposal per session, never minutes; derived from T1DMSIM, not restated.
CF_EXERCISE_G = _EX_DUR_MEAN_MIN * _EX_G_PER_MIN

# Hours counted as night; wraps across midnight when END < START (22-06 = 22:00 to 06:00).
NOCTURNAL_START_HOUR = 22.0
NOCTURNAL_END_HOUR = 6.0

# theta_ema=decay*theta_ema+(1-decay)*theta; validation runs under it. 0.0 disables (no shadow).
EMA_DECAY = 0.999

CHECKPOINT_INTERVAL = 1000       # steps
VALIDATION_INTERVAL = 1000       # steps
LOG_INTERVAL = 100               # steps
# Coverage row 95% CI: 5-11 pts at n=100 (hid a 0.83 band reading 0.905), ~1.9 pts at n=1000.

# Cost is the per-sample rolling long-horizon loop, which runs on every window.
VALIDATION_N_PATIENTS = 1000

# Long-horizon roll + counterfactual probe cost ~77% of validation; each figure has its own n.
VALIDATION_PROBE_N_PATIENTS = 250

# Stats pass mirrors data generation's window (data.py), not a long 720 h run with wider spread.
NORM_N_PATIENTS = 10000

# One file, 4 channels: bg z(f(bg)) risk space, others log1p+z; see RISK_SPACE_CHANNELS.
NORM_STATS_FILE = "normalization_stats.json"

# Calibration seed offset, disjoint from train and norm's +1_000_000 band; feeds no loss.
CALIBRATION_RESERVE_SEED_OFFSET = 2_000_000
# Conformal coverage sd over 60 splits: 0.036 at n=64, 0.020 at 256, 0.015 at 512, 0.011 at 1024.

# Below n=39 a mondrian bin can't form its own tau=0.05 order stat, takes the marginal delta.
CALIBRATION_RESERVE_N_PATIENTS = 2000

# Long-prediction horizon fits the drawing, rolled to cover the furthest painted dose.
GUI_LONG_PREDICTION_HOURS = 8    # hours
GUI_MAX_PREDICTION_HOURS = 12    # hours
# Smooths the freehand stroke SHAPE only; override compiler applies no further smoothing.
GUI_PENCIL_SMOOTH_STEPS = 9      # centered-Gaussian window, in 5-min steps
