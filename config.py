"""
T1DMAI Configuration — all hyperparameters as uppercase constants.
====================================================================
The Kovatchev risk transform bakes the clinical hypo > hyper asymmetry into the
loss geometry (equal risk-distance = larger danger at low BG). The model both
takes its bg INPUT and emits its bg OUTPUT in this risk space.

Single source of truth.  Every other file in the project imports from this
module instead of redefining values, so changing a constant here propagates
to model, data pipeline, training loop, inference and GUI in one shot.

Sections (top → bottom):
    Model architecture   — d_model, n_layers, n_heads, FFN dim, patch geometry
    Sequence lengths     — context / prediction patch counts
    Prediction horizon   — PREDICTION_HORIZON_HOURS (single model, any time of day)
    Masked-BG objective  — mask sampler knobs, head slot count M
    BG quantile head     — quantile levels, head width / init, spread floor
    RoPE                 — base frequency
    Training             — total steps, batch size, schedule
    Patient sampling     — uniform-skill mixing (RETIRED; PATIENT_UNIFORM_SAMPLE_PROB pinned 0.0)
    Optimizer (Muon)     — LR, momentum, NS iterations, weight decay
    Optimizer (AdamW)    — LR, betas, weight decay, eps
    Loss                 — gradient clip, per-d balancing, Kendall-Gal dynamic PINBALL/DILATE weighting
    Excursion thresholds — hypo / hyper clinical cutoffs (CG-EGA / Clarke / TIR) + band-edge alarm τ
    EMA                  — shadow weight decay for stable validation
    Checkpoints / logs   — intervals
    Normalization        — patient pool sizing for the stats pre-pass
    Calibration reserve  — conformal-calibration seed band / count

Resolution order at training time (high → low):
    CLI flags > the values in this file.  The three mask-sampler knobs come from
    the arm named by ``$T1DMAI_ARM`` (``arms.py``), resolved at import.
``train.py`` prints the resolved config on startup so the user can confirm
what's being used.
"""

# T1DMSIM owns these three values and carries their derivations; they are read
# from it rather than restated, so a change there cannot leave a stale copy here.
# ``import config`` therefore requires the T1DMSIM symlink.
#   * SIMULATOR_WARMUP_HOURS — hours discarded from the start of every simulator
#     run so hour 0 of a trajectory sits at basal steady state.
#   * the two exercise constants — the session model CF_EXERCISE_G is derived from.
from T1DMSIM.simulator import (
    SIMULATOR_WARMUP_HOURS,
    EXERCISE_CARB_EQUIV_PER_MIN as _EX_G_PER_MIN,
    EXERCISE_DURATION_MEAN_MIN as _EX_DUR_MEAN_MIN,
)

# The mask-sampler arm, resolved ONCE here from ``$T1DMAI_ARM`` and never
# afterwards.  ``MASK_SPAN_LENGTHS``, ``MAX_MASKED_PATCHES`` and
# ``D_BALANCED_LOSS`` below are the arm's three values, and data.py, risk_loss.py,
# utils.py and metrics/protocols.py bind them at THEIR import, so rewriting one on
# the config module object after that reaches none of them.  ``arms.py`` carries
# the arms, the evidence behind each and the --help text; an unknown name raises
# there.
from arms import resolve_arm

SAMPLER_ARM, _ARM = resolve_arm()

# === Model Architecture ===
# D_MODEL, N_LAYERS, N_HEADS, PATCH_SIZE, FFN_DIM and BG_HEAD_HIDDEN (plus
# MIN/MAX_CONTEXT_PATCHES) can be rewritten by ``resize_model.py`` via its
# manual-override flags; the resulting parameter count is computed, not
# targeted.  HEAD_DIM = D_MODEL // N_HEADS and the symbolic relations
# FFN_DIM = k * D_MODEL, BG_HEAD_HIDDEN = k * D_MODEL are preserved (the FFN
# multiplier defaults to 4, the BG-head-hidden multiplier to 1), and the script
# enforces HEAD_DIM ∈ {16, 32, 64, 128}, so don't bake their numeric values into
# other code or comments.  PATCH_SIZE must divide 12 (so an integer number of
# patches tiles the hour); changing it rescales the wall-clock span of
# MIN/MAX_CONTEXT_PATCHES, which count patches, not hours.
D_MODEL = 128                    # Hidden dimension throughout the transformer
N_LAYERS = 8                     # Number of transformer blocks
N_HEADS = 8                      # Number of temporal attention heads
HEAD_DIM = D_MODEL // N_HEADS    # Per-head dimension for temporal attention
FFN_DIM = 4 * D_MODEL                   # SwiGLU FFN inner dimension (default 4 × D_MODEL; override via resize_model.py --ffn-mult)
PATCH_SIZE = 6                   # Timesteps per patch (6 × 5min = 30min)

# === Frozen input/patch geometry ===
# Input features per timestep (``N_INPUT_FEATURES``):
#     0 bg_absolute (risk-space z(f(bg))) | 1 carbs | 2 insulin |
#     3 exercise_equiv | 4 bg_masked
# feat 0 is the bg channel encoded as z( f(bg) ) — the Kovatchev risk transform
# applied BEFORE the z-score — so the model's INPUT bg lives in the same risk
# space as its OUTPUT. carb / insulin / exercise keep their log1p + z encoding.
# feat 3 is T1DMSIM's total exercise as a carbohydrate-EQUIVALENT glucose-disposal
# curve in g/step, fed at that scale: never Kovatchev-transformed and never
# rescaled to an intensity. There are NO temporal (sin/cos hour-of-day,
# day-of-week) features.
#
# Every patch is VISIBLE or MASKED. A masked patch withholds bg (feat 0) and the
# model emits a quantile fan for it. Feats 1-3 are PLAN channels: they ALWAYS
# carry their true (training / metrics) or announced (what-if / deployment)
# values, masked patches included, and nothing but what the patient announced is
# ever written into them.
#
# feat 4 bg_masked is a per-PATCH BIT written into all PATCH_SIZE of that patch's
# step-major columns. It carries no normalization statistics — no mean, no std,
# no log1p — so the feature count and the normalized-channel count are not the
# same number. It must never sit outside the step-major block: PATCH_DIM =
# PATCH_SIZE * N_INPUT_FEATURES and the ``[:, f::N_INPUT_FEATURES]`` stride idiom
# both depend on that. The masked set is ANNOUNCED rather than inferred because
# masking is not positional and z = 0 in a withheld bg slot decodes to an
# ordinary reading, not a sentinel.
N_INPUT_FEATURES = 5             # bg_absolute (risk-space), carbs, insulin, exercise_equiv, bg_masked
PATCH_DIM = PATCH_SIZE * N_INPUT_FEATURES  # PATCH_SIZE * N_INPUT_FEATURES, step-major

# Masked-patch signal handling (output-channel space {0:carb, 1:insulin, 2:exercise}).
# feat 4 is in NEITHER tuple: it is written from the sampled mask rather than from
# a signal, on the training builder and the inference builder alike.
NON_MASKABLE_FEATS = (0,)        # bg_absolute input slot is zeroed at every masked patch
MASKABLE_FEATS = (1, 2, 3)       # carb / insulin / exercise carry the announced values
# Single shared (output-channel -> input-feat) mapping consumed by BOTH data.py
# (announced-dose write) and inference.py (override-write). The only place the
# output-channel -> feature +offset lives.
CHANNEL_TO_FEAT = {0: 1, 1: 2, 2: 3}   # carb -> feat 1; insulin -> feat 2; exercise -> feat 3

# === Sequence Lengths ===
# Variable 8–24 h context. The T1DMSIM diff report (§0.5) measures pooled-CGM
# Pearson ACF₀.₂ at 5.3 h for the simulator and 2.4–4.8 h across the three
# real cohorts. The 8 h floor sits ABOVE every one of those autocorrelation
# lengths, so the whole autocorrelated span is in scope even at the narrowest
# context. 24 h is held as the working ceiling for two reasons: (1) basal is
# injected once daily on a fixed interval with no jitter (T1DMSIM/simulator.py
# BASAL_DOSE_INTERVAL_HOURS — the two analogues differ in ACTION duration,
# glargine 26 h and degludec 42 h, not in dosing cadence), so the ceiling covers
# one full injection cycle, supporting patient-identity inference for the
# patient-agnostic model; (2) the 8 h night long-horizon rolling validation
# needs ≥ 16 h of GT context to preserve real evening dynamics in the window
# through all rolls (see inference.predict_rolling sliding logic). Training
# samples n_ctx uniformly in [MIN, MAX]; the collate function left-pads to the
# BATCH maximum so every sample's window ends at the right edge.
MAX_CONTEXT_PATCHES = 48         # Max context patches (ceiling; hours = MAX_CONTEXT_PATCHES / _PATCHES_PER_HOUR)
MIN_CONTEXT_PATCHES = 16         # Min context patches (floor; hours = MIN_CONTEXT_PATCHES / _PATCHES_PER_HOUR)

# === Prediction Horizon ===
# The span of the fixed FORECAST protocol: a dense right-edge window of
# ``PREDICTION_PATCHES`` masked patches, which may start at any patch-aligned
# position in the 24-hour day. It is what ``inference.predict``, the rolling
# long-horizon validation and the counterfactual probe run, and the reference
# span the per-span median basis and the time probe scale against. A single
# model is trained on both day-time and night-time windows simultaneously;
# time-of-day is not fed explicitly (no sin/cos features) — the model infers day
# vs night dynamics from the bg / carb / insulin / exercise context alone,
# without an external band restriction.
PREDICTION_HORIZON_HOURS = 2     # Hours predicted per forward pass

# Patches per hour: 60 / (PATCH_SIZE * 5) = 2 at PATCH_SIZE=6 (one patch = 30 min).
_PATCHES_PER_HOUR = 60 // (PATCH_SIZE * 5)
PREDICTION_PATCHES = PREDICTION_HORIZON_HOURS * _PATCHES_PER_HOUR
MAX_SEQ_LEN = MAX_CONTEXT_PATCHES + PREDICTION_PATCHES

# === Masked-BG objective (mask sampler) ===
# A sample's window is T <= MAX_SEQ_LEN patches. A span ending at patch T-1 is a
# FORECAST, one starting at patch 0 a BACKCAST, anything else INFILL — three
# cases of one objective, not three modes.
#
# The sampler (data.sample_mask_spans) draws, per sample:
#     n_spans  ~ U{1 .. MASK_MAX_SPANS}
#     each L_i ~ U(MASK_SPAN_LENGTHS), independently
#     sum(L) > MAX_MASKED_PATCHES  ⇒  resample the WHOLE length vector
#     placement: stars-and-bars over the n_spans + 1 gaps
# The over-budget rejection redraws the whole length vector, never one element:
# per-element redrawing yields a different length distribution, and so a
# different d histogram. Placement spreads the slack T - sum(L) - (n_spans - 1)
# uniformly over the gaps, with one MANDATORY visible separator between
# neighbouring spans. Rejection is on LENGTHS and there is none on PLACEMENT;
# placement is uniform — no right-edge quota, no curriculum, no annealing.
#
# Two masked spans never abut. The separator is what makes the anchor, the
# per-span median basis and the DILATE length bucket well defined per span; two
# spans with nothing between them are one longer span.
MASK_MAX_SPANS = 3                             # spans per sample, drawn U{1..MASK_MAX_SPANS}
MASK_SPAN_LENGTHS = _ARM['mask_span_lengths']  # per arm SAMPLER_ARM (arms.py)
# MAX_MASKED_PATCHES is two things at once: the sampler's cap on sum(L), and M,
# the BG head's fixed slot count. The head always emits M slots, so a sample with
# fewer masked patches pads the surplus; a padded slot gathers patch 0, which
# makes its output a real number against a real target. Every loss and metric
# path MUST discard padded slots by their ``valid`` flag, or they train against
# patch 0's BG behind a plausible neighbouring anchor.
MAX_MASKED_PATCHES = _ARM['max_masked_patches']   # per arm SAMPLER_ARM (arms.py)

# === Night long-horizon validation ===
# Validation rolls ``predict_rolling`` out to ``NIGHT_LONG_HORIZON_HOURS`` hours
# (night-only samples) so the validation table can report ``bg_rmse`` at
# horizons longer than a single forward pass. When
# ``NIGHT_LONG_HORIZON_HOURS`` > ``PREDICTION_HORIZON_HOURS`` the validation runs
# ``(NIGHT_LONG_HORIZON_HOURS − PREDICTION_HORIZON_HOURS) / PREDICTION_HORIZON_HOURS``
# extra rolls per nocturnal sample so the longer-horizon (``BG_HORIZONS_MIN``)
# SOTA targets in the validation table are populated.
# Set equal to ``PREDICTION_HORIZON_HOURS`` to skip rolling entirely.
NIGHT_LONG_HORIZON_HOURS = 8
NIGHT_LONG_HORIZON_PATCHES = NIGHT_LONG_HORIZON_HOURS * _PATCHES_PER_HOUR

# === BG quantile head (risk space) ===
# Absolute BG is forecast directly in Kovatchev RISK space by a single shared
# high-capacity head. It reads each MASKED patch's hidden state, gathered by
# index into the MAX_MASKED_PATCHES slots, and emits, per (slot, timestep), a
# median delta plus 2*N_SPREADS softplus spreads from which the 7 quantiles are
# assembled monotonically around that slot's OWN anchor f(anchor_bg) — the
# nearest visible BG, not one anchor broadcast across the window. See
# utils.assemble_quantiles.
#   * QUANTILE_LEVELS  — the 7 forecast quantile levels τ, ascending. q_tau[...,i]
#     corresponds to QUANTILE_LEVELS[i] index-for-index; QUANTILE_LEVELS.index(0.5)
#     is the median column.
#   * N_QUANTILES      — len(QUANTILE_LEVELS) = 7.
#   * N_SPREADS        — quantile spreads emitted per side of the median (3 up + 3 down).
#   * BG_HEAD_HIDDEN   — hidden width of the 3-layer MLP (override via resize_model.py
#     --head-hidden-mult). Raw output width is 1 + 2*N_SPREADS = 7
#     (col0 = median delta; cols 1..3 = τ>.5 spreads .75/.9/.95 nearest->far;
#      cols 4..6 = τ<.5 spreads .25/.1/.05).
#   * BG_HEAD_INIT_SCALE — std of the head's final-layer weight init. Small so the
#     median delta ≈ 0 at step 0 and the initial forecast is persistence
#     (median == f(anchor_bg)), departing from there under the loss.
#   * BG_QUANTILE_SPREAD_MIN — additive floor on every softplus spread, preventing
#     σ-collapse and guaranteeing a strict quantile gap.
QUANTILE_LEVELS = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)   # ascending τ; QUANTILE_LEVELS.index(0.5) = median
N_QUANTILES = 7                  # len(QUANTILE_LEVELS)
N_SPREADS = 3                    # quantile spreads per side of the median (head raw width = 1 + 2*N_SPREADS = 7)
BG_HEAD_HIDDEN = 1 * D_MODEL     # hidden width of the BG quantile head (override via resize_model.py --bg-head-hidden-mult)
BG_HEAD_INIT_SCALE = 1e-2        # final-layer init std — small ⇒ median≈f(anchor_bg) at init ⇒ forecast starts at persistence
BG_QUANTILE_SPREAD_MIN = 1e-3    # additive floor on each softplus spread (anti σ-collapse; guarantees strict quantile gap)

# --- Smooth-basis step expansion (structural anti-oscillation) ---
# The head emits BG_HEAD_STEP_BASIS_DIM (= K) coefficients per (slot, channel),
# expanded across the PATCH_SIZE within-patch timesteps by a FIXED low-frequency
# basis (model.py step_basis buffer), NOT PATCH_SIZE independent values. With
# K < PATCH_SIZE the highest-frequency (period-2) mode is unrepresentable, so an
# intra-patch zigzag of the median cannot be emitted by construction.
# K = PATCH_SIZE recovers a fully-free per-step head exactly.
BG_HEAD_STEP_BASIS_DIM = 3       # K: coeffs per (patch, channel); K<PATCH_SIZE forbids the period-2 within-patch mode
BG_HEAD_STEP_BASIS_TYPE = 'dct'  # within-patch basis over PATCH_SIZE points: 'dct' (low-freq DCT-II rows) or 'poly' (orthonormal polynomials)

# Median-assembly mode (R3 selector). assemble_quantiles is the SOLE chokepoint
# (train forward, inference.predict, predict_rolling), so this gate propagates
# everywhere identically.
#   * 'global'      — R3 DEFAULT. The per-patch median delta head_raw[...,0] is
#                     projected onto a fixed low-frequency DCT-II subspace span(Bg)
#                     spanning ONE masked span's n = L*PATCH_SIZE steps (Bg orthonormal,
#                     G_L columns): m = anchor + (delta_flat @ Bg @ Bgᵀ).reshape(L,S).
#                     A projection is an L2 CONTRACTION (||proj(x)|| <= ||x||) so the
#                     per-patch offset o[p]=m[:,p,0]-anchor CANNOT drift or amplify — it
#                     is bounded and non-monotone in p, structurally killing R1's
#                     unconstrained-integrator drift. Low-pass G_L keeps the median
#                     smooth (the per-patch period-2 / seam sawtooth is unrepresentable),
#                     so the seam penalty is redundant under this mode. C0
#                     seam-continuity and first-step pinning are NOT guaranteed
#                     (intentional — enforcing exact C0 is what caused R1's drift).
#   * 'cumulative'  — R1 integrator: each patch continues from the previous patch's
#                     endpoint (exclusive cumsum of the within-patch rise), C0-continuous
#                     at every seam, first step pinned to the anchor. BIT-IDENTICAL legacy
#                     path (kept for ablation/regression).
#   * 'independent' — legacy flat m = anchor + delta. BIT-IDENTICAL legacy path.
# At init (delta≈0 from BG_HEAD_INIT_SCALE) every mode gives m≈anchor (persistence).
BG_HEAD_MEDIAN_MODE = 'global'
# G — the number of low-frequency DCT-II basis columns — is per span, not fixed:
#     G_L = max(1, ceil(BG_HEAD_MEDIAN_GLOBAL_DIM * L / PREDICTION_PATCHES))
# (utils.global_median_dim), so BG_HEAD_MEDIAN_GLOBAL_DIM is the value at
# L = PREDICTION_PATCHES and shorter spans scale down from it. The basis kind
# reuses BG_HEAD_STEP_BASIS_TYPE. Only consulted in 'global' mode.
# A FIXED G is a DEFECT here, not an approximation: at L = 1 the projection would
# have as many columns as the span has steps, i.e. the identity, so the
# anti-drift contraction is ABSENT rather than weakened — and every fan assert
# still passes. What G_L holds roughly constant is the fraction of the span the
# basis can bend; the cutoff period 2n/G_L varies with L and is what to report.
BG_HEAD_MEDIAN_GLOBAL_DIM = 6

# === Time-of-day probe (co-trains the trunk; TIME_PROBE_DETACH=False) ===
# An auxiliary head that classifies each MASKED patch's absolute hour-of-day
# into one of TIME_PROBE_N_BINS circular bins (softmax logits, NO mean-pool — the
# head runs on every masked-patch hidden state). It is trained by cross-entropy
# against a wrapped-Gaussian circular soft label (TIME_PROBE_LABEL_SMOOTH_BINS std in
# bins). The probe is trained by PER-PATCH CROSS-ENTROPY ONLY: within one window the
# masked patches share one forward/context and are already clock-consistent through the
# trunk (ablation: tod_jump_h ≈ 0.12 h with or without a within-window penalty), so
# there is no within-window inter-patch advance penalty. A SEPARATE cross-window
# teacher-forced penalty (TIME_PROBE_CROSS_WINDOW_WEIGHT) couples consecutive
# INDEPENDENT-forward windows so the rolling clock advances by exactly one horizon
# across each seam; 0.0 disables it (and skips the 2nd forward), and
# TIME_PROBE_CROSS_WINDOW_FRACTION subsamples the 2nd training forward.
# TIME_PROBE_DETACH=False (the current, CRUCIAL
# setting) leaves the per-patch hidden states ATTACHED, so the probe loss back-props
# into the shared trunk — a REPRESENTATION-SHAPING auxiliary that forces the per-patch
# hidden states (which the BG head also reads) to encode circadian phase, making the
# forecast time-aware while the model still INFERS the clock from the trajectory (no
# time input). The forward VALUE of q_tau/median is unaffected either way (the head
# never feeds them in the forward); only the training gradient differs. Set
# TIME_PROBE_DETACH=True to restore the read-only diagnostic (probe learns but never
# touches the trunk). The CE (+ cross-window) loss is added to the TRAINING backward only
# (scaled by TIME_PROBE_LOSS_WEIGHT — the key knob trading BG accuracy for
# phase-awareness); it NEVER enters risk_total_loss, the validation loss total, or
# checkpoint selection (which stay on BG accuracy). Decoding uses the resultant vector
# of the bin distribution; its length R encodes phase confidence (R→1 certain, R→0
# ambiguous). TIME_PROBE_ENABLED = False leaves the head unbuilt.
TIME_PROBE_ENABLED = True        # master switch; False ⇒ head not built, forward bit-identical to today
TIME_PROBE_HIDDEN = 1 * D_MODEL  # hidden width of the 2-layer SiLU probe MLP (D_MODEL defined above)
TIME_PROBE_DETACH = False        # False ⇒ probe co-trains the shared trunk (CRUCIAL); True ⇒ read-only diagnostic
TIME_PROBE_LOSS_WEIGHT = 1.0     # weight of the probe CE (+ cross-window) loss in the TRAINING backward only
TIME_PROBE_INIT_SCALE = 1e-2     # std of the probe final-layer weight init
# The bin count is a LABEL-RESOLUTION choice for circadian phase, not a
# consequence of a forecast horizon: a masked span is 1..MASK_SPAN_LENGTHS[-1]
# patches and the probe scores each masked patch against its own absolute hour.
# The expression is kept because it yields bins that tile 24 h exactly, with
# PREDICTION_HORIZON_HOURS entering as the reference span — the fixed forecast
# protocol's length — rather than as the horizon of every forward.
TIME_PROBE_N_BINS = max(1, round(24.0 / PREDICTION_HORIZON_HOURS))   # 12 circular bins, 2 h each
TIME_PROBE_BIN_HOURS = 24.0 / TIME_PROBE_N_BINS                      # exact tiling of 24 h
TIME_PROBE_LABEL_SMOOTH_BINS = 0.75                                  # wrapped-Gaussian soft-label std in bins; <=0 => one-hot
TIME_PROBE_CROSS_WINDOW_WEIGHT = 1.0    # weight of the cross-window (paired-window) phase-advance penalty rel. to CE; 0.0 => 2nd forward skipped entirely (feature OFF)
TIME_PROBE_CROSS_WINDOW_FRACTION = 1.0  # run the 2nd (next-window) TRAINING forward on the first ceil(frac*B) batch rows; <1.0 trades cross-window coupling strength for step cost (validation always uses all valid samples)

# === RoPE ===
# Position is carried by RoPE alone — there is no additive per-head distance bias
# on the logits. Per-head QK-norm (RMSNorm on Q and K) is kept.
ROPE_BASE = 1000                 # Base frequency for rotary position embeddings

# === Training ===
MASTER_SEED = 11011922           # Default master seed for all training data
DETERMINISTIC = False            # full-determinism toggle; True trades some speed (TF32 off, cuDNN deterministic) for reproducibility; set False for max throughput (GPU: data stream reproducible, SDPA backward not bit-exact)
TOTAL_STEPS = 100000             # Total training steps
BATCH_SIZE = 512                 # Training batch size
NUM_WORKERS = 20                 # DataLoader worker processes

# When training off a memory-mapped simulator cache (cache_format=npy-memmap),
# each random row read faults pages of the multi-TB cache files into the kernel
# page cache. Over a large pool the access never repeats, so the page cache
# climbs until it consumes free RAM (on unified-memory GPUs this presents as
# rising "VRAM"). With this enabled, T1DMDataset copies each row out and then
# issues madvise(MADV_DONTNEED) on the touched range. (fadvise(DONTNEED) cannot
# reclaim pages mapped into the process; madvise on the mapping can.)
#
# CRUCIAL: MADV_DONTNEED alone does NOT bound growth, because the kernel's
# per-fault readahead (128 KB ≈ 32 pages) pulls far more than the row's ~2
# pages, and DONTNEED only drops the row range — the readahead remainder stays
# resident (measured: ~128 KB/read, ~0.5 GB/step). T1DMDataset therefore also
# issues madvise(MADV_RANDOM) on each channel mapping at open (see _load_cache),
# suppressing readahead at the source; the two together drop growth ~160×.
#
# No effect on the blosc2 cache format or on-the-fly generation. blosc2 has the
# same unbounded-growth failure and no cure — it exposes no mapping to madvise —
# so _load_cache does not map a .b2nd at all and reads it through ordinary file
# I/O, whose page cache is charged to nobody and reclaimed under pressure.
CACHE_MADVISE_DONTNEED = True
WARMUP_STEPS = 2000              # Steps for linear LR warmup
LR_MIN_RATIO = 0.001             # Cosine decay floor as fraction of peak LR

# === Patient sampling ===
# Uniform-skill mixing is RETIRED. The project now trains off the shared
# T1DMSIM generator's cache (T1DMSIM/cache_simulator.py), whose tail exposure
# comes from that generator's regime — a CGM-rail filter (windows touching the
# rails are discarded; T1DMSIM derives them as the open interval
# BG_CLAMP_MIN + 1 … BG_CLAMP_MAX - 1, never as literals, so a clamp change
# cannot desynchronise them) plus optional hypoglycemia oversampling
# (--hypo-oversample) — NOT from uniform-skill patient draws. That generator
# emits pure multivariate-normal-skill patients, so every cache it writes
# carries patient_uniform_sample_prob = 0.0; the loader's equality check against
# this value therefore requires it to be 0.0. With probability p (> 0), a
# training sample's patient would instead be drawn with skills sampled uniformly
# across [SKILL_MIN, SKILL_MAX] (the very skilled and the very unskilled that the
# normal sampler produces rarely); 0.0 disables that entirely (pure normal
# sampling).
PATIENT_UNIFORM_SAMPLE_PROB = 0.0

# === Optimizer: Muon ===
MUON_LR = 0.02                   # Muon LR for 2D weight matrices
MUON_MOMENTUM = 0.95             # Muon momentum coefficient
MUON_NS_ITERATIONS = 5           # Quintic Newton-Schulz iterations (5 = standard count)
MUON_WEIGHT_DECAY = 0.05         # Muon decoupled weight decay (prevents weight growth)

# === Optimizer: AdamW ===
ADAM_LR = 0.003                  # AdamW LR for embeddings and 1D parameters
ADAM_BETAS = (0.9, 0.95)         # AdamW beta coefficients
ADAM_WEIGHT_DECAY = 0.05         # AdamW weight decay
ADAM_EPS = 1e-8                  # AdamW epsilon for numerical stability

# === Schedule-aware weight-decay correction (AdamC — arXiv 2506.02285) ===
# Defazio (2025), "Why Gradients Rapidly Increase Near the End of Training".
# A weight matrix feeding a normalization has its gradient orthogonal to its
# weights, so decoupled weight decay drives it to the steady state
# ||g||/||x|| = sqrt(2*lambda/gamma_t). A cosine schedule sends gamma_t -> 0 at
# the end of training, so that target diverges and the gradient norm blows up.
# The AdamC fix (Algorithm 1) scales the decay coefficient by gamma_t/gamma_max,
# fixing the steady state at the schedule-independent sqrt(2*lambda/gamma_max).
# It is applied ONLY to the normalized layers — every 2D weight matrix feeding a
# norm, i.e. the Muon-owned matrices here — and EXCLUDES the output projections
# (bg_head[-1], time_head[-1]) and the 1D AdamW group (norm gains / biases),
# none of which is a matrix-followed-by-a-norm. Since gamma_max = peak LR and
# gamma_t = peak*ratio, the correction factor is exactly the schedule ``ratio``:
# the per-step multiply p*(1 - gamma_t*lambda) becomes
# p*(1 - gamma_t*(gamma_t/gamma_max)*lambda) = p*(1 - (gamma_t^2/gamma_max)*lambda).
# At peak LR (ratio == 1) this is bit-identical to plain decoupled decay, so the
# tuned peak-LR regime is untouched; the correction only reduces decay as the LR
# decays. Set False to restore uncorrected Muon weight decay.
WEIGHT_DECAY_SCHEDULE_CORRECTION = True

# === Loss ===
GRADIENT_CLIP_NORM = 1.0         # Max gradient norm

# Per-d loss balancing. Uniform mask PLACEMENT is not uniform supervision by
# DIFFICULTY: d — a masked patch's distance in patches to the nearest visible
# evidence on either side — is heavily concentrated at d = 1 under the sampler,
# while the deployed forecast needs one patch at each d it spans. With this on,
# risk_loss multiplies each masked patch by the per-d weight from d_balance, so
# d = 1, 2, 3 and d >= 4 each carry equal loss mass. It reweights the LOSS only:
# the sampler and its uniform placement are untouched. It costs variance, by a
# factor the span ceiling sets: quote d_balance.effective_sample_size() at the
# active arm with any result produced under it.
D_BALANCED_LOSS = _ARM['d_balanced_loss']         # per arm SAMPLER_ARM (arms.py)

# === Risk-space BG loss (DILATE + pinball, Kendall-Gal uncertainty-weighted) ===
# The BG forecast is supervised entirely in Kovatchev RISK space (see
# risk_loss.py / dilate.py). Two terms are fused by LEARNED Kendall-Gal
# homoscedastic uncertainty weighting (risk_loss.KendallGalWeighting):
#   * L_Q — the pinball (quantile) loss over the 7 QUANTILE_LEVELS, anchoring the
#     pointwise quantile levels (τ=0.5 included as the level anchor, complementing
#     the warp-invariant DILATE).
#   * L_D — DILATE on the median only: DILATE_ALPHA * shape + (1 - DILATE_ALPHA) * TDI,
#     where shape is a DIVERGENCE soft-DTW and TDI is the temporal-distortion index.
#     DILATE is recomputed at validation (at VAL_BATCH_SIZE) — the soft-DTW DP runs
#     in both phases.
# The combine learns a per-term log-σ (KendallGalWeighting.log_sigma_Q/_D, each
# init KENDALL_LOGVAR_INIT, clamped [-7, 7]) in the numerically-safe form
#   L_KG = 0.5*exp(-2*log_sigma_Q)*L_Q + log_sigma_Q
#        + 0.5*exp(-2*log_sigma_D)*L_D + log_sigma_D.
# The two scalar log-σ live on a SEPARATE module (their own AdamW group, never
# Muon) and are EMA-EXCLUDED by living off-model.
# DILATE_GAMMA is the soft-DTW softmin softness knob, NOT an overflow guard: the
# max-subtracted softmin is overflow-free down to γ=1e-3, and soft-DTW is
# 1-homogeneous in (cost, γ) so the single-cell cost peak
# (= (f(BG_CLAMP_MAX) - f(BG_CLAMP_MIN))² = 99.6416) does not force any
# particular γ. γ purely tunes how soft the min is.
DILATE_ALPHA = 0.5               # DILATE shape/TDI mix: alpha*shape + (1-alpha)*TDI. Equal split.
DILATE_GAMMA = 1.0               # soft-DTW softmin softness (1-homogeneous in (cost,γ); stabilized softmin is overflow-free to γ=1e-3)
DILATE_TDI_FD_EPS = 0.05         # finite-difference step for the TDI directional-derivative surrogate: TDI = <A,Ω> = d/dε sDTW(C+εΩ)|0 ≈ [sDTW(C+εΩ)−sDTW(C)]/ε. The median gradient is the exact TDI gradient to O(ε); smaller ⇒ less bias but less fp headroom in the value difference
KENDALL_LOGVAR_INIT = 0.0        # init for both KendallGalWeighting log-σ params (log_sigma_Q, log_sigma_D); clamped [-7, 7]

# === Checkpoint schema-version stamps (provenance) ===
# The architecture + loss-schema strings train.py stamps into every checkpoint,
# the training summary JSON, the resolved-config dump, the training_config block
# and the export descriptor (exporters/descriptor.py). They record which
# architecture and loss produced a set of weights and gate nothing: no code
# compares them at load time, there is no checkpoint-loading path in train.py,
# and the fine-tune scripts carry a base checkpoint's stamp forward unchanged.
ARCH_VERSION = 'risk-v4'              # model-architecture schema (risk-space input + output; [40,400]-anchored Kovatchev, no input/target smoothing)
LOSS_SCHEMA = 'kendall-pinball-dilate-v3'  # loss schema (Kendall-Gal dynamic pinball/DILATE uncertainty-weighted combine on the risk-v4 arch)

# === Input / target signal space (NO smoothing) ===
# The model consumes the RAW simulator post-noise signals (bg_observed with CGM
# sensor noise, carb/insulin with the AR(1) absorption noise, exercise as the
# deterministic gamma disposal curve) directly — there is no causal smoother on
# inputs or on the forecast target. The forecast TARGET and every per-slot anchor
# are the raw bg_observed as well, so inputs, target, loss and metrics all live
# in one raw post-noise space (bg is only clamped to the physical
# [BG_CLAMP_MIN, BG_CLAMP_MAX] range; carb/insulin/exercise are floored at 0 by
# the log1p transform). Deployment realism is intrinsic: the live CGM/dose stream
# is fed exactly as-is, and the autoregressive roll re-feeds the model's own raw
# output.

# BG thresholds (mg/dL) defining a hypo / hyper excursion and the glycemic
# regions for CG-EGA / Clarke / TIR. These drive the CG-EGA region split and the
# hypo_recall / hyper_recall validation metrics (the risk transform itself bakes
# the hypo>hyper danger asymmetry into the loss, so there is no separate focal
# weighting). Time-in-range keeps the fixed clinical 70-180 band (BG_TARGET_LO/HI
# in train.py) so TIR stays comparable across runs. These are glycemic-region
# cutoffs consumed by CG-EGA / Clarke / TIR, not focal / alarm / anneal weighting
# (which the risk transform subsumes into the loss geometry).
BG_HYPO_THRESHOLD = 70.0
BG_HYPER_THRESHOLD = 180.0

# Band-edge alarm quantiles for the clinical hypo / hyper RECALL + PRECISION
# detectors (validation table only — pooled, per-horizon, nocturnal and
# night-onset variants all key off these). The alarm reads a BAND EDGE, not the
# median: hypo fires when the τ=HYPO_ALARM_QUANTILE_TAU LOWER band edge
# f_inv(q_tau[..., idx]) dips below BG_HYPO_THRESHOLD; hyper fires when the
# τ=HYPER_ALARM_QUANTILE_TAU UPPER band edge rises above BG_HYPER_THRESHOLD — the
# clinically conservative call (lower / upper envelope, not the point forecast).
# The two τ are SELECTABLE: pick any QUANTILE_LEVELS entry on the correct side
# (lower-half for hypo, upper-half for hyper); the band index is
# QUANTILE_LEVELS.index(τ), never a bare literal. MARD / RMSE / Clarke / CG-EGA /
# roc / excursion / coverage all stay median-based.
HYPO_ALARM_QUANTILE_TAU = 0.25   # τ whose LOWER band edge triggers the hypo alarm (< BG_HYPO_THRESHOLD)
HYPER_ALARM_QUANTILE_TAU = 0.75  # τ whose UPPER band edge triggers the hyper alarm (> BG_HYPER_THRESHOLD)
assert HYPO_ALARM_QUANTILE_TAU in QUANTILE_LEVELS and HYPO_ALARM_QUANTILE_TAU < 0.5, \
    "HYPO_ALARM_QUANTILE_TAU must be a lower-half level in QUANTILE_LEVELS"
assert HYPER_ALARM_QUANTILE_TAU in QUANTILE_LEVELS and HYPER_ALARM_QUANTILE_TAU > 0.5, \
    "HYPER_ALARM_QUANTILE_TAU must be an upper-half level in QUANTILE_LEVELS"

# Band edges the REAL/SIM comparison suite (realdata/metrics.py) SCORES against — a
# knob distinct from the alarm taus above, though numerically equal today. The forecast
# is a quantile fan, so the effective point forecast is the band point nearest the
# truth: pred_eff = clip(true, q[METRIC_BAND_TAU_LO], q[METRIC_BAND_TAU_HI]) — zero
# error inside the band, distance to the nearer edge outside it. Every downstream
# formula (RMSE / MAE / MARD / Clarke / CG-EGA / skill) is unchanged; a degenerate band
# reproduces the median-line numbers exactly. The band index is
# QUANTILE_LEVELS.index(τ), never a bare literal.
METRIC_BAND_TAU_LO = 0.25   # lower band edge the real/sim level metrics score against
METRIC_BAND_TAU_HI = 0.75   # upper band edge
assert METRIC_BAND_TAU_LO in QUANTILE_LEVELS and METRIC_BAND_TAU_LO < 0.5, \
    "METRIC_BAND_TAU_LO must be a lower-half level in QUANTILE_LEVELS"
assert METRIC_BAND_TAU_HI in QUANTILE_LEVELS and METRIC_BAND_TAU_HI > 0.5, \
    "METRIC_BAND_TAU_HI must be an upper-half level in QUANTILE_LEVELS"

# Precision-only forgiveness band for the clinical detectors (validation table).
# A predicted hypo/hyper whose BAND EDGE is within EXCURSION_PRECISION_TOLERANCE_MGDL
# of the true value is NOT counted as a false alarm — so CGM sensor noise near a
# threshold does not deflate precision. RECALL stays strict (no tolerance): the
# conservative band edge is the whole point. Set 0.0 to disable (precision then the
# plain confusion matrix).
EXCURSION_PRECISION_TOLERANCE_MGDL = 10.0

# === Counterfactual probes (validation table) ===
# The model is always conditioned on the announced future carb / insulin /
# exercise, so the headline property is that it responds correctly to
# HYPOTHETICAL doses the patient dials in. The counterfactual probe
# (train._run_counterfactual_probe) perturbs the doses over a right-edge forecast
# span relative to a baseline forecast and checks the BG response has the correct
# sign, is monotone, and resolves excursions. Magnitudes are in RAW units (grams
# of carb, units of insulin), injected as a bolus at the first masked patch then
# re-normalized through the log1p stats.
CF_CARB_BOLUS_G = 40.0           # carb bolus added for the carb dose-response / hypo-rescue probe (g)
CF_INSULIN_BOLUS_U = 2.0         # insulin bolus added for the insulin dose-response / hyper-rescue probe (U)
# Grams of carbohydrate-EQUIVALENT glucose disposal per exercise session — never
# minutes, never an intensity. Derived from T1DMSIM's own session model (a
# population-mean session at its per-minute disposal rate), not restated.
CF_EXERCISE_G = _EX_DUR_MEAN_MIN * _EX_G_PER_MIN

# === Nocturnal validation ===
# Hours considered "night" for the nocturnal metrics section in the
# validation table.  The prediction-zone start hour is checked against
# this range; only samples starting inside the window contribute to
# nocturnal metrics.  Cross-midnight is supported: NOCTURNAL_END < NOCTURNAL_START
# means the window wraps (e.g. 22–06 = 22:00 to 06:00).
NOCTURNAL_START_HOUR = 22.0
NOCTURNAL_END_HOUR = 6.0

# === Weight EMA for evaluation ===
# Shadow copy of model weights updated each accepted optimizer step as
# θ_ema = decay · θ_ema + (1 - decay) · θ. Validation runs under EMA weights,
# which smooths out optimizer jitter at eval time — two consecutive
# checkpoints 100 steps apart should now produce similar predictions, which
# stabilizes threshold-crossing metrics like hypo/hyper recall. Set to 0.0
# to disable (validation runs on live weights, no shadow copy maintained).
# Typical values: 0.999 (decays over ~1000 steps) or 0.9999 (~10000 steps).
EMA_DECAY = 0.999

# === Checkpointing & Logging ===
CHECKPOINT_INTERVAL = 1000       # Save checkpoint every N steps
VALIDATION_INTERVAL = 1000       # Run validation every N steps
LOG_INTERVAL = 100               # Log to CSV every N steps
VALIDATION_N_PATIENTS = 100      # Number of patients in the validation set

# === Normalization ===
# The normalization pass mirrors the cache / on-the-fly DATA GENERATION so the
# saved per-channel stats match the distribution the model actually trains on:
# the same patient_uniform_sample_prob skill mix and the same
# ON_THE_FLY_SIM_HOURS post-warmup window (defined in data.py — NOT a long 720 h
# run, whose pooled spread is wider than the windows the model ever sees).
# Because that generation window is short relative to a full multi-day run, the
# patient count is raised to keep the sample broad and capture more
# patient-parameter diversity; the warmup-dominated per-patient cost keeps total
# runtime modest.
NORM_N_PATIENTS = 10000          # Patients for normalization statistics

# === Normalization stats file ===
# The bg_absolute INPUT channel (feat 0) is fed as z( f(bg) ) — the Kovatchev risk
# transform applied BEFORE the z-score — so the model's INPUT bg lives in the same
# risk space as its OUTPUT / loss / target. The bg mean/std therefore live in risk
# space; carb/insulin/exercise keep their log1p+z transform. A single unconditional
# stats file holds all four channels (see normalization.RISK_SPACE_CHANNELS).
NORM_STATS_FILE = "normalization_stats.json"

# === Conformal-calibration reserve (ACTIVE) ===
# A disjoint patient partition that backs split-conformal quantile calibration of
# the BG bands. Seeds use the band ``master_seed + CALIBRATION_RESERVE_SEED_OFFSET
# + i`` for ``i in range(CALIBRATION_RESERVE_N_PATIENTS)`` — disjoint from both the
# train hashed seeds and normalization's ``+1_000_000`` band, so this partition
# never leaks into training or the headline validation. ``calibrate_conformal.py``
# runs the trained model over it, fits the per-(step, quantile) conformal ``delta``
# (``conformal.fit_quantile_conformal``), and stores it in the checkpoint under
# ``conformal_delta``; the in-training ``conf_*`` validation probe is a separate
# held-out witness. The partition still feeds NEITHER the loss nor the headline
# validation metrics — only the post-hoc band recalibration.
CALIBRATION_RESERVE_SEED_OFFSET = 2_000_000   # disjoint seed band (vs norm's +1_000_000 and train's hashed seeds)
CALIBRATION_RESERVE_N_PATIENTS = 500          # patients in the held-out conformal-calibration partition

# === GUI what-if / long-prediction knobs ===
# The interactive GUI (``gui.py``) drives the model through ``inference.predict``
# (single 2 h forward, bound to SPACE) and ``inference.predict_rolling`` (multi-roll,
# bound to the long-prediction key). ``predict_rolling`` conditions every 2 h roll on
# the doses the user painted via the curve editor / pencil (absolute-patch aware), so
# a single long prediction can honor doses drawn arbitrarily far into the future.
#
# The long-prediction horizon is *fit to the drawing*: it rolls out far enough to
# cover the furthest painted dose (rounded up to a whole ``PREDICTION_HORIZON_HOURS``
# roll), floored at ``GUI_LONG_PREDICTION_HOURS`` (so an unedited long prediction still
# reaches the headline 8 h) and capped at ``GUI_MAX_PREDICTION_HOURS``.
GUI_LONG_PREDICTION_HOURS = 8    # floor / default horizon (h) for the long-prediction key
GUI_MAX_PREDICTION_HOURS = 12    # cap (h) for the fit-to-drawing horizon extension

# Freehand pencil strokes are jagged; smooth each stroke with a centered Gaussian of
# this window (in 5-min steps) BEFORE it becomes an announced override. The override
# compiler normalizes the raw painted dose directly with NO further smoothing (the
# model trains on raw post-noise inputs, so smoothing here would MISMATCH that
# distribution); this is purely the "make my drawing look smooth" pass on the raw
# painted shape.
GUI_PENCIL_SMOOTH_STEPS = 9      # centered-Gaussian window (5-min steps) for pencil strokes
