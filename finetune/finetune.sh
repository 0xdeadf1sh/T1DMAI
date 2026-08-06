#!/bin/bash

# python finetune.py --checkpoint large.pt --holdout 591 --steps 700 --batch-size 32 --lr-scale 0.2 --warmup 60 --eval-interval 100 --device cuda --dataset ohiot1dm --seed 2
# python finetune_multi.py --checkpoint large.pt --steps 5000 --batch-size 32 --lr-scale 0.2 --warmup 150 --eval-interval 100 --device cuda --seed 0
python finetune_personal.py --checkpoint large.pt --db ~/t1dm-backup-20260726.db --steps 400 --batch-size 32 --lr-scale 0.05 --warmup 40 --eval-interval 25 --device cuda --seed 0

# --------------------------------------------------------------------------- #
# nano-personal / small-personal, 2026-08-06. Run from the REPO ROOT, not here.
#
# The record is the live sync database; the excluded range is the ~21 h from
# 2026-07-30 11:10 to 2026-07-31 08:10 where a long-acting basal was taken but
# never logged, so the insulin channel reads near-zero over hours the patient
# was in fact covered.
# --------------------------------------------------------------------------- #

# Align config.py to the nano checkpoint — the architecture guard hard-fails otherwise.
# python resize_model.py --d-model 32 --heads 2 --layers 2

# Leave-one-day-out sweep over the 7 days that can host the 3-day reservation, at three learning rates, to pick nano's.
# for LR in 0.05 0.15 0.30; do for D in 2026-07-21 2026-07-22 2026-07-23 2026-07-24 2026-08-03 2026-08-04 2026-08-05; do
#   python finetune/finetune_personal.py --checkpoint models/nano/weights_sim.pt --db ~/Desktop/T1DMSERVER/data/t1dm.db --exclude-range 2026-07-30T11:10 2026-07-31T08:10 --val-day $D --out /tmp/nano-lr$LR-$D.pt --steps 300 --batch-size 32 --lr-scale $LR --warmup 30 --eval-interval 10 --device cuda --seed 0
# done; done

# The shipped nano fit, on the default median-variability reservation at the learning rate the sweep chose.
# python finetune/finetune_personal.py --checkpoint models/nano/weights_sim.pt --db ~/Desktop/T1DMSERVER/data/t1dm.db --exclude-range 2026-07-30T11:10 2026-07-31T08:10 --out models/nano/weights_personal.pt --steps 300 --batch-size 32 --lr-scale 0.15 --warmup 30 --eval-interval 10 --device cuda --seed 0

# Align config.py to the small checkpoint before touching it.
# python resize_model.py --d-model 64 --heads 4 --layers 4

# The same sweep for small, plus a 600-step arm because 300 was a binding budget on two folds.
# for LR in 0.05 0.15 0.30; do for D in 2026-07-21 2026-07-22 2026-07-23 2026-07-24 2026-08-03 2026-08-04 2026-08-05; do
#   python finetune/finetune_personal.py --checkpoint models/small/weights_sim.pt --db ~/Desktop/T1DMSERVER/data/t1dm.db --exclude-range 2026-07-30T11:10 2026-07-31T08:10 --val-day $D --out /tmp/small-lr$LR-$D.pt --steps 300 --batch-size 32 --lr-scale $LR --warmup 30 --eval-interval 10 --device cuda --seed 0
# done; done

# The shipped small fit — 7/7 folds improved on every metric, so this is the one worth deploying.
# python finetune/finetune_personal.py --checkpoint models/small/weights_sim.pt --db ~/Desktop/T1DMSERVER/data/t1dm.db --exclude-range 2026-07-30T11:10 2026-07-31T08:10 --out models/small/weights_personal.pt --steps 600 --batch-size 32 --lr-scale 0.30 --warmup 30 --eval-interval 20 --device cuda --seed 0

# Export each capacity to both backends, from .venv-export (executorch 1.3.1 / torch 2.12.0), with config.py aligned to that capacity first.
# .venv-export/bin/python -m exporters.executorch_xnnpack --checkpoint models/nano/weights_personal.pt  --model-id nano-personal  --out-dir exported
# .venv-export/bin/python -m exporters.executorch_vulkan  --checkpoint models/nano/weights_personal.pt  --model-id nano-personal  --out-dir exported --write-pte --fp16
# .venv-export/bin/python -m exporters.executorch_xnnpack --checkpoint models/small/weights_personal.pt --model-id small-personal --out-dir exported
# .venv-export/bin/python -m exporters.executorch_vulkan  --checkpoint models/small/weights_personal.pt --model-id small-personal --out-dir exported --write-pte --fp16
# .venv-export/bin/python -m exporters.executorch_xnnpack --checkpoint models/large/weights_personal.pt --model-id large-personal --out-dir exported
# .venv-export/bin/python -m exporters.executorch_vulkan  --checkpoint models/large/weights_personal.pt --model-id large-personal --out-dir exported --write-pte --fp16
#
# A first xnnpack export of a RETRAINED model reports RESULT: FAIL on d_regress — that check
# compares against whatever .pte already sits in exported/, and new weights are meant to differ.
# Re-run it once the new file is in place and d_regress falls to 0.

# Align to medium, sweep it, and take lr 0.30 — the only capacity to improve on 7/7 folds at every metric.
# python resize_model.py --d-model 128 --heads 8 --layers 8
# python finetune/finetune_personal.py --checkpoint models/medium/weights_sim.pt --db ~/Desktop/T1DMSERVER/data/t1dm.db --exclude-range 2026-07-30T11:10 2026-07-31T08:10 --out models/medium/weights_personal.pt --steps 600 --batch-size 32 --lr-scale 0.30 --warmup 30 --eval-interval 20 --device cuda --seed 0

# Align to large and fit it — best of the ladder at the long horizon (h120 RMSE -12.8 mg/dL against small's -9.6).
# python resize_model.py --d-model 256 --heads 16 --layers 16
# python finetune/finetune_personal.py --checkpoint models/large/weights_sim.pt --db ~/Desktop/T1DMSERVER/data/t1dm.db --exclude-range 2026-07-30T11:10 2026-07-31T08:10 --out models/large/weights_personal.pt --steps 600 --batch-size 32 --lr-scale 0.30 --warmup 30 --eval-interval 20 --device cuda --seed 0

# Counterfactual dose response, per capacity and variant, with config.py aligned to that capacity first.
# The insulin curves come from metrics/curvegen (which links the app's own t1dm-core) as JSON;
# the carb triples are the (k, theta, duration) a real meal_event row stores. Neither maths is restated here.
# python metrics/whatif.py --checkpoint models/<CAP>/weights_<sim|personal>.pt --dataset personal --db ~/Desktop/T1DMSERVER/data/t1dm.db --exclude-range 2026-07-30T11:10 2026-07-31T08:10 --all-windows --stride-patches 4 --cap 400 --carb-gamma 3.25 22.5 292.5 --insulin-curve-json metrics/curvegen/presets/aspart-novorapid-novolog.json --arm-label ref --out whatif/<CAP>-<VAR>-ref.json --no-figures

# Restore config.py to the large architecture it is checked in at.
# python resize_model.py --d-model 256 --heads 16 --layers 16

# Deploy: the server pairs each artifact with a SIBLING <stem>.json, so the descriptor loses its .descriptor infix.
# for V in nano-personal small-personal large-personal; do for E in xnnpack vulkan; do
#   cp exported/$V.$E.pte ~/Desktop/T1DMSERVER/data/models/$V.$E.pte
#   cp exported/$V.$E.descriptor.json ~/Desktop/T1DMSERVER/data/models/$V.$E.json
# done; done
