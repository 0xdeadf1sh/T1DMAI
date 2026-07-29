#!/usr/bin/env bash
# Regenerate the entire metrics/ directory on the live best checkpoint
# (checkpoints/t1dmai_best.pt): the two formal announced-event reports
# (real / sim) and the folded-in deep-dive (48 h day-curve figures in real/ &
# sim/, plus the Q1-Q4 analyses and the what-if dose-response probe, whose JSONs
# land at metrics/ top level, whose panels land in metrics/figures/, and which
# feed metrics/README.md).
#
# Usage:  bash metrics/rebuild_all.sh
set -u
cd "$(dirname "$0")/.."                       # repo root
# Use the project venv python by default (bare `python` is the distro python with no
# torch); override with PYTHON=... if you keep the interpreter elsewhere.
PY="${PYTHON:-$PWD/venv/bin/python}"
[ -x "$PY" ] || PY=python
sep() { echo; echo "########## $* ##########"; }

# day-curves append by counting existing *_day*.png; clear so numbering restarts
rm -f metrics/real/figures/*_day*.png metrics/sim/figures/*_day*.png

sep "real report";            "$PY" metrics/real/build_report.py
sep "real figures";           "$PY" metrics/real/make_comparison_figures.py
sep "sim report";             "$PY" metrics/sim/build_report.py
sep "sim figures";            "$PY" metrics/sim/make_comparison_figures.py
sep "day-curves (real)";      "$PY" metrics/curves.py
sep "day-curves (sim)";       "$PY" metrics/curves_sim.py
sep "Q3/Q4 amplitude+var";    "$PY" metrics/amp_var.py
sep "Q2 event response";      "$PY" metrics/q2_event_response.py
sep "what-if dose response";  "$PY" metrics/whatif.py
sep "meal schedule";          "$PY" metrics/meal_stats.py
sep "time-of-day probe";      "$PY" metrics/time_probe.py
sep "ALL DONE"
