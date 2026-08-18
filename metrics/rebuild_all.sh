#!/usr/bin/env bash
# Regenerate the metrics/ directory on the live best checkpoint
# (checkpoints/t1dmai_best.pt): the in-domain simulator report, its 48 h
# day-curve figures, and the what-if dose-response probe, whose JSON lands at
# metrics/ top level and whose panels land in metrics/figures/.
#
# Usage:  bash metrics/rebuild_all.sh
#
# set -e is load-bearing: without it a failing step leaves the PREVIOUS run's
# stats.json and README.md on disk, and the report reads as current.
set -eu
cd "$(dirname "$0")/.."                       # repo root
# Use the project venv python by default (bare `python` is the distro python with no
# torch); override with PYTHON=... if you keep the interpreter elsewhere.
PY="${PYTHON:-$PWD/venv/bin/python}"
[ -x "$PY" ] || PY=python
sep() { echo; echo "########## $* ##########"; }

# day-curves append by counting existing *_day*.png; clear so numbering restarts
rm -f metrics/sim/figures/*_day*.png

sep "sim report";             "$PY" metrics/sim/build_report.py
sep "sim figures";            "$PY" metrics/sim/make_comparison_figures.py
sep "day-curves (sim)";       "$PY" metrics/curves_sim.py
sep "what-if dose response";  "$PY" metrics/whatif.py
sep "ALL DONE"
