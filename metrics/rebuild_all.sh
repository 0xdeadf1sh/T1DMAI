#!/usr/bin/env bash
# set -e is load-bearing: a failing step would leave the PREVIOUS run's stats.json looking current.
set -eu
cd "$(dirname "$0")/.."                       # repo root
# Bare `python` is the distro python with no torch; override with PYTHON=... elsewhere.
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
