# curvegen

Reads `T1DMDROID`'s insulin preset catalogue and emits each preset's
per-five-minute action curve as JSON, for `metrics/whatif.py --insulin-curve-json`.

## Why it exists

The what-if probe injects a bolus and asks whether the forecast falls. The curve
it injects decides what the answer means. The catalogue is the insulin table of
`../../../T1DMCOMMON/SPEC/invariants.md` §5, which `T1DMSIM` and the phone share.

`preset_curve` in `t1dm-core` resolves a preset to its curve. This binary links it
and prints its output, so the probe consumes data and T1DMAI carries no formula.

## Use

Needs a `T1DMDROID` checkout beside this one and a Rust toolchain.

```sh
cargo run --release > presets/all.json          # the whole catalogue
```

Then split it per preset, or hand the probe an object carrying a
`curve_per_5min_unit_total` key. `presets/` and `target/` are gitignored — both are
regenerable.

Each record carries the preset's label, family, gamma parameters at 5 U (rapid),
Bateman rates and action window (basal), the citation the insulin panel renders,
and the resolved curve at 5 U, normalized to a unit total. A rapid curve's shape
lengthens with dose, so it matches a 5 U bolus.

## Boundary

Only `curve.rs` is read, and it is present on T1DMDROID's public `main`. Nothing is
vendored. `Cargo.lock` is gitignored on purpose: a lock resolved against a checkout
sitting on T1DMDROID's local-only branch would enumerate that branch's dependency
tree, and this repository is public.
