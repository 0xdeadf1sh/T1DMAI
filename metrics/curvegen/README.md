# curvegen

Reads `T1DMDROID`'s insulin preset catalogue and emits each rapid preset's
per-five-minute action curve as JSON, for `metrics/whatif.py --insulin-curve-json`.

## Why it exists

The what-if probe injects a bolus and asks whether the forecast falls. Which curve
it injects decides what the answer means: models are pretrained on the simulator's
**gamma** family and run on the phone's **Loop/OpenAPS exponential** presets, so a
probe using the wrong family measures an insulin nobody takes
(`../../../T1DMCOMMON/SPEC/invariants.md` §5).

The exponential model is `exp_action_curve` in `t1dm-core`. Reimplementing it here
would be a second copy of curve mathematics the whole suite shares. This binary
links the real one instead and prints its output, so the probe consumes data and
T1DMAI carries no formula.

## Use

Needs a `T1DMDROID` checkout beside this one and a Rust toolchain.

```sh
cargo run --release > presets/all.json          # the whole catalogue
```

Then split it per preset, or hand the probe an object carrying a
`curve_per_5min_unit_total` key. `presets/` and `target/` are gitignored — both are
regenerable.

Each record carries the preset's label, family, `peak_min`, `dia_min`, the Bateman
rates (basal only), its `off_distribution` flag, and the citation the insulin panel
renders. Rapid presets also carry the resolved curve, normalized to a unit total;
basal presets carry an empty curve, since a Bateman needs a duration the caller
chooses.

## Boundary

Only `curve.rs` is read, and it is present on T1DMDROID's public `main`. Nothing is
vendored. `Cargo.lock` is gitignored on purpose: a lock resolved against a checkout
sitting on T1DMDROID's local-only branch would enumerate that branch's dependency
tree, and this repository is public.
