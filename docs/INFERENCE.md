# Inference

The model contract this repository produces — the three spaces, the checkpoint
keys, the architecture, the attention mask, the risk transform, normalization,
the frozen index map, the decode and its reference constants — is specified once,
for the whole suite, in **`T1DMCOMMON/SPEC/inference.md`**. It is not restated
here: `T1DMCOMMON/scripts/check-no-copies.sh` fingerprints that specification and
fails on a second copy, whether it sits in a document or in a source comment.

- Repository: <https://github.com/0xdeadf1sh/T1DMCOMMON>
- Sibling checkout: `../T1DMCOMMON/SPEC/inference.md`
- The two risk spaces it depends on: `../T1DMCOMMON/SPEC/invariants.md` §4

What follows is only what is true of **this repository**.

## Where it is implemented here

| Concern | Where |
| --- | --- |
| Architecture, `forward`, the RoPE cache, the per-patch step basis | `model.py` |
| Risk transform and its guards, the attention masks, `assemble_quantiles`, the per-span median basis | `utils.py` |
| Per-channel normalize / denormalize | `normalization.py` |
| Single-window, rolling and what-if inference | `inference.py` |
| Dimensions and released defaults, including the mask-sampler constants | `config.py` |
| The conformal fit and apply, and its region bins | `conformal.py`, `mondrian.py`, run by `calibrate_conformal.py` |
| Physical BG bounds | `T1DMSIM/simulator.py`, through the symlinked checkout |

`inference.predict` and `inference.predict_rolling` implement the two recipes in
the specification; `predict_what_if` is `predict` with overrides. `predict` takes
its masked set as `mask_spans`.

## Export

`exporters/modified_forward.py` holds the modified forward every target shares —
external struct mask, right-edge slice, graph cut at `head_raw`, dual output —
and `exporters/` lowers it through one partitioner per target:
`executorch_xnnpack.py`, `executorch_vulkan.py`, `litert_npu.py`.
`descriptor.py` writes the sidecar all three emit, and it is the only place the
decode constants leave this repository: the `kovatchev` block it stamps is what
the app decodes against, so an export is the moment a re-anchoring becomes real
for every consumer.

An artifact and its descriptor are one unit. Ship them together, from the same
export run — nothing downstream can detect a mismatched pair.
