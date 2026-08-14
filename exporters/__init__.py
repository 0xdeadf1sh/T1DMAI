"""T1DMAI model exporters.

One module per inference engine so more backends can be added later (per the
project's backend-seam requirement). Everything engine-agnostic — the modified
head_raw forward, the fixed-T struct-mask builder, checkpoint loading, and the
descriptor emitter — lives in the shared helpers here; each engine module
(e.g. ``executorch_xnnpack``) only owns the lowering + serialization specific to
its runtime.

Graph cut: the exported graph stops at ``head_raw`` (B, P, S, 1+2*N_SPREADS) in
Kovatchev risk space. The per-slot anchor, softplus+floor, cumsum, the P*S DCT
median projection, ``carry_spread``, ``f_inv`` and quantile assembly all live
downstream (Rust ``t1dm-core``), NOT in the graph.

Right edge: the exported graph is the RIGHT-EDGE SPECIALISATION of the general
masked-BG objective — the masked set is the trailing ``PREDICTION_PATCHES`` patches,
read by slice, so no ``mask_idx`` crosses the graph boundary.
"""
