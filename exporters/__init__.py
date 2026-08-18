"""T1DMAI model exporters.

One module per inference engine so more backends can be added later (per the
project's backend-seam requirement). Everything engine-agnostic — the modified
head_raw forward, the fixed-T struct-mask builder, checkpoint loading, and the
descriptor emitter — lives in the shared helpers here; each engine module
(e.g. ``executorch_xnnpack``) only owns the lowering + serialization specific to
its runtime.

Graph cut: the exported graph stops at ``head_raw`` (B, M, S, 1+2*N_SPREADS) in
Kovatchev risk space, and emits ``slot_hidden`` beside it so a consumer can re-run
the BG head itself from the exported head weights. The per-slot anchor,
softplus+floor, cumsum, the per-span DCT median projection, ``carry_spread``,
``f_inv`` and quantile assembly all live downstream (Rust ``t1dm-core``), NOT in
the graph.

Masked set: the graph takes it as an input — a ``(M, T)`` one-hot selection matrix
naming the patch each of the ``M = MAX_MASKED_PATCHES`` head slots reads. A
trailing span is a forecast, a leading one a backcast, anything between an infill;
one artifact serves all three.
"""
