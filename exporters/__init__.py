"""Model exporters, one module per inference engine; everything engine-agnostic is shared here.

Graph cut at ``head_raw`` (B, M, S, 1+2*N_SPREADS), Kovatchev risk space, plus ``hidden`` (B, T, D_MODEL).
Anchor, softplus+floor, cumsum, ``carry_spread``, ``f_inv``, assembly, and the node gather + B-spline step states + head MLP that rebuild ``head_raw`` from ``hidden``: consumer side (Rust ``t1dm-core``).
Masked set is an input — ``(M, T)`` one-hot over the ``M = MAX_MASKED_PATCHES`` slots; forecast, backcast and infill are one artifact.
"""
