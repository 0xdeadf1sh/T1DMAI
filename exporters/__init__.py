"""Model exporters, one module per inference engine; everything engine-agnostic is shared here.

Graph cut at ``head_raw`` (B, M, S, 1+2*N_SPREADS), Kovatchev risk space, plus ``slot_hidden``.
Anchor, softplus+floor, cumsum, per-span DCT median, ``carry_spread``, ``f_inv``, assembly: consumer side (Rust ``t1dm-core``).
Masked set is an input — ``(M, T)`` one-hot over the ``M = MAX_MASKED_PATCHES`` slots; forecast, backcast and infill are one artifact.
"""
