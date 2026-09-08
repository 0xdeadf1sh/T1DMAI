"""Model exporters, one module per inference engine; everything engine-agnostic is shared here.
Graph cut at head_raw (B,M,S,1+2*N_SPREADS), Kovatchev risk space, plus hidden (B,T,D_MODEL).
Anchor, softplus+floor, cumsum, carry_spread, f_inv, assembly, node gather + B-spline + head
MLP rebuild head_raw from hidden: consumer side (Rust t1dm-core). Masked set (M,T) one-hot.
"""
