"""Tests for resize_model.py — the meta-instantiation param count and the
config rewrite round-trip. resize_model rewrites config.py in place, so the
write test operates on a throwaway temp copy, never the live config.
"""
import tempfile
from pathlib import Path

import torch


def test_count_params_matches_direct_instantiation():
    """_count_params (meta-instantiation) equals a direct numel of T1DMAI at the
    current config dims, and is positive."""
    import config
    from resize_model import _count_params

    n_meta = _count_params(config.D_MODEL, config.N_HEADS, config.FFN_DIM,
                           config.BG_HEAD_HIDDEN, config.N_LAYERS)
    # _count_params re-imported model with the (unchanged) current config.
    from model import T1DMAI
    with torch.device('meta'):
        m = T1DMAI()
    n_direct = sum(p.numel() for p in m.parameters())

    print(f"[DUMP] resize param count: meta={n_meta} direct={n_direct}")
    assert n_meta == n_direct
    assert n_meta > 0


def test_count_params_scales_with_d_model():
    """A wider model has strictly more parameters (sanity on the override path)."""
    import config
    from resize_model import _count_params

    base = _count_params(config.D_MODEL, config.N_HEADS, config.FFN_DIM,
                         config.BG_HEAD_HIDDEN, config.N_LAYERS)
    wider = _count_params(512, 8, 4 * 512, 2 * 512, config.N_LAYERS)
    print(f"[DUMP] resize scale: base(d={config.D_MODEL})={base} wider(d=512)={wider}")
    assert wider > base
    # Restore the live config dims (the helper mutated the module globals).
    _count_params(config.D_MODEL, config.N_HEADS, config.FFN_DIM,
                  config.BG_HEAD_HIDDEN, config.N_LAYERS)


def test_write_constants_round_trip_preserves_neighbors():
    """_write_constants rewrites only the named RHS, preserving every other line
    and the trailing comment of the rewritten line."""
    from resize_model import _write_constants

    src = Path('config.py').read_text()
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f:
        f.write(src)
        tmp = Path(f.name)
    try:
        _write_constants(tmp, {'D_MODEL': '320'})
        out = tmp.read_text()
        # Target RHS rewritten.
        assert any(line.lstrip().startswith('D_MODEL = 320') for line in out.splitlines())
        assert 'D_MODEL = 128' not in out
        # Trailing comment on the D_MODEL line preserved.
        assert 'Hidden dimension throughout the transformer' in out
        # Same number of lines (an in-place RHS patch, not a structural rewrite).
        assert len(out.splitlines()) == len(src.splitlines())
        print("[DUMP] write round-trip: D_MODEL rewritten, neighbours + comment intact")
    finally:
        tmp.unlink()
