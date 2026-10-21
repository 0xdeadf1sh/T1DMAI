"""Golden vectors for SPEC/zoo.md §3's bg fill, the half a consumer reimplements.

python -m zoo.golden [--out tests/data/zoo_window_golden.json]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from finetune_data import _interp_short_gaps
from zoo.windows import MAX_INTERP_STEPS, _fill_context

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(ROOT, 'tests', 'data', 'zoo_window_golden.json')


def window(history: np.ndarray, ctx: int) -> np.ndarray:
    """``history`` ends at the latest measured reading; NaN marks an unmeasured slot."""
    filled = _interp_short_gaps(history.astype(np.float32), MAX_INTERP_STEPS)
    if len(filled) < ctx:
        filled = np.concatenate([np.full(ctx - len(filled), np.nan, np.float32), filled])
    return _fill_context(filled[None, -ctx:])[0]


def cases() -> list[dict]:
    rng = np.random.default_rng(7)
    out = []
    for i in range(12):
        n = int(rng.integers(20, 120))
        ctx = int(rng.choice([24, 36, 48, 72]))
        h = (140 + np.cumsum(rng.normal(0, 4, n))).round(1)
        for _ in range(int(rng.integers(0, 6))):
            s = int(rng.integers(0, n - 1))
            h[s:s + int(rng.integers(1, 12))] = np.nan
        h[-1] = 150.0 + i
        out.append({'ctx': ctx, 'history': [None if np.isnan(v) else float(v) for v in h],
                    'window': [float(v) for v in window(h, ctx)]})
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--out', default=DEFAULT_OUT)
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({'interp_max_steps': MAX_INTERP_STEPS, 'cases': cases()}, f, indent=1)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
