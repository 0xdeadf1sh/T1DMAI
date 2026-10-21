import json
import os

import numpy as np
import pytest
import torch

from zoo.chronos import ChronosForecaster, load_pretrained, pinball, relative_bucket, weights_present
from zoo.classical import Persistence
from zoo.windows import ZONE, Windows

GOLDEN = os.path.join(os.path.dirname(__file__), 'data', 'chronos_bolt_golden.json')
needs_tiny = pytest.mark.skipif(not weights_present('tiny'),
                                reason='chronos-bolt-tiny weights not in ~/.cache/huggingface')


def _windows(bg: np.ndarray, target: np.ndarray) -> Windows:
    n, ctx = bg.shape
    z = np.zeros((n, ctx + ZONE), np.float32)
    return Windows(bg.astype(np.float32), z, z.copy(), target.astype(np.float32),
                   np.full(n, 'x', object), np.full(n, -1))


def _sinusoid(n: int, ctx: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.arange(ctx + ZONE)[None, :] + rng.uniform(0, 200, (n, 1))
    period = rng.uniform(48, 96, (n, 1))
    x = 140 + 50 * np.sin(2 * np.pi * t / period)
    return x[:, :ctx], x[:, ctx:]


def test_relative_bucket_matches_t5_table():
    rel = torch.arange(-200, 201)
    b = relative_bucket(rel, True, 32, 128)
    assert b[200] == 0 and b[201] == 17 and b[199] == 1
    assert b.min() >= 0 and b.max() == 31 and b[0] == 15
    uni = relative_bucket(rel, False, 32, 128)
    assert (uni[201:] == 0).all() and uni[0] == 31


def test_pinball_ignores_unmeasured_steps():
    lv = torch.tensor([0.1, 0.5, 0.9])
    q = torch.zeros(2, 3, 64)
    y = torch.ones(2, ZONE)
    y_nan = y.clone()
    y_nan[:, 5:] = float('nan')
    full, part = pinball(q, y, lv), pinball(q, y_nan, lv)
    assert torch.isclose(full, torch.tensor(float(ZONE)))
    assert torch.isclose(part, full * 5 / ZONE)


@needs_tiny
def test_golden_parity_with_reference_pipeline():
    with open(GOLDEN) as f:
        g = json.load(f)
    x = torch.tensor([[np.nan if v is None else v for v in r] for r in g['inputs']])
    want = torch.tensor(g['expected'])
    net = load_pretrained('tiny')
    with torch.no_grad():
        got = net(x)[:, :, :g['horizon']]
    scale = net.scale(x)[2].clamp(min=1.0)
    rel = ((got - want).abs() / scale[:, :, None]).amax().item()
    print(f'DUMP chronos-bolt-tiny golden max |diff|/scale {rel:.2e}')
    assert rel < 1e-4
    full = torch.isfinite(x).all(dim=1)
    assert full.sum() >= 2
    with torch.no_grad():
        q, loc, sc = net.dense(x[full], *net.tables(x.shape[1] // net.patch_size + 1))
    dense = (q * sc[:, :, None] + loc[:, :, None])[:, :, :g['horizon']]
    rel_d = ((dense - want[full]).abs() / scale[full][:, :, None]).amax().item()
    print(f'DUMP gap-free dense path vs golden max |diff|/scale {rel_d:.2e}')
    assert rel_d < 1e-4


@needs_tiny
def test_shapes_and_finite():
    bg, target = _sinusoid(5, 288)
    w = _windows(bg, target)
    m = ChronosForecaster('tiny', finetune=False)
    m.fit(w)
    p = m.predict(w)
    assert p.shape == (5, ZONE) and np.isfinite(p).all()
    with torch.no_grad():
        q = m.net(torch.from_numpy(bg[:2].astype(np.float32)))
    assert q.shape == (2, 9, 64)
    assert (q.diff(dim=1) >= -1e-3).all()


@needs_tiny
def test_zero_shot_beats_persistence_on_sinusoid():
    bg, target = _sinusoid(64, 288)
    w = _windows(bg, target)
    zs = ChronosForecaster('tiny', finetune=False).predict(w)
    pers = Persistence().predict(w)
    e_zs = np.sqrt(np.mean((zs - target) ** 2))
    e_p = np.sqrt(np.mean((pers - target) ** 2))
    print(f'DUMP sinusoid rmse chronos-bolt-tiny zs {e_zs:.2f}  persistence {e_p:.2f}')
    assert e_zs < 0.5 * e_p


def _graph_inputs(n: int) -> tuple[Windows, tuple[torch.Tensor, ...]]:
    bg, target = _sinusoid(n, 288, seed=2)
    w = _windows(bg, target)
    return w, tuple(torch.from_numpy(np.asarray(x, np.float32)) for x in (w.bg, w.insulin, w.carb))


@needs_tiny
def test_graph_matches_predict():
    w, args = _graph_inputs(8)
    m = ChronosForecaster('tiny', finetune=False)
    p = m.predict(w)
    with torch.no_grad():
        g = m.graph()(*args).numpy()
    diff = np.abs(g - p).max()
    print(f'DUMP chronos graph vs predict max {diff:.2e} mg/dL')
    assert g.shape == (8, ZONE) and diff < 1e-2


@needs_tiny
def test_graph_exports_without_dropout_or_nan_ops():
    w, args = _graph_inputs(1)
    graph = ChronosForecaster('tiny', finetune=False).graph()
    with torch.no_grad():
        ep = torch.export.export(graph, args, strict=False)
        diff = (ep.module()(*args) - graph(*args)).abs().max().item()
    ops = {str(n.target) for n in ep.graph.nodes if n.op == 'call_function'}
    print(f'DUMP chronos exported vs eager max {diff:.2e} mg/dL, {len(ops)} ops')
    assert diff < 1e-3
    banned = {'dropout', 'isnan', 'nan_to_num', 'embedding', 'log'}
    assert not [o for o in ops if any(b in o for b in banned)], sorted(ops)


@needs_tiny
def test_finetune_moves_weights_and_round_trips():
    bg, target = _sinusoid(96, 288, seed=1)
    w = _windows(bg, target)
    m = ChronosForecaster('tiny', finetune=True, epochs=1, batch=32, lr=1e-4)
    before = m.predict(w)
    m.fit(w)
    after = m.predict(w)
    assert np.isfinite(after).all() and not np.allclose(before, after)
    m2 = ChronosForecaster('tiny', finetune=True)
    m2.load(m.state())
    assert np.allclose(m2.predict(w), after, atol=1e-4)
