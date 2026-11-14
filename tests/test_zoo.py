import numpy as np

from finetune_data import HORIZON_MINUTES
from zoo.classical import LinearExtrapolation, Persistence, Ridge
from zoo.windows import ZONE, Windows, _cut, _fill_context, horizon_columns


def _windows(bg: np.ndarray, target: np.ndarray, ins=None, carb=None) -> Windows:
    n, ctx = bg.shape
    z = np.zeros((n, ctx + ZONE), np.float32)
    return Windows(bg.astype(np.float32), z if ins is None else ins, z if carb is None else carb,
                   target.astype(np.float32), np.full(n, 'x', object), np.full(n, -1))


def test_cut_zone_opens_one_step_after_the_reading():
    bg = np.arange(100, dtype=np.float32)
    ins = np.zeros(100, np.float32)
    b, _, _, t = _cut(bg, ins, ins.copy(), np.array([40]), 12, bg)
    assert b[0, -1] == 39.0
    assert t[0, 0] == 40.0
    pred_30 = horizon_columns(t)[0, 0]
    print(f'DUMP reading 39, pred_30 truth {pred_30}')
    assert pred_30 == 39.0 + HORIZON_MINUTES[0] // 5


def test_cut_past_the_end_is_nan_and_dose_free():
    bg = np.arange(50, dtype=np.float32)
    ins = np.ones(50, np.float32)
    _, i, _, t = _cut(bg, ins, ins.copy(), np.array([45]), 6, bg)
    assert np.isnan(t[0, 5:]).all() and np.isfinite(t[0, :5]).all()
    assert (i[0, 6 + 5:] == 0).all() and (i[0, :6 + 5] == 1).all()


def test_fill_context_carries_forward_then_back():
    bg = np.array([[np.nan, 5.0, np.nan, 7.0]])
    assert _fill_context(bg).tolist() == [[5.0, 5.0, 5.0, 7.0]]


def test_persistence_and_linear_trend():
    ramp = np.tile(np.arange(24, dtype=np.float32) * 2 + 100, (3, 1))
    truth = ramp[:, -1:] + 2 * np.arange(1, ZONE + 1)
    w = _windows(ramp, truth)
    assert np.allclose(Persistence().predict(w), ramp[:, -1:])
    assert np.allclose(LinearExtrapolation().predict(w), truth)


def test_graph_matches_predict_and_exports():
    import torch

    from zoo.registry import models
    rng = np.random.default_rng(1)
    reg = models()
    for name in ('persistence', 'linear_extrapolation', 'ridge_ar', 'ridge_arx', 'glunet', 'nhits'):
        m = reg[name]()
        n = 256
        bg = (130 + np.cumsum(rng.normal(0, 3, (n, m.ctx)), axis=1)).astype(np.float32)
        doses = rng.exponential(0.5, (n, m.ctx + ZONE)).astype(np.float32)
        target = bg[:, -1:] + rng.normal(0, 10, (n, ZONE))
        w = _windows(bg, target, ins=doses, carb=doses * 5)
        if hasattr(m, 'epochs'):
            m.epochs = 1
        m.fit(w)
        g = m.graph().eval()
        args = tuple(torch.from_numpy(x) for x in (w.bg, w.insulin, w.carb))
        with torch.no_grad():
            out = g(*args).numpy()
            ep = torch.export.export(g, tuple(a[:1] for a in args), strict=False)
            exp = ep.module()(*(a[:1] for a in args)).numpy()
        err = np.abs(out - m.predict(w)).max()
        print(f'DUMP {name} graph-vs-predict max {err:.2e} mg/dL')
        assert err < 1e-2
        assert np.abs(exp - out[:1]).max() < 1e-3


def test_window_golden_is_current():
    import json

    from zoo.golden import DEFAULT_OUT, cases
    with open(DEFAULT_OUT) as f:
        assert json.load(f)['cases'] == cases(), 'run python -m zoo.golden'


def test_ridge_arx_learns_announced_carbs():
    rng = np.random.default_rng(0)
    n, ctx = 4000, 72
    bg = 120 + rng.normal(0, 5, (n, ctx)).astype(np.float32)
    carb = np.zeros((n, ctx + ZONE), np.float32)
    carb[:, ctx] = rng.uniform(0, 60, n)
    target = bg[:, -1:] + 1.5 * carb[:, ctx:ctx + 1] * np.linspace(0, 1, ZONE)[None, :]
    w = _windows(bg, target, carb=carb)
    ar, arx = Ridge(exogenous=False), Ridge(exogenous=True)
    ar.fit(w)
    arx.fit(w)
    err_ar = np.sqrt(np.mean((ar.predict(w) - target) ** 2))
    err_arx = np.sqrt(np.mean((arx.predict(w) - target) ** 2))
    print(f'DUMP ridge AR rmse {err_ar:.2f}  ARX rmse {err_arx:.2f}')
    assert err_arx < 0.2 * err_ar


DAY = 288
T0 = 1_700_000_100


class _FakeCache:
    """Records whose measured bg is the calendar step since its person's epoch; 3-step gaps."""

    def __init__(self) -> None:
        def rec(pool, src, sid, day0, days, test_day=-1):
            return {'key': f'{pool}:{src}:{sid}', 'pool': pool, 'source': src, 'sid': sid,
                    'n': days * DAY, 't0': T0 + day0 * DAY * 300,
                    'test_start': test_day * DAY if test_day >= 0 else -1}
        self.subjects = [rec('phone', 'x', 'owner', 0, 40, 30),
                         rec('phone', 'x', 'owner-s1', 1, 14),
                         rec('phone', 'x', 'owner-s2', 18, 12),
                         rec('metabonet', 'B', '1', 0, 20),
                         rec('metabonet', 'C', '1', 0, 8)]

    def train_len(self, rec: dict) -> int:
        ts = int(rec['test_start'])
        return int(rec['n']) if ts < 0 else min(int(rec['n']), ts)

    def channels(self, rec: dict, lo: int, hi: int) -> dict:
        cal = (int(rec['t0']) - T0) // 300 + np.arange(lo, hi)
        bg = np.where(np.isin(cal % 50, (10, 11, 12)), np.nan, cal).astype(np.float32)
        z = np.zeros(hi - lo, np.float32)
        return {'bg': bg, 'bolus_u': z, 'basal_u': z, 'carb_g': z}


def test_holdout_blocks_are_per_person_calendar_days_and_measured():
    from zoo.windows import BLOCKS, block_spans, train_windows
    cache = _FakeCache()
    spans = block_spans(cache)
    assert (spans[1, 1:, 0] == spans[1, 1:, 1]).all(), 'a sibling ending early holds no val/cal'
    assert spans[4, 0, 0] == spans[4, 0, 1], 'a train period under 10 days fits nothing'
    ends = {'x': 30 * DAY, 'B': 20 * DAY, 'C': 8 * DAY}
    for k, block in enumerate(BLOCKS):
        w = train_windows(cache, 4000, 12, seed=k, block=block)
        last = w.bg[:, -1]
        for src, end in ends.items():
            sel = w.source == src
            lo, hi = {'fit': (-np.inf, end - 10 * DAY), 'val': (end - 10 * DAY, end - 5 * DAY),
                      'cal': (end - 5 * DAY, end)}[block]
            if sel.any():
                assert last[sel].min() >= lo and (last[sel] + ZONE).max() < hi, (block, src)
            print(f'DUMP {block} {src}: {int(sel.sum())} windows')
        assert ('C' in set(w.source)) == (block != 'fit')
        assert (w.source == 'x').any()
        assert np.isfinite(w.bg).all()
        assert not np.isin(last % 50, (10, 11, 12)).any(), 'last context reading measured'
        cal = last[:, None] + 1 + np.arange(ZONE)[None, :]
        gap = np.isin(cal % 50, (10, 11, 12))
        assert np.isnan(w.target[gap]).all(), 'targets never interpolated'
        assert np.array_equal(w.target[~gap], cal[~gap].astype(np.float32))
