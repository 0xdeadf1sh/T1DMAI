import numpy as np
import pytest

from zoo.foundation import TIMESTRAL_REVISIONS
from zoo.windows import ZONE, Windows

READS = 0.01  # mg/dL mean shift; a dropped covariate gives 0 up to kernel noise
CASES = [('chronos2', 'chronos', True), ('timesfm25', 'timesfm', False),
         *((f'timestral_{z.lower()}', 'timestral', True) for z in TIMESTRAL_REVISIONS)]


def _windows(n: int, ctx: int, seed: int) -> Windows:
    """Random walks around per-row levels 80-300 mg/dL, a dose 1 h back and a meal in the zone."""
    rng = np.random.default_rng(seed)
    level = rng.uniform(80, 300, (n, 1))
    bg = (level + np.cumsum(rng.normal(0, 2, (n, ctx)), axis=1)).astype(np.float32)
    ins = rng.exponential(0.05, (n, ctx + ZONE)).astype(np.float32)
    ins[:, ctx - 12] += rng.uniform(2, 6, n)
    ins[:, ctx + 2] += rng.uniform(2, 6, n)
    carb = np.zeros((n, ctx + ZONE), np.float32)
    carb[:, ctx - 12] = rng.uniform(20, 60, n)
    carb[:, ctx + 2] = rng.uniform(20, 60, n)
    return Windows(bg, ins, carb, np.full((n, ZONE), np.nan, np.float32),
                   np.full(n, 'x', object), np.full(n, -1))


PERIOD = 32  # steps; not a divisor of ZONE, so a reversed horizon does not fit the phase


def _sines(n: int, ctx: int) -> tuple[Windows, np.ndarray]:
    """``(windows, zone truth)``: 40 mg/dL sines around 150, one phase per row; no doses."""
    phase = np.linspace(0, 2 * np.pi, n, endpoint=False)[:, None]
    full = 150 + 40 * np.sin(2 * np.pi * np.arange(ctx + ZONE) / PERIOD + phase)
    z = np.zeros((n, ctx + ZONE), np.float32)
    w = Windows(full[:, :ctx].astype(np.float32), z, z.copy(),
                np.full((n, ZONE), np.nan, np.float32), np.full(n, 'x', object), np.full(n, -1))
    return w, full[:, ctx:]


def _zeroed(w: Windows, sl: slice, channels=('carb', 'insulin')) -> Windows:
    w = w.take(np.arange(len(w)))
    for ch in channels:
        getattr(w, ch)[:, sl] = 0.0
    return w


@pytest.mark.parametrize('key,pkg,reads_past', CASES)
def test_foundation_dose_reads_per_track(key, pkg, reads_past):
    pytest.importorskip(pkg)
    from zoo.foundation import MODELS
    m = MODELS[key]()
    c, zone = m.ctx, slice(m.ctx, None)
    w = _windows(64, c, 0)
    m.causal = True
    blind = m.predict(w)
    blind_zone0 = m.predict(_zeroed(w, zone))
    past_shift = np.abs(blind - m.predict(_zeroed(w, slice(None, c)))).mean()
    m.causal = False
    given = m.predict(w)
    shift = {ch: np.abs(given - m.predict(_zeroed(w, zone, (ch,)))).mean()
             for ch in ('carb', 'insulin')}
    persist = np.abs(blind - w.bg[:, -1:]).mean()
    print(f'DUMP {key}: blind vs zeroed-zone {np.abs(blind - blind_zone0).max():.2e}  '
          f'blind vs zeroed-past {past_shift:.3f}  given vs zeroed-zone carb '
          f"{shift['carb']:.3f} insulin {shift['insulin']:.3f}  "
          f'blind vs persistence {persist:.2f} mg/dL')
    assert blind.shape == given.shape == (64, ZONE)
    assert np.isfinite(blind).all() and np.isfinite(given).all()
    assert np.abs(blind - blind_zone0).max() < 1e-3
    assert (past_shift > READS) if reads_past else (past_shift < 1e-3)
    assert shift['carb'] > READS and shift['insulin'] > READS
    assert persist < 5.0


class _Recorder:
    """Stands in for ``TimestralForecaster``: logs each call, median = last reading + step."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def forecast(self, context, prediction_length, **kw):
        import torch
        self.calls.append({'context': np.array(context), 'h': prediction_length, **kw})
        last = torch.as_tensor(np.asarray(context)[:, -1:], dtype=torch.float32)
        median = (last + torch.arange(prediction_length))[:, None]
        return type('F', (), {'median': median})()


@pytest.mark.parametrize('causal', [True, False])
def test_timestral_hands_each_chunk_its_own_rows(monkeypatch, causal):
    import zoo.foundation as fnd
    monkeypatch.setattr(fnd, 'TIMESTRAL_BATCH', 7)
    m = fnd.MODELS.get('timestral_5m', fnd._timestral('5M'))()
    m.causal, m._fc = causal, _Recorder()
    w = _windows(20, m.ctx, 1)
    p = m.predict(w)
    doses = np.stack([w.insulin, w.carb], axis=1)
    want = {'past_covariates': doses[..., :m.ctx]} if causal else {'known_covariates': doses}
    assert [len(c['context']) for c in m._fc.calls] == [7, 7, 6]
    for i, call in enumerate(m._fc.calls):
        sl = slice(7 * i, 7 * i + 7)
        assert call['h'] == ZONE and call['symmetric_averaging'] is True
        assert set(call) - {'context', 'h', 'symmetric_averaging'} == set(want)
        np.testing.assert_array_equal(call['context'], w.bg[sl])
        for k, v in want.items():
            np.testing.assert_array_equal(call[k], v[sl])
    np.testing.assert_allclose(p, w.bg[:, -1:] + np.arange(ZONE), rtol=0, atol=1e-4)


@pytest.mark.parametrize('size', list(TIMESTRAL_REVISIONS))
def test_timestral_loads_its_own_pinned_checkpoint(monkeypatch, size):
    import sys
    import types

    import zoo.foundation as fnd
    seen = []
    loader = types.SimpleNamespace(from_pretrained=lambda *a, **kw: seen.append((a, kw)) or 'fc')
    monkeypatch.setitem(sys.modules, 'timestral', types.SimpleNamespace(TimestralForecaster=loader))
    import torch
    key = f'timestral_{size.lower()}'
    m = fnd.MODELS.get(key, fnd._timestral(size))()
    assert m.name == key and m._forecaster() == 'fc'
    (args, kw), = seen
    assert args == (f'mistralai/Timestral-{size}',)
    assert kw == {'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                  'revision': TIMESTRAL_REVISIONS[size]}  # max_context: the trained 32768


@pytest.mark.parametrize('size', list(TIMESTRAL_REVISIONS))
def test_timestral_checkpoint_reads_the_whole_context(size):
    pytest.importorskip('timestral')
    import zoo.foundation as fnd
    m = fnd._timestral(size)()
    fc = m._forecaster()
    assert fc.max_context >= m.ctx and fc.supports_known_covariates


@pytest.mark.parametrize('key,pkg', [c[:2] for c in CASES])
def test_foundation_keeps_row_and_horizon_order(key, pkg):
    pytest.importorskip(pkg)
    from zoo.foundation import MODELS
    m = MODELS[key]()
    w, truth = _sines(32, m.ctx)
    m.causal = True
    p = m.predict(w)
    err = np.abs(p - truth).mean()
    flipped = np.abs(p[:, ::-1] - truth).mean()
    rolled = np.abs(np.roll(p, len(p) // 2, axis=0) - truth).mean()
    print(f'DUMP {key}: sine continuation error {err:.2f}  horizon flipped {flipped:.2f}  '
          f'rows rolled {rolled:.2f} mg/dL')
    assert err < 8.0 < min(flipped, rolled)
