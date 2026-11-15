import types

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import zoo.run as run
from config import QUANTILE_LEVELS
from finetune_data import HORIZON_MINUTES, HORIZON_STEPS
from T1DMSIM.simulator import BG_CLAMP_MAX, BG_CLAMP_MIN
from zoo.windows import ZONE, Windows

MID = QUANTILE_LEVELS.index(0.5)


def _band(rows: np.ndarray) -> np.ndarray:
    """Row r, step s, level k -> 90 + 10 r + s + 40 k; inside the clamp, distinct everywhere."""
    return (90.0 + 10 * rows[:, None, None] + np.arange(ZONE)[None, :, None]
            + 40 * np.arange(len(QUANTILE_LEVELS))[None, None, :])


class _Quantiler:
    ctx = 4

    def predict_quantiles(self, w):
        return _band(w.row.astype(float))

    def predict(self, w):
        raise AssertionError('the median comes from the band, not a second pass')


def _windows_for(rows):
    rows = np.asarray([r for r in rows if r != 2], np.int64)  # row 2 unanchorable
    n = len(rows)
    z = np.zeros((n, 4 + ZONE), np.float32)
    return Windows(np.full((n, 4), 120, np.float32), z, z.copy(),
                   np.full((n, ZONE), np.nan, np.float32), np.full(n, 'x', object), rows)


def test_predict_template_takes_median_and_band_from_one_pass(monkeypatch):
    import export_submission
    monkeypatch.setattr(run, 'template_windows',
                        lambda cache, by, t, ctx: _windows_for(next(iter(by.values()))))
    monkeypatch.setattr(export_submission, '_apply_ladder',
                        lambda pred, *a: pred.__setitem__(np.isnan(pred), 77.0))
    tpl = types.SimpleNamespace(n_rows=5, rows_by_subject={0: [0, 1, 2], 1: [3, 4]},
                                t_epoch=np.zeros(5))
    pred, anchored, band = run.predict_template(_Quantiler(), None, tpl, causal=True)
    want = _band(np.arange(5.0))
    assert anchored == 4 and band.shape == (5, ZONE, len(QUANTILE_LEVELS))
    assert np.isnan(band[2]).all() and (pred[2] == 77.0).all()
    keep = [0, 1, 3, 4]
    assert want.max() < BG_CLAMP_MAX and want.min() > BG_CLAMP_MIN
    np.testing.assert_array_equal(band[keep], want[keep])
    np.testing.assert_array_equal(pred[keep], band[keep][:, list(HORIZON_STEPS), MID])


def test_predict_template_without_quantiles_has_no_band(monkeypatch):
    import export_submission

    class _Median:
        ctx = 4

        def predict(self, w):
            return np.full((len(w), ZONE), 150.0)
    monkeypatch.setattr(run, 'template_windows',
                        lambda cache, by, t, ctx: _windows_for(next(iter(by.values()))))
    monkeypatch.setattr(export_submission, '_apply_ladder', lambda pred, *a: None)
    tpl = types.SimpleNamespace(n_rows=5, rows_by_subject={0: [0, 1, 2], 1: [3, 4]},
                                t_epoch=np.zeros(5))
    pred, anchored, band = run.predict_template(_Median(), None, tpl, causal=False)
    assert band is None and anchored == 4 and (pred[[0, 1, 3, 4]] == 150.0).all()


def test_benchmark_zoo_predictions_writes_the_template_forecast(monkeypatch, tmp_path):
    import benchmark
    import export_submission

    class _Median:
        ctx = 4

        def predict(self, w):
            return np.full((len(w), ZONE), 150.0)
    written = []
    monkeypatch.setattr(run, 'template_windows',
                        lambda cache, by, t, ctx: _windows_for(next(iter(by.values()))))
    monkeypatch.setattr(run, 'fit_or_load', lambda *a, **k: _Median())
    monkeypatch.setattr(export_submission, '_apply_ladder', lambda pred, *a: None)
    monkeypatch.setattr(export_submission, '_write', lambda t, pred, *a: written.append(pred))
    tpl = types.SimpleNamespace(n_rows=5, rows_by_subject={0: [0, 1, 2], 1: [3, 4]},
                                t_epoch=np.zeros(5), table=None)
    args = types.SimpleNamespace(rescore=False, runs='x', train_windows=1, seed=0, refit=False)
    benchmark.zoo_predictions({'key': 'persistence', 'name': 'persistence'}, ['given'], args,
                              None, tpl, str(tmp_path))
    (pred,) = written
    assert (pred[[0, 1, 3, 4]] == 150.0).all() and np.isnan(pred[2]).all()


def test_write_band_files_are_what_the_leaderboard_reads(tmp_path):
    n = 4
    cols = {'id': pa.array(np.arange(n)), 'source_file': pa.array(['s'] * n),
            'date': pa.array(np.arange(n) * 300_000_000_000, type=pa.timestamp('ns'))}
    cols |= {f'pred_{m}': pa.nulls(n, type=pa.float64()) for m in HORIZON_MINUTES}
    table = pa.table(cols)
    band = _band(np.arange(float(n)))
    band[1] = np.nan
    pred = np.where(np.isnan(band[:, list(HORIZON_STEPS), MID]), 77.0,
                    band[:, list(HORIZON_STEPS), MID])
    run.write_band(table, pred, band, str(tmp_path))
    q = pq.read_table(tmp_path / 'predictions_q.parquet').to_pydict()
    for h, m in enumerate(HORIZON_MINUTES):
        np.testing.assert_array_equal(q[f'pred_{m}'], pred[:, h])
        for k, tau in enumerate(QUANTILE_LEVELS):
            col = np.array(q[f'q{round(100 * tau):02d}_{m}'], dtype=float)
            np.testing.assert_array_equal(col, band[:, HORIZON_STEPS[h], k])
    d = pq.read_table(tmp_path / 'predictions_dense.parquet').to_pydict()
    assert d['date'] == q['date']
    for s in range(ZONE):
        m = 5 * (s + 1)
        np.testing.assert_array_equal(np.array(d[f'pred_{m}'], dtype=float), band[:, s, MID])
        for k, tau in enumerate(QUANTILE_LEVELS):
            np.testing.assert_array_equal(np.array(d[f'q{round(100 * tau):02d}_{m}'], dtype=float),
                                          band[:, s, k])
