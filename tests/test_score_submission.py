import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import score_submission as ss


def test_nearest_takes_the_closest_reading_within_half_a_step():
    ts = np.array([0, 300, 600, 1500], dtype=np.int64)
    cgm = np.array([100.0, 110.0, 120.0, 150.0])
    got = ss._nearest(ts, cgm, np.array([290, 449, 451, 1000, 1500, 9000], dtype=np.int64))
    np.testing.assert_array_equal(got[[0, 1, 2, 4]], [110.0, 110.0, 120.0, 150.0])
    assert np.isnan(got[3]) and np.isnan(got[5])


def test_truth_is_the_reading_that_many_minutes_after_the_row(tmp_path):
    t0 = np.datetime64('2024-01-01T00:00:00')
    n = 40
    dates = t0 + np.arange(n) * np.timedelta64(5, 'm')
    cgm = 100.0 + np.arange(n, dtype=np.float64)
    cgm[10] = np.nan
    truth_path = str(tmp_path / 'test.parquet')
    pq.write_table(pa.table({
        'source_file': ['S'] * n, 'id': ['7'] * n,
        'date': pa.array(dates.astype('datetime64[ns]')), 'CGM': cgm}), truth_path)
    rows = pa.table({
        'source_file': ['S', 'S', 'Other'], 'id': ['7', '7', '7'],
        'date': pa.array(dates[[0, 4, 0]].astype('datetime64[ns]'))})
    got = ss.truth_at_horizons(rows, truth_path)
    np.testing.assert_array_equal(got[0], [106.0, 112.0, 118.0, 124.0])
    assert np.isnan(got[1, 0]) and got[1, 1] == 116.0
    assert np.isnan(got[2]).all()


def test_metrics_skip_rows_without_truth_or_prediction():
    pred = np.array([100.0, 200.0, np.nan, 150.0])
    truth = np.array([110.0, np.nan, 120.0, 150.0])
    r = ss.horizon_metrics(pred, truth)
    assert r['n'] == 2
    np.testing.assert_allclose(r['rmse'], np.sqrt(50.0))
    np.testing.assert_allclose(r['mard'], 100.0 * (10.0 / 110.0) / 2)
    assert ss.horizon_metrics(np.array([np.nan]), np.array([100.0]))['n'] == 0
