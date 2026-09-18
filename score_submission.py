"""Score a leaderboard predictions.parquet against MetaboNet's test.parquet.

DTS zone-A share, RMSE and MARD of ``pred_30/60/90/120`` on every row whose truth — the CGM
reading that many minutes after the row's ``date`` — was measured. Median line, mg/dL.
"""

import argparse

import numpy as np

HORIZON_MINUTES = (30, 60, 90, 120)
STEP_S = 300
KEYS = ("source_file", "id", "date")


def _epoch_s(col) -> np.ndarray:
    return col.to_numpy(zero_copy_only=False).astype("datetime64[s]").astype(np.int64)


def _subject_keys(table) -> np.ndarray:
    src = np.asarray(table.column("source_file").to_pylist(), dtype=object)
    sid = np.asarray([str(x) for x in table.column("id").to_pylist()], dtype=object)
    # \x1f not \x00: <U dtype strips trailing NUL, so ('Loop','2x')/('Loop2','x') would collide.
    return np.char.add(np.char.add(src.astype(str), "\x1f"), sid.astype(str))


def load_truth(truth_path: str, wanted: set[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """``{subject key: (epoch s ascending, CGM mg/dL)}`` over the measured readings."""
    import pyarrow.parquet as pq

    rows: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    pf = pq.ParquetFile(truth_path)
    for g in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(g, columns=[*KEYS, "CGM"])
        keys = _subject_keys(t)
        ts = _epoch_s(t.column("date"))
        cgm = t.column("CGM").to_numpy(zero_copy_only=False).astype(np.float64)
        # A subject's rows are not always one contiguous block, so every run is appended.
        cuts = np.flatnonzero(keys[1:] != keys[:-1]) + 1
        for s, e in zip(np.concatenate([[0], cuts]), np.concatenate([cuts, [len(keys)]])):
            if keys[s] in wanted:
                rows.setdefault(str(keys[s]), []).append((ts[s:e], cgm[s:e]))
    truth = {}
    for key, parts in rows.items():
        ts = np.concatenate([p[0] for p in parts])
        cgm = np.concatenate([p[1] for p in parts])
        have = np.isfinite(cgm)
        order = np.argsort(ts[have], kind="stable")
        truth[key] = (ts[have][order], cgm[have][order])
    return truth


def _nearest(ts: np.ndarray, cgm: np.ndarray, target: np.ndarray) -> np.ndarray:
    """The reading nearest each target time, NaN when none lies within half a step."""
    out = np.full(len(target), np.nan)
    if not len(ts):
        return out
    hi = np.clip(np.searchsorted(ts, target), 0, len(ts) - 1)
    lo = np.clip(hi - 1, 0, len(ts) - 1)
    pick = np.where(np.abs(ts[hi] - target) < np.abs(ts[lo] - target), hi, lo)
    ok = np.abs(ts[pick] - target) <= STEP_S // 2
    out[ok] = cgm[pick[ok]]
    return out


def truth_at_horizons(pred_table, truth_path: str) -> np.ndarray:
    """``(rows, horizons)`` CGM at each row's ``date`` plus the horizon; NaN where unmeasured."""
    keys = _subject_keys(pred_table)
    ts = _epoch_s(pred_table.column("date"))
    truth = load_truth(truth_path, set(keys.tolist()))
    out = np.full((len(keys), len(HORIZON_MINUTES)), np.nan)
    order = np.argsort(keys, kind="stable")
    cuts = np.flatnonzero(keys[order][1:] != keys[order][:-1]) + 1
    for rows in np.split(order, cuts):
        entry = truth.get(str(keys[rows[0]]))
        if entry is None:
            continue
        for h, m in enumerate(HORIZON_MINUTES):
            out[rows, h] = _nearest(*entry, ts[rows] + m * 60)
    return out


def horizon_metrics(pred: np.ndarray, truth: np.ndarray) -> dict:
    """DTS-A %, RMSE mg/dL, MARD % over the rows where both are finite and truth is positive."""
    import dts_grid
    from T1DMSIM.simulator import BG_CLAMP_MIN

    ok = np.isfinite(pred) & np.isfinite(truth) & (truth > 0.0)
    p, t = pred[ok], truth[ok]
    if not len(t):
        return {"n": 0, "dts_a": None, "rmse": None, "mard": None}
    # Floor-only: the ceiling half would collapse truth above 400 and score it zone A.
    counts = dts_grid.dts_zone_counts(np.clip(t, BG_CLAMP_MIN, None), p)
    frac = dts_grid.dts_zone_fractions(counts)["a"]
    return {
        "n": int(len(t)),
        "dts_a": None if frac is None else 100.0 * float(frac),
        "rmse": float(np.sqrt(np.mean((p - t) ** 2))),
        "mard": 100.0 * float(np.mean(np.abs(p - t) / t)),
    }


def _cell(v: float | None, width: int) -> str:
    return f"{'—':>{width}}" if v is None else f"{v:>{width}.2f}"


def score_table(pred_table, truth_path: str, by_source: bool = False) -> dict:
    """Print and return ``{minutes: metrics}``, plus ``by_source`` when asked."""
    truth = truth_at_horizons(pred_table, truth_path)
    preds = {
        m: pred_table.column(f"pred_{m}").to_numpy(zero_copy_only=False).astype(np.float64)
        for m in HORIZON_MINUTES
    }
    out: dict = {
        m: horizon_metrics(preds[m], truth[:, h]) for h, m in enumerate(HORIZON_MINUTES)
    }
    print(f"scores against {truth_path} (median line):")
    print("  min    DTS-A%     RMSE    MARD%        n")
    for m in HORIZON_MINUTES:
        r = out[m]
        print(f"  {m:<5}{_cell(r['dts_a'], 8)} {_cell(r['rmse'], 8)} {_cell(r['mard'], 8)} "
              f"{r['n']:8d}")
    if by_source:
        src = np.asarray(pred_table.column("source_file").to_pylist(), dtype=object)
        out["by_source"] = {}
        print(f"  {'sub-dataset':<16}{'n@' + str(HORIZON_MINUTES[-1]):>8}  "
              + "".join(f"{'A%@' + str(m):>8}" for m in HORIZON_MINUTES)
              + "".join(f"{'RMSE@' + str(m):>9}" for m in HORIZON_MINUTES))
        for s in sorted(set(src.tolist())):
            sel = src == s
            rs = {m: horizon_metrics(preds[m][sel], truth[sel, h])
                  for h, m in enumerate(HORIZON_MINUTES)}
            out["by_source"][s] = rs
            print(f"  {s:<16}{rs[HORIZON_MINUTES[-1]]['n']:>8}  "
                  + "".join(_cell(rs[m]["dts_a"], 8) for m in HORIZON_MINUTES)
                  + "".join(_cell(rs[m]["rmse"], 9) for m in HORIZON_MINUTES))
    return out


def main() -> None:
    import pyarrow.parquet as pq

    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("predictions", metavar="predictions.parquet")
    p.add_argument("--truth", default="metabonet/test.parquet", metavar="test.parquet")
    p.add_argument("--by-source", action="store_true", help="also one row per sub-dataset")
    args = p.parse_args()
    score_table(pq.read_table(args.predictions), args.truth, args.by_source)


if __name__ == "__main__":
    main()
