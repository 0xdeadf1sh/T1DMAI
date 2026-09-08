"""Continuous Glucose-Error Grid Analysis (CG-EGA), vectorized numpy, mg/dL.
Port of dotXem/CG-EGA (Kovatchev et al. 2004), not the published paper: deviates on three
points, under-reporting danger, enumerated in ../T1DMCOMMON/SPEC/invariants.md §6.3.
Region and every axis key off y_true — argument order is load-bearing, do not transpose.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "cg_ega_rates",
    "cg_ega_marks",
    "cg_ega_counts",
    "cg_ega_fractions",
]

# AP/BE/EP filter matrices (8 R-marks x region P-cols), VERBATIM; rows [A,B,uC,lC,uD,lD,uE,lE].
_FILTER_AP_HYPO = np.array(
    [[1, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0],
     [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=bool)
_FILTER_BE_HYPO = np.array(
    [[0, 0, 0], [0, 0, 0], [1, 0, 0], [1, 0, 0],
     [0, 0, 0], [1, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=bool)

_FILTER_AP_EU = np.array(
    [[1, 1, 0], [1, 1, 0], [0, 0, 0], [0, 0, 0],
     [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=bool)
_FILTER_BE_EU = np.array(
    [[0, 0, 0], [0, 0, 0], [1, 1, 0], [1, 1, 0],
     [1, 1, 0], [1, 1, 0], [0, 0, 0], [0, 0, 0]], dtype=bool)

_FILTER_AP_HYPER = np.array(
    [[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
    dtype=bool)
_FILTER_BE_HYPER = np.array(
    [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0],
     [1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
    dtype=bool)

# P-mark columns each region's filters use: hypo[A,D,E]->0,3,4; eu[A,B,C]->0,1,2; hyper->all 5.
_REGION_P_COLS: dict[str, list[int]] = {
    "hypo": [0, 3, 4],
    "eu": [0, 1, 2],
    "hyper": [0, 1, 2, 3, 4],
}
_REGION_FILTERS: dict[str, tuple[np.ndarray, np.ndarray]] = {
    "hypo": (_FILTER_AP_HYPO, _FILTER_BE_HYPO),
    "eu": (_FILTER_AP_EU, _FILTER_BE_EU),
    "hyper": (_FILTER_AP_HYPER, _FILTER_BE_HYPER),
}

# Per-region AP/BE/EP over the full (8x5) grid: 0=AP, 1=BE, 2=EP (P-mark outside filters = EP).
_LABEL_AP, _LABEL_BE, _LABEL_EP = 0, 1, 2


def _region_label_table(region: str) -> np.ndarray:
    """(8, 5) AP/BE/EP code table for one region (0=AP, 1=BE, 2=EP)."""
    f_ap, f_be = _REGION_FILTERS[region]
    cols = _REGION_P_COLS[region]
    table = np.full((8, 5), _LABEL_EP, dtype=np.int8)
    for j, p in enumerate(cols):
        table[:, p] = np.where(f_ap[:, j], _LABEL_AP,
                               np.where(f_be[:, j], _LABEL_BE, _LABEL_EP))
    return table


_LABEL_TABLE: dict[str, np.ndarray] = {
    r: _region_label_table(r) for r in ("hypo", "eu", "hyper")
}

# int8 region codes 0/1/2 ↔ their public string names
_NAMES = np.array(["hypo", "eu", "hyper"])


def cg_ega_rates(y: np.ndarray, last_bg: np.ndarray,
                 freq_min: float = 5.0) -> np.ndarray:
    """Per-step rate of change dy (mg/dL/min), anchored at last_bg.

    dy[:, t] = (y[:, t] - y[:, t-1]) / freq_min, with y[:, -1] := last_bg.
    y (N,T) mg/dL; last_bg (N,) mg/dL; returns dy (N,T) mg/dL/min.
    """
    y = np.asarray(y, dtype=np.float64)
    last_bg = np.asarray(last_bg, dtype=np.float64)
    assert y.ndim == 2, f"expected (N, T) y, got shape {y.shape}"
    assert last_bg.shape == (y.shape[0],), (
        f"last_bg must be (N,) matching y, got {last_bg.shape} vs {y.shape}"
    )
    prev = np.concatenate([last_bg[:, None], y[:, :-1]], axis=1)  # (N, T)
    return (y - prev) / freq_min


def _p_ega_marks(y_true: np.ndarray, y_pred: np.ndarray,
                 dy_true: np.ndarray) -> np.ndarray:
    """P-EGA single mark per point, argmax over [A,B,C,D,E] (first max wins).

    y_true/y_pred (M,) mg/dL; dy_true (M,) mg/dL/min drives mod. Returns (M,) int in {0..4}.
    """
    # rate-of-change widening of the acceptance bands (mg/dL/min):
    mod = np.zeros_like(y_true)
    mod[((dy_true > -2) & (dy_true <= -1)) | ((dy_true >= 1) & (dy_true < 2))] = 10.0
    mod[(dy_true <= -2) | (dy_true >= 2)] = 20.0

    A = (((y_pred <= 70 + mod) & (y_true <= 70))
         | ((y_pred <= y_true * 6 / 5 + mod) & (y_pred >= y_true * 4 / 5 - mod)))

    E = (((y_true > 180) & (y_pred < 70 - mod))
         | ((y_pred > 180 + mod) & (y_true <= 70)))

    D = (((y_pred > 70 + mod) & (y_pred > y_true * 6 / 5 + mod)
          & (y_true <= 70) & (y_pred <= 180 + mod))
         | ((y_true > 240) & (y_pred < 180 - mod) & (y_pred >= 70 - mod)))

    C = (((y_true > 70) & (y_pred > y_true * 22 / 17 + (180 - 70 * 22 / 17) + mod))
         | ((y_true <= 180) & (y_pred < y_true * 7 / 5 - 182 - mod)))

    B = ~(A | C | D | E)

    stack = np.stack([A, B, C, D, E], axis=0)  # (5, M)
    return np.argmax(stack, axis=0)


def _r_ega_marks(dy_true: np.ndarray, dy_pred: np.ndarray) -> np.ndarray:
    """R-EGA single mark per point, argmax over 8 marks (first max wins).

    Mark order [A,B,uC,lC,uD,lD,uE,lE]. dy_true/dy_pred (M,) mg/dL/min; returns (M,) int {0..7}.
    """
    A = (((dy_pred >= dy_true - 1) & (dy_pred <= dy_true + 1))
         | ((dy_pred <= dy_true / 2) & (dy_pred >= dy_true * 2))
         | ((dy_pred <= dy_true * 2) & (dy_pred >= dy_true / 2)))

    B = (~A) & (
        ((dy_pred <= -1) & (dy_true <= -1))
        | ((dy_pred <= dy_true + 2) & (dy_pred >= dy_true - 2))
        | ((dy_pred >= 1) & (dy_true >= 1))
    )

    uC = (dy_true < 1) & (dy_true >= -1) & (dy_pred > dy_true + 2)
    lC = (dy_true <= 1) & (dy_true > -1) & (dy_pred < dy_true - 2)
    uD = (dy_pred <= 1) & (dy_pred >= -1) & (dy_pred > dy_true + 2)
    lD = (dy_pred <= 1) & (dy_pred >= -1) & (dy_pred < dy_true - 2)
    uE = (dy_pred > 1) & (dy_true < -1)
    lE = (dy_pred < -1) & (dy_true > 1)

    stack = np.stack([A, B, uC, lC, uD, lD, uE, lE], axis=0)  # (8, M)
    return np.argmax(stack, axis=0)


def cg_ega_marks(y_true: np.ndarray, y_pred: np.ndarray, last_bg: np.ndarray,
                 freq_min: float = 5.0) -> dict:
    """P-EGA and R-EGA marks + region for every (N,T) point, flattened to (N*T,).

    Returns dict: p_mark int{0..4}->[A,B,C,D,E], r_mark int{0..7}->[A,B,uC,lC,uD,lD,uE,lE],
    region str{hypo,eu,hyper}, region_code int8{0,1,2}, dy_true/dy_pred mg/dL/min.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    last_bg = np.asarray(last_bg, dtype=np.float64)
    assert y_true.shape == y_pred.shape, (
        f"y_true {y_true.shape} and y_pred {y_pred.shape} must match"
    )
    assert y_true.ndim == 2, f"expected (N, T), got shape {y_true.shape}"

    dy_true = cg_ega_rates(y_true, last_bg, freq_min).reshape(-1)
    dy_pred = cg_ega_rates(y_pred, last_bg, freq_min).reshape(-1)
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)

    p_mark = _p_ega_marks(yt, yp, dy_true)
    r_mark = _r_ega_marks(dy_true, dy_pred)

    region_code = np.where(yt <= 70, 0,
                           np.where(yt <= 180, 1, 2)).astype(np.int8)
    return {
        "p_mark": p_mark, "r_mark": r_mark,
        "region": _NAMES[region_code], "region_code": region_code,
        "dy_true": dy_true, "dy_pred": dy_pred,
    }


def cg_ega_counts(y_true: np.ndarray, y_pred: np.ndarray, last_bg: np.ndarray,
                  freq_min: float = 5.0) -> dict:
    """CG-EGA Accurate/Benign/Erroneous counts per glycemic region.

    last_bg anchors dy_true/dy_pred at the step before t=0. Returns 9 ints:
    {ap,be,ep}_{hypo,eu,hyper}.
    """
    marks = cg_ega_marks(y_true, y_pred, last_bg, freq_min)
    p_mark = marks["p_mark"]
    r_mark = marks["r_mark"]
    region_code = marks["region_code"]

    counts: dict[str, int] = {}
    for code, reg in enumerate(("hypo", "eu", "hyper")):
        sel = region_code == code
        labels = _LABEL_TABLE[reg][r_mark[sel], p_mark[sel]]  # (n_reg,) 0/1/2
        counts[f"ap_{reg}"] = int((labels == _LABEL_AP).sum())
        counts[f"be_{reg}"] = int((labels == _LABEL_BE).sum())
        counts[f"ep_{reg}"] = int((labels == _LABEL_EP).sum())
    return counts


def cg_ega_fractions(counts: dict) -> dict:
    """Per-region %AP/%BE/%EP from the 9 integer counts.

    Fractions in [0, 1], NOT percent. A region with zero total points yields None for all
    three of its keys.
    """
    out: dict[str, float | None] = {}
    for reg in ("hypo", "eu", "hyper"):
        ap = counts.get(f"ap_{reg}", 0)
        be = counts.get(f"be_{reg}", 0)
        ep = counts.get(f"ep_{reg}", 0)
        total = ap + be + ep
        if total > 0:
            out[f"ap_{reg}"] = ap / total
            out[f"be_{reg}"] = be / total
            out[f"ep_{reg}"] = ep / total
        else:
            out[f"ap_{reg}"] = None
            out[f"be_{reg}"] = None
            out[f"ep_{reg}"] = None
    return out
