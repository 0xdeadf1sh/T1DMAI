"""Render a leaderboard predictions.parquet from a finetuned checkpoint.

Keys, row order and dtypes are the template's; a horizon short of full coverage
is written all-null. Built at the CHECKPOINT's architecture, not config.py's.
"""

import argparse
import sys

import numpy as np
import torch

from finetune import _apply_checkpoint_dims

FILL_NONE = "none"
FILL_LADDER = "ladder"
DOSES_ANNOUNCED = "announced"
DOSES_ZERO = "zero"

# Last resort of the ladder; also what a masked patch's zeroed bg input decodes to.
_FALLBACK_BG_MGDL = 120.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", default="checkpoints_finetune/finetune_best.pt")
    p.add_argument("--template", required=True, metavar="template.parquet")
    p.add_argument("--cache", default="metabonet/cache_finetune")
    p.add_argument("--out", default="predictions.parquet")
    p.add_argument(
        "--max-context",
        type=int,
        default=168,
        help="context patches per row, capped again by the history the row has",
    )
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-interp-steps", type=int, default=1)
    p.add_argument(
        "--future-doses",
        choices=(DOSES_ANNOUNCED, DOSES_ZERO),
        default=DOSES_ANNOUNCED,
        help="carb/insulin/exercise over the forecast zone; 'zero' is strictly "
        "causal but out of distribution for a model trained on announced doses",
    )
    p.add_argument(
        "--fill",
        choices=(FILL_NONE, FILL_LADDER),
        default=FILL_NONE,
        help="unpredictable rows: 'none' nulls the whole column, 'ladder' falls "
        "back to last measured bg, then subject median, then 120 mg/dL",
    )
    p.add_argument("--limit", type=int, default=None, help="first N template rows")
    return p.parse_args()


def _read_template(path: str) -> tuple[object, np.ndarray, np.ndarray, np.ndarray]:
    """``(table, source_file, id, epoch_seconds)``; the table is written back into."""
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    missing = {"id", "source_file", "date"} - set(table.schema.names)
    if missing:
        sys.exit(f"--template {path}: missing column(s) {sorted(missing)}")
    src = np.asarray(table.column("source_file").to_pylist(), dtype=object)
    sid = np.asarray(table.column("id").to_pylist(), dtype=object)
    date = np.asarray(table.column("date").to_numpy(zero_copy_only=False))
    return table, src, sid, date.astype("datetime64[s]").astype(np.int64)


def _plan(
    cache: object,
    rows_by_subject: dict[int, list[int]],
    t_epoch: np.ndarray,
    args: argparse.Namespace,
    patch_size: int,
) -> dict[str, np.ndarray]:
    """Anchorable template rows as parallel ``row``/``subject``/``origin``/``n_ctx``."""
    from finetune_data import _interp_short_gaps

    cols: dict[str, list[np.ndarray]] = {
        k: [] for k in ("row", "subject", "origin", "n_ctx")
    }
    for s, rows in rows_by_subject.items():
        rec = cache.subjects[s]
        n = int(rec["n"])
        idx = np.asarray(rows, dtype=np.int64)
        origin = np.rint((t_epoch[idx] - int(rec["t0"])) / 300.0).astype(np.int64)
        keep = (origin >= patch_size) & (origin <= n)
        idx, origin = idx[keep], origin[keep]
        if not len(idx):
            continue
        bg = _interp_short_gaps(
            np.asarray(cache.channels(rec, 0, n)["bg"], dtype=np.float32),
            args.max_interp_steps,
        )
        c = np.concatenate([[0], np.cumsum(np.isfinite(bg).astype(np.int64))])
        # fin[p]: steps p..p+patch_size-1 all measured, i.e. one whole readable patch.
        fin = (c[patch_size:] - c[:-patch_size]) == patch_size
        anchored = fin[origin - patch_size]
        idx, origin = idx[anchored], origin[anchored]
        if not len(idx):
            continue
        cols["row"].append(idx)
        cols["subject"].append(np.full(len(idx), s, dtype=np.int64))
        cols["origin"].append(origin)
        cols["n_ctx"].append(
            np.minimum(args.max_context, origin // patch_size).astype(np.int64)
        )
    if not cols["row"]:
        return {k: np.empty(0, dtype=np.int64) for k in cols}
    out = {k: np.concatenate(v) for k, v in cols.items()}
    # Grouping equal-length windows keeps collate's left pad near zero.
    order = np.lexsort((out["origin"], out["subject"], out["n_ctx"]))
    return {k: v[order] for k, v in out.items()}


def _slice_padded(cache: object, rec: dict, lo: int, hi: int) -> dict[str, np.ndarray]:
    """``cache.channels`` clamped to the subject: past its end reads the NEXT subject."""
    n = int(rec["n"])
    a, b = max(lo, 0), min(hi, n)
    inner = cache.channels(rec, a, b)
    if a == lo and b == hi:
        return inner
    out: dict[str, np.ndarray] = {}
    for c, v in inner.items():
        fill = np.nan if c == "bg" else 0.0
        out[c] = np.concatenate(
            [
                np.full(a - lo, fill, dtype=v.dtype),
                np.asarray(v),
                np.full(hi - b, fill, dtype=v.dtype),
            ]
        )
    return out


class _ExportDataset(torch.utils.data.Dataset):
    """Right-edge forecast windows at template timestamps; no truth rides along."""

    def __init__(
        self,
        cache: object,
        stats: dict[str, dict[str, float]],
        jobs: dict[str, np.ndarray],
        args: argparse.Namespace,
        patch_size: int,
        prediction_patches: int,
    ) -> None:
        self.cache = cache
        self.stats = stats
        self.jobs = jobs
        self.max_interp_steps = args.max_interp_steps
        self.future_doses = args.future_doses
        self.patch_size = patch_size
        self.prediction_patches = prediction_patches

    def __len__(self) -> int:
        return len(self.jobs["row"])

    def __getitem__(self, i: int) -> dict:
        from finetune_data import (
            _BG_GAP_FILL_MGDL,
            _assemble_sample,
            _interp_short_gaps,
            _normalize_features,
        )

        ps, pp = self.patch_size, self.prediction_patches
        rec = self.cache.subjects[int(self.jobs["subject"][i])]
        origin = int(self.jobs["origin"][i])
        n_ctx = int(self.jobs["n_ctx"][i])
        seq_len = n_ctx + pp
        ch = _slice_padded(self.cache, rec, origin - n_ctx * ps, origin + pp * ps)

        bg = _interp_short_gaps(ch["bg"].astype(np.float32), self.max_interp_steps)
        # The zone's own bg is the answer; it must not reach the input by any route.
        bg[n_ctx * ps :] = np.nan
        carb, ins, ex = ch["carb"], ch["insulin"], ch["exercise"]
        if self.future_doses == DOSES_ZERO:
            carb, ins, ex = carb.copy(), ins.copy(), ex.copy()
            for a in (carb, ins, ex):
                a[n_ctx * ps :] = 0.0

        visible = np.isfinite(bg).reshape(seq_len, ps).all(axis=1)
        visible[n_ctx:] = False
        gap_patches = np.flatnonzero(~visible[:n_ctx]).astype(np.int64)
        feats = _normalize_features(
            np.nan_to_num(bg, nan=_BG_GAP_FILL_MGDL), carb, ins, ex, self.stats
        )
        return _assemble_sample(
            feats,
            np.nan_to_num(bg, nan=_BG_GAP_FILL_MGDL),
            [(n_ctx, pp)],
            gap_patches,
            seq_len,
            n_ctx,
        )


def _apply_ladder(
    pred: np.ndarray,
    cache: object,
    rows_by_subject: dict[int, list[int]],
    t_epoch: np.ndarray,
) -> None:
    """Fill still-NaN rows: last measured bg before the timestamp, subject median, 120."""
    for s, rows in rows_by_subject.items():
        idx = np.asarray(rows, dtype=np.int64)
        idx = idx[idx < len(pred)]
        idx = idx[np.isnan(pred[idx]).any(axis=1)]
        if not len(idx):
            continue
        rec = cache.subjects[s]
        n = int(rec["n"])
        bg = np.asarray(cache.channels(rec, 0, n)["bg"], dtype=np.float64)
        finite = np.isfinite(bg)
        med = float(np.median(bg[finite])) if finite.any() else _FALLBACK_BG_MGDL
        # src[k]: index of the most recent measured step at or before k, else -1.
        src_of = np.maximum.accumulate(np.where(finite, np.arange(n), -1))
        origin = np.clip(
            np.rint((t_epoch[idx] - int(rec["t0"])) / 300.0).astype(np.int64), 0, n - 1
        )
        src = src_of[np.maximum(origin - 1, 0)]
        val = np.where(src >= 0, bg[np.maximum(src, 0)], med)
        val = np.where(np.isfinite(val), val, _FALLBACK_BG_MGDL)
        for h in range(pred.shape[1]):
            col = pred[idx, h]
            pred[idx, h] = np.where(np.isnan(col), val, col)
    rest = np.isnan(pred)
    if rest.any():
        pred[rest] = _FALLBACK_BG_MGDL


def _write(
    table: object, pred: np.ndarray, n_rows: int, horizons: tuple, out_path: str
) -> int:
    """Template table with ``pred_*`` replaced; a column short of full coverage goes null."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    total = table.num_rows
    print("  horizon coverage:", flush=True)
    kept = 0
    for h, m in enumerate(horizons):
        col = f"pred_{m}"
        if col not in table.schema.names:
            sys.exit(f"template has no column {col!r}")
        ok = int(np.isfinite(pred[:n_rows, h]).sum())
        full = ok == total
        kept += int(full)
        print(
            f"    {col:<9} {ok}/{total} ({100.0 * ok / max(total, 1):.2f}%) "
            f"-> {'written' if full else 'ALL-NULL'}",
            flush=True,
        )
        values = (
            pa.array(pred[:, h], type=pa.float64())
            if full
            else pa.nulls(total, type=pa.float64())
        )
        table = table.set_column(table.schema.get_field_index(col), col, values)
    if kept == 0:
        print(
            "  every horizon all-null; the validator needs one fully populated "
            "column. Re-run with --fill ladder.",
            flush=True,
        )
    pq.write_table(table, out_path)
    print(f"wrote {out_path}  {total} rows, {kept}/{len(horizons)} horizons", flush=True)
    return kept


def main() -> None:
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    _apply_checkpoint_dims(ckpt)

    from torch.utils.data import DataLoader

    from config import ARCH_VERSION, PATCH_SIZE, PREDICTION_PATCHES
    from finetune_data import (
        HORIZON_MINUTES,
        HORIZON_STEPS,
        FinetuneCache,
        finetune_collate_fn,
    )
    from model import T1DMAI
    from T1DMSIM.simulator import BG_CLAMP_MAX, BG_CLAMP_MIN
    from utils import kovatchev_f_inv

    if ckpt.get("arch_version") != ARCH_VERSION:
        sys.exit(
            f"--checkpoint {args.checkpoint}: arch_version "
            f"{ckpt.get('arch_version')!r} != {ARCH_VERSION!r}"
        )
    stats = ckpt["normalization_stats"]

    table, t_src, t_sid, t_epoch = _read_template(args.template)
    n_rows = len(t_epoch) if args.limit is None else min(len(t_epoch), args.limit)
    print(f"template: {args.template}  {len(t_epoch)} rows, using {n_rows}", flush=True)

    cache = FinetuneCache(args.cache)
    by_key = {(r["source"], r["sid"]): i for i, r in enumerate(cache.subjects)}
    rows_by_subject: dict[int, list[int]] = {}
    unresolved = 0
    for row in range(n_rows):
        s = by_key.get((str(t_src[row]), str(t_sid[row])))
        if s is None:
            unresolved += 1
        else:
            rows_by_subject.setdefault(s, []).append(row)
    if unresolved:
        print(f"  {unresolved} rows name a subject absent from the cache", flush=True)

    jobs = _plan(cache, rows_by_subject, t_epoch, args, PATCH_SIZE)
    print(
        f"  anchorable {len(jobs['row'])}/{n_rows} "
        f"({100.0 * len(jobs['row']) / max(n_rows, 1):.2f}%)",
        flush=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = T1DMAI()
    model.load_state_dict(ckpt["model_ema_state_dict"], strict=False)
    model.to(device).eval()

    ds = _ExportDataset(cache, stats, jobs, args, PATCH_SIZE, PREDICTION_PATCHES)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=finetune_collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    pred = np.full((len(t_epoch), len(HORIZON_MINUTES)), np.nan, dtype=np.float64)
    pos = 0
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            _, median = model(
                batch["patches"].to(device),
                batch["attn_mask"].to(device),
                batch["anchor_bg"].to(device),
                batch["mask_idx"].to(device),
            )
            B = median.shape[0]
            mgdl = (
                kovatchev_f_inv(median[:, :PREDICTION_PATCHES, :])
                .reshape(B, -1)
                .cpu()
                .numpy()
            )
            rows = jobs["row"][pos : pos + B]
            pos += B
            for h, hs in enumerate(HORIZON_STEPS):
                pred[rows, h] = mgdl[:, hs]
            if bi % 200 == 0:
                print(f"  batch {bi}  {pos}/{len(jobs['row'])}", flush=True)
    assert pos == len(jobs["row"]), f"loader covered {pos} of {len(jobs['row'])}"

    if args.fill == FILL_LADDER:
        _apply_ladder(pred, cache, rows_by_subject, t_epoch)
    np.clip(pred, BG_CLAMP_MIN, BG_CLAMP_MAX, out=pred)
    _write(table, pred, n_rows, HORIZON_MINUTES, args.out)


if __name__ == "__main__":
    main()
