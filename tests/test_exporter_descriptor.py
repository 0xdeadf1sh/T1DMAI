"""Exporter descriptor assembly: the model card's two checkpoint shapes and the
T1DMSERVER sidecar naming contract."""

import json
import os

import torch
import torch.nn as nn

from exporters.descriptor import build_descriptor, build_model_card, deploy_to_server


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(3, 4)


def _pretrained_ckpt() -> dict:
    return {
        "step": 87000,
        "val_history": [
            {"step": 86000, "bg_rmse_30": 99.0},
            {
                "step": 87000,
                "bg_rmse_30": 12.1919, "bg_rmse_60": 22.5888, "bg_rmse_120": 32.0049,
                "evalfix_mard@30": 6.0748, "evalfix_mard@60": 11.5741,
                "evalfix_mard@120": 16.4332,
                "evalfix_clarke_A@30": 98.5, "clarke_AB_pct": 99.9, "tod_mae_h": 0.61,
            },
        ],
    }




def test_model_card_reads_last_val_history_entry():
    card = build_model_card(_TinyModel(), _pretrained_ckpt())
    rm = card["reference_metrics"]
    print(f"[DUMP] sim card: step={card['val_step']} rmse={rm['rmse_mgdl']}")

    assert card["param_count"] == 16
    assert card["val_step"] == 87000, "must read the LAST entry, not the first"
    assert rm["source"] == "sim-validation"
    assert rm["rmse_mgdl"] == [12.1919, 22.5888, 32.0049]
    assert rm["mard_pct"] == [6.0748, 11.5741, 16.4332]
    assert rm["clarke_ab_pct"] == 99.9
    # Absent keys degrade to None rather than aborting the export.
    assert rm["clarke_a_pct"] == [98.5, None, None]
    assert rm["coverage90"] == [None, None, None]






def test_deploy_to_server_pairs_artifact_with_stem_json_sidecar(tmp_path):
    """t1dm-store::refresh_models pairs an artifact with a SIBLING <stem>.json, so
    `large-sim.xnnpack.pte` must deploy alongside `large-sim.xnnpack.json`."""
    src = tmp_path / "build"
    src.mkdir()
    pte = src / "large-sim.xnnpack.pte"
    pte.write_bytes(b"\x00pte-bytes")
    stats = {
        "bg_absolute": {"mean": 0.4285, "std": 1.0598},
        "carb_intake": {"mean": 0.3687, "std": 0.4765},
        "insulin_combined": {"mean": 0.1471, "std": 0.1180},
    }
    desc = build_descriptor(
        model_id="large-sim", engine="executorch_xnnpack_fp32", executorch_version="1.3.1",
        artifact_filename=pte.name, normalization_stats=stats,
    )

    deploy_dir = tmp_path / "data" / "models"
    art, side = deploy_to_server(str(pte), desc, str(deploy_dir))
    print(f"[DUMP] deployed {os.path.basename(art)} + {os.path.basename(side)}")

    assert os.path.basename(art) == "large-sim.xnnpack.pte"
    assert os.path.basename(side) == "large-sim.xnnpack.json"
    assert os.path.splitext(art)[0] + ".json" == side, "sidecar must be the stem-paired json"
    assert open(art, "rb").read() == b"\x00pte-bytes"
    with open(side) as f:
        written = json.load(f)
    assert written["artifact"] == "large-sim.xnnpack.pte"
    assert written["normalization_stats"] == stats


def test_descriptor_kovatchev_block_tracks_the_live_transform():
    """The descriptor is the Rust runtime's SOLE source for f/f_inv; a stale block
    would decode risk space against the wrong parameterization."""
    from utils import _KOVATCHEV_SCALE, _KOVATCHEV_OFFSET, _KOVATCHEV_POWER, kovatchev_f
    from T1DMSIM.simulator import BG_CLAMP_MIN, BG_CLAMP_MAX

    stats = {c: {"mean": 0.0, "std": 1.0}
             for c in ("bg_absolute", "carb_intake", "insulin_combined")}
    kov = build_descriptor(
        model_id="m", engine="e", executorch_version="1.3.1",
        artifact_filename="m.pte", normalization_stats=stats,
    )["kovatchev"]
    print(f"[DUMP] kovatchev={kov}")

    assert kov["SCALE"] == _KOVATCHEV_SCALE
    assert kov["POWER"] == _KOVATCHEV_POWER
    assert kov["OFFSET"] == _KOVATCHEV_OFFSET
    assert kov["BG_CLAMP_MIN"] == BG_CLAMP_MIN and kov["BG_CLAMP_MAX"] == BG_CLAMP_MAX
    f_lo = float(kovatchev_f(torch.tensor([float(BG_CLAMP_MIN)])).item())
    f_hi = float(kovatchev_f(torch.tensor([float(BG_CLAMP_MAX)])).item())
    assert abs(kov["RISK_CLAMP_MIN"] - f_lo) < 1e-6
    assert abs(kov["RISK_CLAMP_MAX"] - f_hi) < 1e-6
