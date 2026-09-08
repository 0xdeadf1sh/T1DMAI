"""The bit-identity gate: a checked-in reference, and the round trip.

bitident_ref.json freezes one forward; refreeze only when its config moves, never
to turn a red test green. Gated at max|delta|==0.0 on head_raw/q_tau/median and the
attention mask over ALL rows — a misopened pad row is invisible to an output-only check."""

import base64
import importlib.util
import json
import tempfile
import warnings
from pathlib import Path

import pytest
import torch

import config

BITIDENT = Path(__file__).resolve().parent.parent / "scratch" / "bitident.py"
REFERENCE = Path(__file__).resolve().parent / "bitident_ref.json"

# scratch/ is gitignored; skip+warn not raise — raise ends the session at zero tests.
if not BITIDENT.exists():
    _ABSENT = (
        "BIT-IDENTITY GATE OFF -- the forward is UNGATED. Its implementation "
        f"{BITIDENT} is absent, and being under gitignored scratch/ it is in no "
        "commit, so git cannot restore it: copy it in from a working tree that "
        "has it. Nothing in this repository regenerates it, and the tracked "
        "reference beside this file can be neither compared against nor "
        "refrozen without it."
    )
    warnings.warn(_ABSENT, stacklevel=1)
    pytest.skip(_ABSENT, allow_module_level=True)

_spec = importlib.util.spec_from_file_location("bitident_gate", BITIDENT)
bitident = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bitident)

# Spans must clear MAX_CONTEXT_PATCHES-MIN (shortest row's pad); derived, never hardcoded.
_INFILL_FLOOR = config.MAX_CONTEXT_PATCHES - config.MIN_CONTEXT_PATCHES
INFILL_SPANS = [(_INFILL_FLOOR + 6, 2), (_INFILL_FLOOR + 12, 3)]

# batch shape/content aren't in capacity(); changing them moves inputs, not arithmetic.
GEOMETRY_KEYS = (
    "PATCH_SIZE",
    "N_INPUT_FEATURES",
    "MIN_CONTEXT_PATCHES",
    "MAX_CONTEXT_PATCHES",
    "PREDICTION_PATCHES",
)

_DTYPES = {"float32": torch.float32, "int64": torch.int64, "bool": torch.bool}


def _encode(t: torch.Tensor) -> dict:
    return {
        "dtype": str(t.dtype).removeprefix("torch."),
        "shape": list(t.shape),
        "b64": base64.b64encode(t.contiguous().numpy().tobytes()).decode("ascii"),
    }


def _decode(d: dict) -> torch.Tensor:
    raw = bytearray(base64.b64decode(d["b64"]))
    return torch.frombuffer(raw, dtype=_DTYPES[d["dtype"]]).reshape(d["shape"])


def _stamp() -> dict:
    """What the reference must have been frozen at to be comparable.

    Capacity and the masked set are ``compare``'s to stamp and reject, not a second
    copy here.
    """
    return {
        "geometry": {k: int(getattr(config, k)) for k in GEOMETRY_KEYS},
        "seed": bitident.SEED,
        "torch": torch.__version__,
    }


def refreeze(path: Path = REFERENCE) -> Path:
    """Freeze at the current config and write the tracked reference."""
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as d:
        pt = Path(d) / "ref.pt"
        assert bitident.freeze(pt) == 0
        payload = torch.load(pt, weights_only=False)
    doc = {
        "stamp": _stamp(),
        "payload": {
            "capacity": payload["capacity"],
            "seed": payload["seed"],
            "spans": [list(s) for s in payload["spans"]],
            "alibi_params_zeroed": payload["alibi_params_zeroed"],
            "tensors": {
                k: _encode(v) for k, v in payload.items() if torch.is_tensor(v)
            },
        },
    }
    path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(f"[FREEZE] {path}  stamp {doc['stamp']}  capacity {payload['capacity']}")
    return path


def _materialise(doc: dict, path: Path) -> Path:
    """The reference in the form ``compare`` reads."""
    ref = doc["payload"]
    payload = {k: _decode(v) for k, v in ref["tensors"].items()}
    payload["capacity"] = ref["capacity"]
    payload["seed"] = ref["seed"]
    payload["spans"] = [tuple(s) for s in ref["spans"]]
    payload["alibi_params_zeroed"] = ref["alibi_params_zeroed"]
    torch.save(payload, path)
    return path


def _flat(stamp: dict) -> dict:
    """One level down, so a drift names the constant rather than two dicts."""
    flat = {}
    for key, value in stamp.items():
        flat.update(value if isinstance(value, dict) else {key: value})
    return flat


def _unanswerable(doc: dict) -> str | None:
    """Why the reference cannot speak to today's forward, or None."""
    want, have = _flat(doc["stamp"]), _flat(_stamp())
    moved = {k: (want.get(k), v) for k, v in have.items() if want.get(k) != v}
    if not moved:
        return None
    return ", ".join(f"{k} {w!r} -> {c!r}" for k, (w, c) in sorted(moved.items()))


@pytest.fixture(scope="module", autouse=True)
def pinned():
    """One thread, and the global RNG left as it was found.

    Attention matmul reduction order is thread-count dependent, so freeze/compare
    at different counts aren't bitwise comparable. ``build_model`` reseeds the
    global generator, forked here so no other module inherits it."""
    prior = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            yield
    finally:
        torch.set_num_threads(prior)


@pytest.fixture(scope="module")
def frozen(pinned, tmp_path_factory):
    """The reference, frozen once at the default (right-edge) masked set."""
    path = tmp_path_factory.mktemp("bitident") / "ref.pt"
    assert bitident.freeze(path) == 0
    return path


def test_forward_matches_the_checked_in_reference(pinned, tmp_path):
    """The round trip below builds both sides from the code in front of it, so a
    forward shifting every output by a constant passes it."""
    doc = json.loads(REFERENCE.read_text())
    print(f"[DUMP] reference {REFERENCE} stamp {doc['stamp']}")

    moved = _unanswerable(doc)
    if moved is not None:
        reason = (
            f"the checked-in reference cannot answer this config -- {moved}. "
            "Refreeze with `venv/bin/python -m tests.test_bitident` once the "
            "move is intended; the forward is UNGATED until you do."
        )
        warnings.warn(reason, stacklevel=1)
        pytest.xfail(reason)

    rc = bitident.compare(_materialise(doc, tmp_path / "ref.pt"))
    if rc == 2:
        reason = (
            "the checked-in reference is at another capacity or another masked "
            "set; refreeze with `venv/bin/python -m tests.test_bitident`"
        )
        warnings.warn(reason, stacklevel=1)
        pytest.xfail(reason)
    assert rc == 0, "the forward moved since the reference was frozen"


def test_round_trip_is_bit_identical(frozen):
    ref = torch.load(frozen, weights_only=False)
    assert ref["capacity"] == bitident.capacity()

    batch = bitident.build_batch()
    model, _ = bitident.build_model()
    out = bitident.run(model, batch)

    for key in bitident.GATED:
        delta = (out[key].double() - ref[key].double()).abs().max().item()
        print(f"[DUMP] max|delta {key}| = {delta:.6e}")
        assert delta == 0.0, f"{key} is not bit-identical: max|delta| = {delta:.6e}"

    assert torch.equal(batch.attn, ref["attn_mask"])
    assert torch.equal(batch.patches, ref["patches"])
    assert bitident.compare(frozen) == 0


def test_mask_matches_over_padded_rows(frozen):
    """The pad rows agree, and they are closed.

    A head_raw comparison cannot see either: freeze and compare build the mask
    through the same call, so the round trip gates reproducibility, this the rule.
    """
    ref = torch.load(frozen, weights_only=False)
    batch = bitident.build_batch()
    attn, is_pad = ref["attn_mask"], batch.is_pad
    T = attn.shape[-1]

    assert int(is_pad.sum()) > 0, "no padded position, so this test is vacuous"
    assert torch.equal(batch.attn[is_pad], attn[is_pad])

    diag = torch.eye(T, dtype=torch.bool)
    off = attn & ~diag
    assert not bool(off[is_pad].any()), "a pad row reads more than itself"
    assert not bool(off.transpose(1, 2)[is_pad].any()), "a pad column is read"
    assert bool(attn.diagonal(dim1=1, dim2=2).all()), "an all-False row NaNs softmax"


def test_masked_set_is_a_parameter(pinned, tmp_path):
    """The gate holds at an infill masked set, not only at the right edge."""
    path = tmp_path / "infill.pt"
    assert bitident.freeze(path, INFILL_SPANS) == 0
    ref = torch.load(path, weights_only=False)
    assert [tuple(s) for s in ref["spans"]] == INFILL_SPANS
    start, length = INFILL_SPANS[0]
    assert not bool(ref["visible"][:, start : start + length].any())
    assert bitident.compare(path, INFILL_SPANS) == 0


def test_compare_rejects_a_perturbed_reference(frozen, tmp_path):
    """A reference perturbed in head_raw fails, and fails non-zero."""
    ref = torch.load(frozen, weights_only=False)
    ref["head_raw"] = ref["head_raw"] + 1e-7
    path = tmp_path / "perturbed.pt"
    torch.save(ref, path)
    assert bitident.compare(path) == 1


def test_compare_rejects_a_foreign_reference(frozen, tmp_path):
    """Another capacity or masked set is rejected, never compared: let through, both
    report a large delta, reading as a broken change rather than a stale reference."""
    ref = torch.load(frozen, weights_only=False)
    ref["capacity"] = dict(ref["capacity"], D_MODEL=ref["capacity"]["D_MODEL"] * 2)
    path = tmp_path / "wrong_capacity.pt"
    torch.save(ref, path)
    assert bitident.compare(path) == 2
    assert bitident.compare(frozen, INFILL_SPANS) == 2


def test_encoding_is_bit_exact(frozen):
    """The tracked format loses no bit of any tensor it carries."""
    for t in torch.load(frozen, weights_only=False).values():
        if torch.is_tensor(t):
            assert torch.equal(_decode(_encode(t)), t)


if __name__ == "__main__":
    refreeze()
