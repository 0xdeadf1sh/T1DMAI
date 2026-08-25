"""Offline report builders; ``metrics.scoring`` is a pure scoring-rule library — no model, no I/O.

``scoring`` is exposed lazily so importing a sibling costs nothing extra.
"""
from __future__ import annotations

__all__ = ['scoring']


def __getattr__(name: str):
    if name == 'scoring':
        from . import scoring as _scoring
        return _scoring
    raise AttributeError(f"module 'metrics' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
