"""Offline metric scripts, and the scoring rules they share.

Most entries here are stand-alone report builders run from ``rebuild_all.sh``
against a checkpoint.  ``metrics.scoring`` is different: it is a pure library of
proper scoring rules over a decoded quantile fan, imported rather than run, and
holding no model, no I/O and no gate.

The submodule is exposed lazily so that importing any sibling (``metrics.figstyle``,
``metrics.core.suite``) still costs nothing beyond what that sibling itself needs.
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
