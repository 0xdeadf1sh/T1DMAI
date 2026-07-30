"""Real-data adapters: parse OhioT1DM / AZT1D / ShanghaiT1DM / a personal record into
canonical Segments."""
from __future__ import annotations

from .schema import Segment, GRID_MIN, MGDL_PER_MMOL

__all__ = ['Segment', 'GRID_MIN', 'MGDL_PER_MMOL', 'load_dataset']


def load_dataset(name: str, root_dir: str | None = None) -> list[Segment]:
    """Load one dataset by name into a list of Segments.

    Args:
        name: one of ``ohiot1dm``, ``azt1d``, ``shanghai``, ``personal``.
        root_dir: optional override of the dataset's location on disk. Required for
            ``personal``, whose location is a SQLite file rather than a directory and
            has no default (a personal record's path is never baked into the repo).
    """
    name = name.lower()
    if name == 'personal':
        from . import personal
        assert root_dir is not None, "load_dataset('personal') requires root_dir=<db path>"
        return personal.load(root_dir)
    if name in ('ohio', 'ohiot1dm'):
        from . import ohio
        return ohio.load(root_dir) if root_dir else ohio.load()
    if name == 'azt1d':
        from . import azt1d
        return azt1d.load(root_dir) if root_dir else azt1d.load()
    if name in ('shanghai', 'shanghait1dm'):
        from . import shanghai
        return shanghai.load(root_dir) if root_dir else shanghai.load()
    raise ValueError(
        f"unknown dataset {name!r}; expected ohiot1dm|azt1d|shanghai|personal")
