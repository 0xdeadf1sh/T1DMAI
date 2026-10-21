"""Name -> factory for every zoo model; each module registers its own ``MODELS``."""
from __future__ import annotations

import importlib

MODULES = ('zoo.classical', 'zoo.kernel', 'zoo.boost','zoo.rnn', 'zoo.patch', 'zoo.chronos')


def models() -> dict:
    out: dict = {}
    for m in MODULES:
        try:
            mod = importlib.import_module(m)
        except ModuleNotFoundError as e:
            if e.name != m:
                raise
            continue
        out.update(mod.MODELS)
    return out
