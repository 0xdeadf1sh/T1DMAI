"""
Mask-sampler arms — the (span ceiling, head slots, loss weighting) settings.

``MASK_SPAN_LENGTHS``, ``MAX_MASKED_PATCHES`` and ``D_BALANCED_LOSS`` are read AT
IMPORT by ``data``, ``risk_loss``, ``utils`` and ``metrics.protocols``, so the arm
is resolved before ``config`` binds them and is never mutated afterwards.
``config`` calls ``resolve_arm`` once, during its own import, and republishes the
name as ``config.SAMPLER_ARM``; the values themselves are written here and
nowhere else.

Selection is the ``T1DMAI_ARM`` environment variable, defaulting to ``a``.  An
unknown name RAISES rather than falling back: a typo that quietly trains the
default costs a full run, and nothing in its logs, its checkpoint or its metrics
tables looks wrong afterwards.

A CLI flag can only select an arm before ``config`` is imported: see
``arm_from_argv``.  ``python arms.py`` prints the table exactly as
``train.py --help`` renders it.
"""

from __future__ import annotations

import os
import textwrap
from collections.abc import Sequence
from typing import TypedDict

ARM_ENV_VAR = 'T1DMAI_ARM'
DEFAULT_ARM = 'a'


class Arm(TypedDict):
    mask_span_lengths: tuple[int, ...]
    max_masked_patches: int
    d_balanced_loss: bool
    when_to_use: str


# Provenance and error bars for every figure in a ``when_to_use``, rendered once
# by ``help_text`` so no arm carries its own copy.
EVIDENCE = (
    "Evidence: six runs, one per (arm, pool), 5000 steps at batch 128, one seed, "
    "each figure the step-4999 validation row. A pool is a TRAINING distribution, "
    "not a shared benchmark: each run trained on the pool it names and validated on "
    "that same pool's validation partition. The validation WINDOWS are "
    "arm-independent — one seed, one partition — but the SCORED SUBSET is not: the "
    "forecast protocol drops every row whose context-edge patch that sample's own "
    "training mask covered, and the mask comes from the arm's sampler. a scores 86 "
    "of the 100 windows, b and c an identical 84, so the 'of N' on a figure is that "
    "arm's own denominator on a re-run as much as here — balanced 87 hypo steps for "
    "a against 85 for b and c, hypo-enriched 240 against 221, n = 4-76 per horizon. "
    "a against b or c is not a paired comparison however freshly produced; b against "
    "c is, the two sharing a sampler and differing only in the loss weights. Figures "
    "are hypo recall in percent; @30/@60/@90/@120 min are d = 1/2/3/4 one-sided. "
    "5000 steps is not convergence and one seed carries no interval: a starting "
    "point, not a tuned result."
)

ARMS: dict[str, Arm] = {
    'a': {
        'mask_span_lengths': (1, 2, 3, 4),
        'max_masked_patches': 8,
        'd_balanced_loss': False,
        'when_to_use': (
            "Default. Strongest in the hypo-enriched run: pooled 92.5 of 240 "
            "against 90.0 (b) and 88.7 (c), both of 221; 96.0 of 50 at @90m and "
            "89.8 of 59 at @120m. Weakest at the far horizons of the balanced "
            "run, 62.1 of 29 at @120m."
        ),
    },
    'b': {
        'mask_span_lengths': (1, 2, 3, 4, 5, 6, 7, 8),
        'max_masked_patches': 12,
        'd_balanced_loss': False,
        'when_to_use': (
            "Wider spans and 12 head slots, no loss reweighting: the d histogram "
            "moves through the sampler alone. Between a and c in both runs — "
            "balanced @120m 68.6 of 35 against a's 62.1 of 29, hypo-enriched "
            "pooled 90.0 of 221 against a's 92.5 of 240."
        ),
    },
    'c': {
        'mask_span_lengths': (1, 2, 3, 4, 5, 6, 7, 8),
        'max_masked_patches': 12,
        'd_balanced_loss': True,
        'when_to_use': (
            "b's sampler plus the per-d loss weights. Strongest in the balanced "
            "run: pooled 85.9 of 85, @120m 74.3 of 35 against a's 62.1 of 29, "
            "night 95.0 of 40. Weakest in the hypo-enriched run: pooled 88.7 of "
            "221, and @90m 85.7 of 49 against a's 96.0 of 50. "
            "The reweighting buys far-horizon recall where the far horizons "
            "are underlearned and costs where they are not, spending effective "
            "sample size on d >= 4 that is already learned. Quote "
            "d_balance.effective_sample_size() with any figure from this arm."
        ),
    },
}


def arm_names() -> tuple[str, ...]:
    """The valid arm names, in table order."""
    return tuple(ARMS)


def resolve_arm(name: str | None = None) -> tuple[str, Arm]:
    """Resolve the active arm; ``config`` calls this once, at import.

    ``name`` overrides the environment; without it the arm is ``$T1DMAI_ARM``, or
    ``DEFAULT_ARM`` when that is unset.  Anything else raises ``ValueError``.
    """
    raw = os.environ.get(ARM_ENV_VAR, DEFAULT_ARM) if name is None else name
    key = raw.strip().lower()
    if key not in ARMS:
        source = f"{ARM_ENV_VAR}={raw!r}" if name is None else f"arm {raw!r}"
        raise ValueError(
            f"unknown {source}; valid arms are {', '.join(arm_names())}")
    return key, ARMS[key]


def arm_from_argv(argv: Sequence[str], flag: str = '--arm') -> str | None:
    """The arm named by ``--arm X`` / ``--arm=X`` in ``argv``, or None; validated.

    ``config`` binds the arm's three constants at ITS import, which runs at the
    top of an entry point, before argparse.  A CLI flag therefore reaches the
    sampler only by being read out of ``sys.argv`` and written to the
    environment BEFORE ``import config``::

        import os, sys
        from arms import ARM_ENV_VAR, arm_from_argv
        _arm = arm_from_argv(sys.argv)
        if _arm is not None:
            os.environ[ARM_ENV_VAR] = _arm
        import config

    The flag is left in ``argv`` for argparse to parse, report and echo as
    usual.  An unknown name raises here, before the run starts.
    """
    for i, token in enumerate(argv):
        if token == flag and i + 1 < len(argv):
            raw = argv[i + 1]
        elif token.startswith(f'{flag}='):
            raw = token.split('=', 1)[1]
        else:
            continue
        return resolve_arm(raw)[0]
    return None


def help_text() -> str:
    """The arm table for ``train.py --help``, wrapped and carrying EVIDENCE."""
    blocks = [
        f"  {name}: MASK_SPAN_LENGTHS={arm['mask_span_lengths']}"
        f" MAX_MASKED_PATCHES={arm['max_masked_patches']}"
        f" D_BALANCED_LOSS={arm['d_balanced_loss']}\n"
        + textwrap.fill(arm['when_to_use'], width=78,
                        initial_indent='     ', subsequent_indent='     ')
        for name, arm in ARMS.items()
    ]
    blocks.append('\n' + textwrap.fill(EVIDENCE, width=78,
                                       initial_indent='  ', subsequent_indent='  '))
    return '\n'.join(blocks)


def _validate() -> None:
    """Check every arm, not just the active one, so a table typo raises here.

    Only what is internal to an arm is checkable: whether the masked patches plus
    their separators fit the shortest window depends on ``config``'s own
    constants, and ``tests/test_config.py`` asserts that for the active arm.
    """
    for name, arm in ARMS.items():
        lengths = arm['mask_span_lengths']
        assert lengths and list(lengths) == sorted(set(lengths)) and lengths[0] >= 1, \
            f"arm {name}: mask_span_lengths must be ascending positive ints, got {lengths}"
        assert arm['max_masked_patches'] >= lengths[-1], (
            f"arm {name}: max_masked_patches {arm['max_masked_patches']} is below the "
            f"longest span {lengths[-1]}, so the sampler rejects for ever")
        assert isinstance(arm['d_balanced_loss'], bool), (
            f"arm {name}: d_balanced_loss must be a bool — a string is truthy and "
            f"turns the weights on wherever it is read")
        assert arm['when_to_use'].strip(), f"arm {name}: when_to_use is empty"


_validate()


if __name__ == '__main__':
    print(help_text())
