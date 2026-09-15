"""Finetune a checkpoint on the patient's own phone record, as cached by t1dmdroid_converter.py.

finetune.py's loop and flags, unchanged; only the defaults differ, and any flag given wins. The run
lands in models/<capacity>_omar/checkpoints, so deploy_model.sh names the export after it.
"""

import os
import re
import sys

import finetune

CHECKPOINT = os.path.join('models', 'd16_rmse', 'checkpoints', 't1dmai_best.pt')
# One patient's weeks: far fewer steps than MetaboNet, scored often, on every test window.
DEFAULTS = {
    '--cache': os.path.join('datasets', 't1dmdroid', 't1dm-omar'),
    '--checkpoint': CHECKPOINT,
    '--total-steps': '2000',
    '--warmup-steps': '200',
    '--validation-interval': '200',
    '--log-interval': '50',
    '--eval-windows': '1000000',
}


def _given(flag: str) -> str | None:
    """The value passed as ``flag v`` or ``flag=v``, else None."""
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(flag + '='):
            return a.split('=', 1)[1]
    return None


def main() -> None:
    ckpt = _given('--checkpoint') or CHECKPOINT
    m = re.search(r'models/([^/]+)/checkpoints/[^/]+$', ckpt.replace(os.sep, '/'))
    out_dir = (os.path.join('models', f'{m.group(1)}_omar', 'checkpoints') if m
               else 'checkpoints_finetune_omar')
    defaults = {**DEFAULTS, '--out-dir': out_dir}
    sys.argv[1:1] = [x for k, v in defaults.items() if _given(k) is None for x in (k, v)]
    finetune.main()


if __name__ == '__main__':
    main()
