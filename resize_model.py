"""
resize_model.py — rewrite config.py with manual architecture overrides.

Usage::

    python resize_model.py                                  # print current architecture + param count
    python resize_model.py --d-model 192 --heads 3          # D_MODEL=192, N_HEADS=3 (head_dim 64)
    python resize_model.py --layers 12                      # change depth only
    python resize_model.py --patch-size 3                   # 15-min patches (PATCH_SIZE * 5 min)
    python resize_model.py --ffn-mult 3 --bg-head-hidden-mult 4
    python resize_model.py --min-context-patches 32 --max-context-patches 96
    python resize_model.py --d-model 256 --heads 4 --report-only   # preview, don't write

Run with no flags, the script prints the live architecture constants from
``config.py`` and the resulting parameter count, then exits without touching
the file.

Given one or more override flags, the script applies exactly those overrides
on top of the current config, leaves everything else as-is, recomputes the
parameter count (it is *computed* from the architecture, never targeted),
rewrites the affected lines of ``config.py``, and prints a before -> after diff.

Overridable constants::

    --d-model               D_MODEL
    --layers                N_LAYERS
    --heads                 N_HEADS
    --patch-size            PATCH_SIZE
    --ffn-mult              FFN_DIM = <mult> * D_MODEL            (kept symbolic)
    --bg-head-hidden-mult   BG_HEAD_HIDDEN = <mult> * D_MODEL     (kept symbolic)
    --min-context-patches   MIN_CONTEXT_PATCHES
    --max-context-patches   MAX_CONTEXT_PATCHES

Constants that are not overridden keep their existing source line.  The
symbolic relations ``FFN_DIM = k * D_MODEL``, ``BG_HEAD_HIDDEN = k * D_MODEL``
and ``HEAD_DIM = D_MODEL // N_HEADS`` are preserved, so changing ``D_MODEL``
alone propagates to the derived widths automatically.

``PATCH_SIZE`` (timesteps per patch; ``PATCH_SIZE * 5`` is the patch span in
minutes) must divide 12 so that an integer number of patches tiles each hour —
the script hard-fails otherwise.  Changing it also rescales the wall-clock span
of ``MIN/MAX_CONTEXT_PATCHES`` (which count *patches*, not hours), so pass those
alongside it to preserve the context window in hours.

Validation is strict — the script refuses to rewrite ``config.py`` unless:

  * ``D_MODEL`` is divisible by ``N_HEADS`` and
    ``HEAD_DIM = D_MODEL // N_HEADS`` is in {16, 32, 64, 128} (so
    ``F.scaled_dot_product_attention`` dispatches the Flash / memory-efficient
    kernels instead of materializing the full T×T attention matrix),
  * ``PATCH_SIZE * 5`` divides 60 (an integer number of patches per hour),
  * ``MIN_CONTEXT_PATCHES`` <= ``MAX_CONTEXT_PATCHES``,
  * every resulting value is positive.

The rest of the config (optimizer, loss, day/night windowing, etc.) is untouched.

Parameter-count method
-----------------------
The count is *computed*, never targeted: the script mutates the in-process
``config`` module to the candidate architecture, re-imports ``model`` so it
binds the new constants, instantiates ``T1DMAI`` on the ``meta`` device (no
real allocation), and returns ``sum(p.numel() for p in m.parameters())``.  The
modules that contribute are::

    patch_embed   Linear(PATCH_DIM, D_MODEL) with bias
    N_LAYERS x TransformerBlock:
        TemporalSelfAttention  w_q/w_k/w_v/w_o : Linear(D_MODEL, D_MODEL) no bias
                               q_norm/k_norm   : RMSNorm(HEAD_DIM)
                               alibi_slopes    : (N_HEADS,)
                               norm1           : RMSNorm(D_MODEL)
        SwiGLUFFN              w1/w3 : Linear(D_MODEL, FFN_DIM) no bias
                               w2    : Linear(FFN_DIM, D_MODEL) no bias
                               norm2 : RMSNorm(D_MODEL)
    final_norm    RMSNorm(D_MODEL)
    bg_head       Linear(D_MODEL, BG_HEAD_HIDDEN) -> SiLU
                  Linear(BG_HEAD_HIDDEN, BG_HEAD_HIDDEN) -> SiLU
                  Linear(BG_HEAD_HIDDEN, BG_HEAD_STEP_BASIS_DIM * (1 + 2*N_SPREADS))   (all with bias)
"""
import argparse
import re
import sys
from pathlib import Path

import torch


VALID_HEAD_DIMS: tuple[int, ...] = (16, 32, 64, 128)

# Architecture constants displayed in the before -> after diff, in order.
# HEAD_DIM is derived (D_MODEL // N_HEADS) and shown for context but never
# written directly; every other key here is settable via a CLI flag.
_VIEW_KEYS: tuple[str, ...] = (
    'D_MODEL', 'N_LAYERS', 'N_HEADS', 'HEAD_DIM', 'FFN_DIM',
    'BG_HEAD_HIDDEN', 'MIN_CONTEXT_PATCHES', 'MAX_CONTEXT_PATCHES', 'PATCH_SIZE',
)


def _arch_view(d_model: int, n_layers: int, n_heads: int, ffn_dim: int,
               bg_head_hidden: int, min_ctx: int, max_ctx: int,
               patch_size: int) -> dict[str, int]:
    """Return the ordered architecture-constant dict used for display/diffing.

    ``HEAD_DIM`` is rendered as ``D_MODEL // N_HEADS`` when the division is
    exact, otherwise as the raw ``d_model/n_heads`` string so an invalid
    combination is still legible in the diff before validation rejects it.
    """
    head_dim: object = d_model // n_heads if n_heads and d_model % n_heads == 0 else f'{d_model}/{n_heads}'
    return {
        'D_MODEL': d_model,
        'N_LAYERS': n_layers,
        'N_HEADS': n_heads,
        'HEAD_DIM': head_dim,  # type: ignore[dict-item]
        'FFN_DIM': ffn_dim,
        'BG_HEAD_HIDDEN': bg_head_hidden,
        'MIN_CONTEXT_PATCHES': min_ctx,
        'MAX_CONTEXT_PATCHES': max_ctx,
        'PATCH_SIZE': patch_size,
    }


def _view_from_config(config) -> dict[str, int]:
    """Snapshot the live architecture constants from the ``config`` module."""
    return _arch_view(
        config.D_MODEL, config.N_LAYERS, config.N_HEADS, config.FFN_DIM,
        config.BG_HEAD_HIDDEN, config.MIN_CONTEXT_PATCHES,
        config.MAX_CONTEXT_PATCHES, config.PATCH_SIZE,
    )


def _effective(config, args: argparse.Namespace) -> dict[str, int]:
    """Resolve the post-override architecture from current config + CLI flags.

    A flag that is ``None`` keeps the current config value.  ``FFN_DIM`` and
    ``BG_HEAD_HIDDEN`` are kept symbolic: when their multiplier flag is omitted
    but ``--d-model`` is set, they re-scale with the new ``D_MODEL`` (mirroring
    the ``k * D_MODEL`` source expression); when neither is set they retain the
    current numeric value exactly.
    """
    d_model = args.d_model if args.d_model is not None else config.D_MODEL
    n_layers = args.layers if args.layers is not None else config.N_LAYERS
    n_heads = args.heads if args.heads is not None else config.N_HEADS

    if args.ffn_mult is not None:
        ffn_dim = args.ffn_mult * d_model
    elif args.d_model is not None:
        ffn_dim = (config.FFN_DIM // config.D_MODEL) * d_model
    else:
        ffn_dim = config.FFN_DIM

    if args.bg_head_hidden_mult is not None:
        bg_head_hidden = args.bg_head_hidden_mult * d_model
    elif args.d_model is not None:
        bg_head_hidden = (config.BG_HEAD_HIDDEN // config.D_MODEL) * d_model
    else:
        bg_head_hidden = config.BG_HEAD_HIDDEN

    min_ctx = args.min_context_patches if args.min_context_patches is not None else config.MIN_CONTEXT_PATCHES
    max_ctx = args.max_context_patches if args.max_context_patches is not None else config.MAX_CONTEXT_PATCHES
    patch_size = args.patch_size if args.patch_size is not None else config.PATCH_SIZE

    return {
        'd_model': d_model, 'n_layers': n_layers, 'n_heads': n_heads,
        'ffn_dim': ffn_dim, 'bg_head_hidden': bg_head_hidden,
        'min_ctx': min_ctx, 'max_ctx': max_ctx, 'patch_size': patch_size,
    }


def _validate(eff: dict[str, int]) -> list[str]:
    """Return a list of human-readable constraint violations (empty == valid)."""
    errs: list[str] = []
    positives = {
        'D_MODEL': eff['d_model'], 'N_LAYERS': eff['n_layers'], 'N_HEADS': eff['n_heads'],
        'FFN_DIM': eff['ffn_dim'], 'BG_HEAD_HIDDEN': eff['bg_head_hidden'],
        'MIN_CONTEXT_PATCHES': eff['min_ctx'], 'MAX_CONTEXT_PATCHES': eff['max_ctx'],
        'PATCH_SIZE': eff['patch_size'],
    }
    for name, val in positives.items():
        if val <= 0:
            errs.append(f'{name} must be positive, got {val}')

    if eff['patch_size'] > 0 and 60 % (eff['patch_size'] * 5) != 0:
        errs.append(
            f"PATCH_SIZE={eff['patch_size']} ({eff['patch_size'] * 5} min/patch) does not tile "
            f"the hour: PATCH_SIZE * 5 must divide 60 (valid PATCH_SIZE: 1, 2, 3, 4, 6, 12)"
        )

    if eff['n_heads'] > 0 and eff['d_model'] % eff['n_heads'] != 0:
        errs.append(f"D_MODEL={eff['d_model']} is not divisible by N_HEADS={eff['n_heads']}")
    elif eff['n_heads'] > 0:
        head_dim = eff['d_model'] // eff['n_heads']
        if head_dim not in VALID_HEAD_DIMS:
            errs.append(
                f"head_dim = D_MODEL // N_HEADS = {head_dim} is not in {VALID_HEAD_DIMS} "
                f"(required for F.scaled_dot_product_attention flash kernels)"
            )

    if eff['min_ctx'] > eff['max_ctx']:
        errs.append(
            f"MIN_CONTEXT_PATCHES={eff['min_ctx']} exceeds MAX_CONTEXT_PATCHES={eff['max_ctx']}"
        )

    return errs


def _changes(args: argparse.Namespace) -> dict[str, str]:
    """Map each provided override flag to the source text to write into config.py.

    Only flags the user actually passed appear here, so untouched lines (and
    their trailing comments) are preserved.  ``FFN_DIM`` and ``BG_HEAD_HIDDEN``
    are written as ``<mult> * D_MODEL`` expressions to keep the architectural
    relation intact.
    """
    out: dict[str, str] = {}
    if args.d_model is not None:
        out['D_MODEL'] = str(args.d_model)
    if args.layers is not None:
        out['N_LAYERS'] = str(args.layers)
    if args.heads is not None:
        out['N_HEADS'] = str(args.heads)
    if args.patch_size is not None:
        out['PATCH_SIZE'] = str(args.patch_size)
    if args.ffn_mult is not None:
        out['FFN_DIM'] = f'{args.ffn_mult} * D_MODEL'
    if args.bg_head_hidden_mult is not None:
        out['BG_HEAD_HIDDEN'] = f'{args.bg_head_hidden_mult} * D_MODEL'
    if args.min_context_patches is not None:
        out['MIN_CONTEXT_PATCHES'] = str(args.min_context_patches)
    if args.max_context_patches is not None:
        out['MAX_CONTEXT_PATCHES'] = str(args.max_context_patches)
    return out


def _count_params(d_model: int, n_heads: int, ffn_dim: int,
                  bg_head_hidden: int, n_layers: int,
                  patch_size: int | None = None) -> int:
    """Instantiate T1DMAI with overridden dims on the meta device; return numel.

    Overrides the architecture constants on the already-loaded ``config``
    module, then evicts ``model`` from ``sys.modules`` so the next
    ``from model import T1DMAI`` re-runs ``model.py`` and picks up the new
    constants.  Uses ``torch.device('meta')`` so no real memory is allocated
    for the parameter tensors — only the ``numel`` count is needed.  Context-patch
    counts are deliberately not set here: they do not affect the parameter count
    (RoPE / ALiBi are parameter-light).  ``patch_size`` *does* affect the count
    (via PATCH_DIM; the ``(S, K)`` step_basis is a non-parameter buffer and the
    bg_head output width K*(1 + 2*N_SPREADS) no longer scales with it), so when it
    is given the derived constants are re-evaluated to match.
    """
    import config
    # Snapshot every config global this helper mutates so it stays side-effect-free
    # (no global-state pollution of a long-lived `config` module across calls/tests).
    _MUTATED = ('D_MODEL', 'N_HEADS', 'HEAD_DIM', 'FFN_DIM', 'BG_HEAD_HIDDEN',
                'N_LAYERS', 'PATCH_SIZE', 'PATCH_DIM', '_PATCHES_PER_HOUR',
                'PREDICTION_PATCHES', 'MAX_SEQ_LEN', 'NIGHT_LONG_HORIZON_PATCHES')
    _saved = {k: getattr(config, k) for k in _MUTATED}
    try:
        config.D_MODEL = d_model
        config.N_HEADS = n_heads
        config.HEAD_DIM = d_model // n_heads
        config.FFN_DIM = ffn_dim
        config.BG_HEAD_HIDDEN = bg_head_hidden
        config.N_LAYERS = n_layers
        if patch_size is not None:
            # PATCH_SIZE feeds patch_embed (PATCH_DIM) and the (S, K) step_basis
            # buffer; the bg_head output width is K*(1 + 2*N_SPREADS), independent
            # of PATCH_SIZE.  Re-derive every PATCH_SIZE-dependent constant before
            # re-importing model.  The caller validates that PATCH_SIZE * 5 divides
            # 60, so the floor division below is exact.
            pph = 60 // (patch_size * 5)
            config.PATCH_SIZE = patch_size
            config.PATCH_DIM = patch_size * config.N_INPUT_FEATURES
            config._PATCHES_PER_HOUR = pph
            config.PREDICTION_PATCHES = config.PREDICTION_HORIZON_HOURS * pph
            config.MAX_SEQ_LEN = config.MAX_CONTEXT_PATCHES + config.PREDICTION_PATCHES
            config.NIGHT_LONG_HORIZON_PATCHES = config.NIGHT_LONG_HORIZON_HOURS * pph
        sys.modules.pop('model', None)
        from model import T1DMAI
        with torch.device('meta'):
            m = T1DMAI()
        return sum(p.numel() for p in m.parameters())
    finally:
        # Restore the live config dims and evict the overridden `model` module so
        # the next `from model import` re-reads the restored constants.
        for k, v in _saved.items():
            setattr(config, k, v)
        sys.modules.pop('model', None)


def _replace_rhs(src: str, name: str, new_value: str) -> str:
    """Replace the RHS of ``NAME = <value>`` in config.py source, preserving any trailing comment."""
    pattern = rf'^({re.escape(name)}\s*=\s*)([^#\n]+?)(\s*(?:#[^\n]*)?)$'
    replaced = [False]

    def _sub(m: re.Match[str]) -> str:
        replaced[0] = True
        return f"{m.group(1)}{new_value}{m.group(3)}"

    out = re.sub(pattern, _sub, src, count=1, flags=re.MULTILINE)
    if not replaced[0]:
        raise RuntimeError(f"could not locate assignment for {name} in config.py")
    return out


def _write_constants(path: Path, changes: dict[str, str]) -> None:
    """Patch each ``NAME = <value>`` line in ``config.py`` from the changes map.

    Every other line of the file (and trailing comments on the rewritten lines)
    is preserved verbatim — only the architecture constants this script owns are
    rewritten.  ``config.py`` is a plain file (there are no config variants), so
    ``write_text`` simply rewrites it in place.
    """
    src = path.read_text()
    for name, value in changes.items():
        src = _replace_rhs(src, name, value)
    path.write_text(src)


def _print_view(view: dict[str, int], count: int) -> None:
    """Print an architecture view and its parameter count."""
    name_w = max(len(k) for k in view)
    val_w = max(len(str(v)) for v in view.values())
    for k in _VIEW_KEYS:
        print(f'  {k:<{name_w}}  {str(view[k]):>{val_w}}')
    print()
    print(f'  param count = {count:>12,}  ({count / 1e6:.2f}M)')


def _print_diff(old: dict[str, int], new: dict[str, int],
                old_count: int, new_count: int) -> None:
    """Print a before -> after architecture diff and the resulting param counts."""
    name_w = max(len(k) for k in new)
    old_w = max(len(str(v)) for v in old.values())
    new_w = max(len(str(v)) for v in new.values())
    changed_any = False
    for k in _VIEW_KEYS:
        marker = '*' if old[k] != new[k] else ' '
        changed_any = changed_any or old[k] != new[k]
        print(f'  {marker} {k:<{name_w}}  {str(old[k]):>{old_w}}  ->  {str(new[k]):>{new_w}}')
    if not changed_any:
        print('  (no parameters changed — config.py already matches)')
    print()
    print(f'  before = {old_count:>12,}  ({old_count / 1e6:.2f}M)')
    print(f'  after  = {new_count:>12,}  ({new_count / 1e6:.2f}M)')
    print(f'  delta  = {new_count - old_count:+12,}  ({(new_count - old_count) / 1e6:+.2f}M)')


def _print_current_arch() -> None:
    """Print the live architecture constants and parameter count, then return."""
    import config
    view = _view_from_config(config)
    count = _count_params(
        config.D_MODEL, config.N_HEADS, config.FFN_DIM,
        config.BG_HEAD_HIDDEN, config.N_LAYERS,
    )
    print('[resize_model] current architecture (from config.py):')
    print()
    _print_view(view, count)


def main() -> None:
    # Patch geometry for the context-patch help text: the wall-clock span of a
    # context-patch count depends on the current PATCH_SIZE, so render the help
    # against whatever PATCH_SIZE is presently in config.py rather than baking in
    # the 30-min / 16-patch numbers that only held at PATCH_SIZE=6.
    import config
    _patch_min = config.PATCH_SIZE * 5
    _floor_patches = round(8 * 60 / _patch_min)  # patches in the ~8 h ACF floor

    parser = argparse.ArgumentParser(
        description=(
            'Rewrite config.py with manual architecture overrides. Pass one or '
            'more of the flags below to change those constants; everything else '
            'is preserved. The parameter count is computed from the resulting '
            'architecture (it is never targeted). Run with no flags to print the '
            'current architecture and parameter count without rewriting config.py.'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--d-model', type=int, default=None,
        help='Override D_MODEL: residual-stream width and the primary capacity '
             'lever. Most matmul FLOPs and ~D_MODEL^2 of the params scale with '
             'it, so raising it lifts model quality but costs VRAM/compute.')
    parser.add_argument(
        '--layers', type=int, default=None,
        help='Override N_LAYERS: transformer depth. More blocks add sequential/'
             'compositional capacity (params and step time grow ~linearly) but '
             'are harder to optimize and raise activation memory (every block '
             'keeps its activations — there is no gradient checkpointing).')
    parser.add_argument(
        '--heads', type=int, default=None,
        help='Override N_HEADS: temporal attention heads (HEAD_DIM = D_MODEL // '
             'N_HEADS). More heads give finer attention subspaces at a smaller '
             'per-head dim; near param-neutral (only ALiBi slopes scale).')
    parser.add_argument(
        '--patch-size', type=int, default=None,
        help='Override PATCH_SIZE: timesteps per patch (x5 min = patch span; e.g. '
             '3 = 15-min patches). Finer patches sharpen temporal/event resolution '
             'but roughly double the token count and quadruple attention cost; must '
             'divide 12. Also rescales the wall-clock span of the context-patch bounds.')
    parser.add_argument(
        '--ffn-mult', type=int, default=None,
        help='Set FFN_DIM = <mult> * D_MODEL (kept symbolic): per-token SwiGLU '
             'width and the largest single share of params. Bigger = more '
             'per-token nonlinear capacity, more compute/VRAM.')
    parser.add_argument(
        '--bg-head-hidden-mult', type=int, default=None,
        help='Set BG_HEAD_HIDDEN = <mult> * D_MODEL (kept symbolic): hidden width '
             'of the shared 3-layer BG quantile head. More capacity to decode the '
             'risk-space median + quantile spreads; small param share.')
    parser.add_argument(
        '--min-context-patches', type=int, default=None,
        help=f'Override MIN_CONTEXT_PATCHES: shortest sampled context (1 patch = '
             f'{_patch_min} min at the current PATCH_SIZE={config.PATCH_SIZE}). '
             f'Lowering it trains short-history robustness, but below the ~8 h '
             f'(~{_floor_patches}-patch) ACF floor little autoregressive signal '
             f'remains. Param-neutral.')
    parser.add_argument(
        '--max-context-patches', type=int, default=None,
        help=f'Override MAX_CONTEXT_PATCHES: longest context / left-pad width; '
             f'wall-clock span = MAX_CONTEXT_PATCHES x {_patch_min} min at the '
             f'current PATCH_SIZE={config.PATCH_SIZE}. More history aids long-range '
             f'and nocturnal accuracy but attention cost grows ~O(n^2) in sequence '
             f'length. Param-neutral.')
    parser.add_argument(
        '--report-only', action='store_true',
        help='Compute and print the diff without writing config.py (not a '
             'performance knob).')
    args = parser.parse_args()

    override_attrs = (
        'd_model', 'layers', 'heads', 'patch_size', 'ffn_mult', 'bg_head_hidden_mult',
        'min_context_patches', 'max_context_patches',
    )
    if not any(getattr(args, a) is not None for a in override_attrs):
        _print_current_arch()
        return

    import config
    path = Path(__file__).resolve().parent / 'config.py'

    # Snapshot the current architecture BEFORE _count_params mutates the config
    # module in-process, so the before -> after diff is accurate.
    old = _view_from_config(config)

    eff = _effective(config, args)
    errs = _validate(eff)
    if errs:
        parser.error('invalid architecture:\n  - ' + '\n  - '.join(errs))

    old_count = _count_params(
        old['D_MODEL'], old['N_HEADS'], old['FFN_DIM'],
        old['BG_HEAD_HIDDEN'], old['N_LAYERS'], old['PATCH_SIZE'],
    )
    new_count = _count_params(
        eff['d_model'], eff['n_heads'], eff['ffn_dim'], eff['bg_head_hidden'],
        eff['n_layers'], eff['patch_size'],
    )
    new = _arch_view(
        eff['d_model'], eff['n_layers'], eff['n_heads'], eff['ffn_dim'],
        eff['bg_head_hidden'], eff['min_ctx'], eff['max_ctx'], config.PATCH_SIZE,
    )

    changes = _changes(args)

    print('[resize_model] applying manual architecture overrides:')
    print()
    _print_diff(old, new, old_count, new_count)
    if eff['patch_size'] != old['PATCH_SIZE']:
        mins = eff['patch_size'] * 5
        min_h = eff['min_ctx'] * mins / 60
        max_h = eff['max_ctx'] * mins / 60
        print()
        print(f'  note: 1 patch is now {mins} min; at MIN/MAX_CONTEXT_PATCHES = '
              f"{eff['min_ctx']}/{eff['max_ctx']} the context window spans "
              f'{min_h:.1f}/{max_h:.1f} h.')
        print('        pass --min-context-patches / --max-context-patches to preserve '
              'the wall-clock window.')
    print()
    if args.report_only:
        print('  [report-only] config.py NOT modified')
    else:
        _write_constants(path, changes)
        print(f'  wrote {path}')


if __name__ == '__main__':
    main()
