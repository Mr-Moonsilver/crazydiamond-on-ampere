"""attn-meta per-arm selector (default off): one instance wrapper over attn_meta_batch's GdnBatch, switched per arm.

Lead decision A (3 Nov): the box package compares attn-meta off / on inside one engine lifetime, so the switch is a typed
per-arm control, not the startup env. FLASH_ATTN_META_GDN_BATCH=1 at startup installs (``install``):

  1. attn_meta_batch.install(runner) unchanged (its bytes are VM-proven: attn-meta-vm-b): it refuses a second install, a
     model state that already carries an instance ``prepare_attn`` (a seam installed first, e.g. handler anatomy), a class
     that overrides it and a non-mamba-hybrid model state;
  2. a ``Selector`` replaces that instance attribute: selected -> the GdnBatch; not selected -> the class's stock
     ``MambaHybridModelState.prepare_attn`` called directly (the GdnBatch is never entered, so its counters stay still).
     The Selector starts NOT selected; every arm selects explicitly (``select``).

Order: s5_startup.install_builder calls ``startup`` at initialize_kv_cache, before the first worker RPC; handler anatomy
installs at the first RPC and wraps whatever ``mstate.prepare_attn`` holds then, i.e. the Selector. ``startup`` logs one
line per rank (``flash-attn-meta installed``) with what it found and set; ``order(runner)`` reports, at any later time,
whether the outermost ``prepare_attn`` is the Selector or a wrapper whose closure holds it (anatomy over attn-meta), so the
box grade can assert the order. If anything else was installed first, install refuses (startup fails): never a skip.

Switch rule: ``select`` is only called from the worker control action, which the engine driver admits only while paused
idle FULL (no transaction, no native work, no inflight RPC): the engine side holds that rule, the worker side holds the
typed fields. Off: ``startup`` returns before importing attn_meta_batch, torch or vLLM and sets nothing.
"""
import os

ENV = 'FLASH_ATTN_META_GDN_BATCH'
SCHEMA = 'attn-meta-select/1'
_MARK = '_flash_attn_meta_selector'


class AttnMetaSelectError(RuntimeError):
    pass


def enabled(env=None):
    env = os.environ if env is None else env
    value = env.get(ENV, '0')
    if value not in ('0', '1'):
        raise AttnMetaSelectError(ENV + ' must be 0 or 1')
    return value == '1'


class Selector:
    """The instance ``prepare_attn``: the GdnBatch when selected, the class's stock function otherwise."""

    def __init__(self, mstate, batch, stock):
        self.mstate = mstate
        self.batch = batch
        self.stock = stock
        self.selected = False
        self.off_calls = 0
        self.selections = 0

    def __call__(self, *args, **kwargs):
        if self.selected:
            return self.batch(*args, **kwargs)
        self.off_calls += 1
        return self.stock(self.mstate, *args, **kwargs)

    def counters(self):
        inner = self.batch.counters()
        return dict(schema=SCHEMA, batch_schema=inner['schema'], selected=self.selected, batched=inner['batched'],
                    fallback=inner['fallback'], off_calls=self.off_calls, selections=self.selections)


def install(runner, *, env=None):
    """attn_meta_batch.install, then the Selector over it (not selected). None when off."""
    if not enabled(env):
        return None
    import attn_meta_batch
    from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
    mstate = getattr(runner, 'model_state', None)
    if getattr(mstate, _MARK, None) is not None:
        raise AttnMetaSelectError('selector already installed')
    batch = attn_meta_batch.install(runner, env={attn_meta_batch.ENV: '1'})
    if vars(mstate).get('prepare_attn') is not batch:
        raise AttnMetaSelectError('the batch is not the instance prepare_attn after install')
    sel = Selector(mstate, batch, MambaHybridModelState.prepare_attn)
    setattr(mstate, _MARK, sel)
    mstate.prepare_attn = sel
    return sel


def installed(runner):
    return getattr(getattr(runner, 'model_state', None), _MARK, None)


def _holds(fn, target, depth=4):
    """True when ``fn`` is ``target`` or a closure (up to ``depth`` wrappers deep) holding it."""
    if fn is target:
        return True
    if depth == 0:
        return False
    for cell in getattr(fn, '__closure__', None) or ():
        try:
            value = cell.cell_contents
        except ValueError:
            continue
        if callable(value) and _holds(value, target, depth - 1):
            return True
    return False


def order(runner):
    """What wraps what now: dict(outer, selector_outer, wrapped_by_other) or None when not installed."""
    sel = installed(runner)
    if sel is None:
        return None
    outer = vars(sel.mstate).get('prepare_attn')
    return dict(outer=type(outer).__name__, selector_outer=outer is sel,
                wrapped_by_other=outer is not sel and _holds(outer, sel),
                ok=outer is sel or _holds(outer, sel))


def select(runner, on):
    """Set this rank's selection; returns the counters before and after (typed fields only)."""
    if type(on) is not bool:
        raise AttnMetaSelectError('attn-meta-select-not-bool')
    sel = installed(runner)
    if sel is None:
        raise AttnMetaSelectError('attn-meta-not-installed')
    before = sel.counters()
    sel.selections += int(sel.selected != on)
    sel.selected = on
    return dict(before=before, after=sel.counters(), order=order(runner))


def read(runner):
    sel = installed(runner)
    if sel is None:
        raise AttnMetaSelectError('attn-meta-not-installed')
    return dict(counters=sel.counters(), order=order(runner))


def startup(runner, is_profiling=False, *, env=None, rank=None, logger=None):
    """s5_startup's call at initialize_kv_cache: install when on, and log the order line. Off: nothing, no import."""
    if is_profiling or not enabled(env):
        return None
    sel = install(runner, env=env)
    facts = order(runner)
    if rank is None:
        try:
            import torch.distributed as dist
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else None
        except Exception:
            rank = None
    line = ('flash-attn-meta installed schema=%s rank=%s selector_outer=%d wrapped_by_other=%d selected=%d'
            % (SCHEMA, rank, int(facts['selector_outer']), int(facts['wrapped_by_other']), int(sel.selected)))
    if logger is None:
        try:
            from vllm.logger import init_logger
            logger = init_logger(__name__)
        except ImportError:
            import logging
            logger = logging.getLogger(__name__)
    logger.info(line)
    return sel


__all__ = ['ENV', 'SCHEMA', 'AttnMetaSelectError', 'Selector', 'enabled', 'install', 'installed', 'order', 'read',
           'select', 'startup']
