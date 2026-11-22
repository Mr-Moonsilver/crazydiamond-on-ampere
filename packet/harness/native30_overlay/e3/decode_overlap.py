"""Lever 2 "soft split" (FLASH_S5_DECODE_OVERLAP_SMS, default off): the split decode bank in a wider green context.

Today the split decode bank is captured in the 24-SM decode green context and the native lane runs on the other
58 SMs. With this flag set to N (even, above the declared decode count, below the device total) the worker opens a
second green pair through the SAME helper ABI (``s5_create(dev, N)``) and keeps only its decode context: an N-SM
context that holds the declared decode SMs plus N - 24 SMs of the prefill partition. The split bank is captured on
that context's stream, so a decode forward on the split route may also run on those N - 24 prefill SMs while it runs.
The prefill partition is unchanged: the native lane, its streams, communicators and auxiliaries stay on the base
pair's prefill context. Nothing else changes: same descriptors, one split bank, the same route controller.

``OverlapDecodeSplit`` is the split object the bank, the route controller and the native lane see. Decode-side
attributes and methods come from the wide pair, everything else from the base pair, whose identity the startup
grade reads unchanged: ``describe()`` is the base record plus one ``decode_overlap`` entry.

Startup refuses (typed) on an undeclared or out-of-range count, on a wide pair whose own membership fails, on a
measured overlap with no prefill SM or no decode SM in it, and on a synchronized free-memory drop above
MAX_PAIR_BYTES around the pair. A wide group that does not contain every declared decode SM is recorded
(``superset`` False with both counts), not refused.

Evidence boundary: SM sets come from the helper's probe kernel; the CPU suite uses a fake binding and a fake torch.
Whether the driver accepts an overlapping green context, where decode blocks land and what that costs the native
lane are device questions.
"""
import json

from .e3_types import E3ContractError

ENV = 'FLASH_S5_DECODE_OVERLAP_SMS'
MAX_PAIR_BYTES = 256 * 1024 ** 2
SCHEMA = 's5-decode-overlap/1'
LINE = 'flash-s5 decode-overlap '


class DecodeOverlapError(E3ContractError):
    pass


def declared(env):
    """The declared wide decode count, or None when off (absent or '0')."""
    value = env.get(ENV)
    if value is None or value == '0':
        return None
    if type(value) is not str or not value.isdigit() or value.startswith('0'):
        raise DecodeOverlapError(ENV + ' must be 0 or a positive even integer')
    count = int(value)
    if count % 2:
        raise DecodeOverlapError(ENV + ' must be even (sm_86 partitions align to 2 SMs)')
    return count


class OverlapDecodeSplit:
    """Decode side from the wide pair, prefill side and identity from the base pair."""

    _DECODE = ('decode_stream', 'decode_stream_token', 'decode_context_id', 'decode_sms', 'requested_decode_sms')
    _BASE = ('prefill_stream', 'prefill_stream_token', 'prefill_context_id', 'prefill_sms', 'total_sms',
             'binding', 'device_index', 'torch', 'membership_result')

    def __init__(self, base, wide, record):
        self.base = base
        self.wide = wide
        self.overlap = record

    def __getattr__(self, name):
        # only reached for names not set on the instance
        if name in ('base', 'wide', 'overlap'):
            raise AttributeError(name)
        if name in OverlapDecodeSplit._DECODE:
            return getattr(self.wide, name)
        if name in OverlapDecodeSplit._BASE:
            return getattr(self.base, name)
        raise AttributeError(name)

    @property
    def state(self):
        states = (self.base.state, self.wide.state)
        if states == ('open', 'open'):
            return 'open'
        return self.base.state if self.base.state != 'open' else self.wide.state

    # -- decode side: the wide context ------------------------------------
    def create_decode_aux(self):
        return self.wide.create_decode_aux()

    def check_decode_stream(self, stream):
        return self.wide.check_decode_stream(stream)

    def destroy_decode_aux(self, stream):
        return self.wide.destroy_decode_aux(stream)

    def retain_graph_owner(self, owner):
        # graphs captured on the wide stream also run beside allocations of the base pair's streams
        self.base.retain_allocator_context()
        return self.wide.retain_graph_owner(owner)

    def release_graph_owner(self, owner):
        return self.wide.release_graph_owner(owner)

    # -- prefill side: the base pair, unchanged ---------------------------
    def create_prefill_aux(self):
        return self.base.create_prefill_aux()

    def check_prefill_stream(self, stream):
        return self.base.check_prefill_stream(stream)

    def destroy_prefill_aux(self, stream):
        return self.base.destroy_prefill_aux(stream)

    # -- lifecycle ---------------------------------------------------------
    def retain_allocator_context(self):
        self.base.retain_allocator_context()
        self.wide.retain_allocator_context()

    def lifetime_record(self):
        record = dict(self.base.lifetime_record())
        wide = self.wide.lifetime_record()
        # the capture-stream owner reads allocator_exposed to decide whether its (wide) aux streams may be destroyed
        record['allocator_exposed'] = bool(record['allocator_exposed'] or wide['allocator_exposed'])
        record['decode_overlap'] = wide
        return record

    def membership(self, rounds=3, cycles=100000):
        raise DecodeOverlapError('membership is measured per pair at startup, not on the overlap view')

    def describe(self):
        record = self.base.describe()
        record['decode_overlap'] = dict(self.overlap)
        return record

    def close(self):
        """Wide pair first (it owns the bank's streams), then the base pair, also when the wide close raises."""
        try:
            self.wide.close()
        except BaseException:
            self.base.close()
            raise
        return self.base.close()


def _free_bytes(torch, index):
    torch.cuda.synchronize(index)
    return int(torch.cuda.mem_get_info(index)[0])


def open_wide(base, count, *, split_class=None):
    """Open and measure the wide pair beside an open, measured base pair; returns (wide, record)."""
    if split_class is None:
        from .green_context import GreenSplit as split_class
    if getattr(base, 'state', None) != 'open':
        raise DecodeOverlapError('decode overlap needs the open base split')
    base_m = getattr(base, 'membership_result', None)
    if base_m is None or base_m.ok is not True:
        raise DecodeOverlapError('decode overlap needs the base split membership measured and ok')
    if not base.decode_sms < count < base.total_sms:
        raise DecodeOverlapError('%s=%d must lie above the decode count %d and below the device total %d'
                                 % (ENV, count, base.decode_sms, base.total_sms))
    torch = base.torch
    index = base.device_index
    free_before = _free_bytes(torch, index)
    wide = split_class(base.binding, index, count, torch_module=torch).create()
    try:
        if wide.decode_sms != count or wide.total_sms != base.total_sms:
            raise DecodeOverlapError('wide pair reports %d+%d of %d SMs, declared %d'
                                     % (wide.decode_sms, wide.prefill_sms, wide.total_sms, count))
        measured = wide.membership()
        if measured.ok is not True:
            raise DecodeOverlapError('wide pair membership failed')
        free_after = _free_bytes(torch, index)
        delta = free_before - free_after
        if delta > MAX_PAIR_BYTES:
            raise DecodeOverlapError('wide pair took %d bytes of device memory (limit %d)' % (delta, MAX_PAIR_BYTES))
        own = len(measured.decode_sms & base_m.decode_sms)
        borrowed = len(measured.decode_sms & base_m.prefill_sms)
        if own + borrowed != len(measured.decode_sms) or own == 0 or borrowed == 0:
            raise DecodeOverlapError('wide decode set holds %d decode and %d prefill SMs of %d'
                                     % (own, borrowed, len(measured.decode_sms)))
    except BaseException:
        wide.close()
        raise
    record = {'schema': SCHEMA, 'requested_sms': count, 'decode_sms': wide.decode_sms,
              'wide_prefill_sms_unused': wide.prefill_sms, 'base_decode_sms': base.decode_sms,
              'base_prefill_sms': base.prefill_sms, 'own_decode_sms': own, 'borrowed_prefill_sms': borrowed,
              'superset': own == len(base_m.decode_sms), 'membership': measured.as_record(),
              'free_before_bytes': free_before, 'free_after_bytes': free_after, 'pair_delta_bytes': delta,
              'basis': 'probe-kernel SM ids per stream; device-free samples around the synchronized pair create'}
    return wide, record


def _log(record):
    try:
        from vllm.logger import init_logger
        init_logger(__name__).info('%s%s', LINE, json.dumps(record, sort_keys=True, separators=(',', ':')))
    except Exception:
        pass


def install(base, env, *, split_class=None, log=_log):
    """The split object the bank and the native lane use: ``base`` unchanged when off, else the overlap view."""
    count = declared(env)
    if count is None:
        return base
    wide, record = open_wide(base, count, split_class=split_class)
    log(record)
    return OverlapDecodeSplit(base, wide, record)


__all__ = ['DecodeOverlapError', 'ENV', 'LINE', 'MAX_PAIR_BYTES', 'OverlapDecodeSplit', 'SCHEMA', 'declared',
           'install', 'open_wide']
