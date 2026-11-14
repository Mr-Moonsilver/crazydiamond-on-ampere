"""Bounded passive correlation of existing decode and native timing events."""
import hashlib
import json
import math
import time

from native_measurement import ProgressTrace, emit_receipt_lines

ENV = 'FLASH_S5_DECODE_TIMELINE'
MAX_SESSIONS = 256
MAX_METADATA = 64
MAX_FORWARDS = 32
MAX_PHASES = 96
MAX_LOG_BYTES = 4 * 1024 * 1024
OUTSIDE_KEY = '_flash_s5_dt'
OUTSIDE_EVERY = 64
MAX_OUTSIDE = 1024
PHASES = frozenset(('copy', 'prefetch_copy', 'attention', 'indexer',
                    'canonical_selection', 'kv_producer_update'))


def check_startup(env):
    value = env.get(ENV, '0')
    if value not in ('0', '1'):
        raise ValueError('invalid boolean setting: ' + ENV)
    if value == '1' and (env.get('FLASH_S5_HANDLER_ANATOMY') != '1'
            or env.get('FLASH_S5_AUTO_ARM') != '1'
            or env.get('FLASH_S5_NATIVE_PROGRESS') not in ('burst', 'trace')
            or not str(env.get('FLASH_S5_FUSED_PHASE_EVERY', '0')).isdigit()
            or not 1 <= int(env.get('FLASH_S5_FUSED_PHASE_EVERY', '0')) <= 4096):
        raise ValueError(ENV + ' needs auto-arm, handler anatomy, progress and fused sampling')


class TimelineTrace(ProgressTrace):
    SCHEMA = 's5-decode-timeline/1'
    LINE = 'flash-s5 decode-timeline '
    ROWS = {
        'd': ('seq', 'metadata', 'start_ms', 'end_ms', 'device_ns', 'forward_host_ns',
              'observer_ns', 't_enter', 't_exit', 't_complete_query'),
        'p': ('phase', 'layer', 'group', 'bucket', 'stream', 'start_ms', 'end_ms'),
        's': ('complete', 'counters'),
    }


class Timeline:
    """Copies scalars and retains at most one completed prior timing anchor."""

    def __init__(self, recorder, runner):
        self.recorder, self.runner = recorder, runner
        self.active = None
        self.previous = None
        self.outside = None
        self.outside_samples = 0
        self.sessions = self.bytes = 0
        self.blocked = False
        self.timing_failed = False
        self.device = runner._flash_s5_worker.operation_device
        if getattr(self.device, 'decode_timeline', None) is not None:
            raise ValueError('decode timeline already installed')
        self.device.decode_timeline = self

    def _count(self, name, session=None):
        session = self.active if session is None else session
        if session is not None:
            c = session['counts']
            c[name] = c.get(name, 0) + 1

    def _session(self, seq):
        return self.outside if self.outside is not None and self.outside['seq'] == seq else self.active

    def outside_begin(self, sequence, epoch, sent_ns):
        self.poll_outside()
        if self.blocked or self.outside_samples >= MAX_OUTSIDE:
            return
        if self.outside is not None:
            self._count('unresolved_forward', self.outside)
            self._emit(False, self.outside)
            self.outside = None
        if self.blocked:
            return
        self.outside_samples += 1
        self.outside = dict(seq=sequence, anchor=None, metadata={}, counts={}, forwards=0,
            trace=TimelineTrace('worker', self.recorder.proc.rank, limit=2),
            identity=dict(scope='outside-native-progress', engine_epoch=epoch,
                stage=self.recorder.identity.get('pp_rank'),
                sample='outside-' + str(self.outside_samples), sample_every=OUTSIDE_EVERY,
                anchor_sample='outside-' + str(self.outside_samples),
                engine_send_ns=sent_ns, execute_seq=sequence, byte_limit=MAX_LOG_BYTES,
                sample_limit=MAX_OUTSIDE, anchor='this-decode-forward-first-event',
                basis='local queried forward completion; no borrowed get_output or cross-GPU clock'))

    def poll_outside(self):
        if self.outside is not None:
            self.recorder._resolve()
            if self.outside['forwards']:
                self._emit(not self.outside['counts'], self.outside)
                self.outside = None

    def begin(self, trace):
        if self.blocked:
            return
        if self.active is not None:
            self._count('superseded')
            self._emit(False)
            if self.blocked:
                return
        if self.sessions >= MAX_SESSIONS:
            self.blocked = True
            self.previous = None
            return
        self.sessions += 1
        tx = trace.operation.transaction
        if self.previous is not None and self.previous[0] != tx.engine_epoch:
            self.previous = None
        previous = self.previous
        self.active = dict(owner=id(trace), anchor=previous[2] if previous else None,
            next_anchor=None, metadata={}, counts={}, forwards=0,
            trace=TimelineTrace('worker', trace.operation.lane.worker_rank, limit=MAX_FORWARDS + MAX_PHASES + 1),
            identity=dict(engine_epoch=tx.engine_epoch, request_generation=tx.request_generation,
                txn=tx.txn_seq, chunk_start=tx.chunk_start, chunk_end=tx.chunk_end,
                sample=self.sessions, sample_every=self.device.native_fused_phase_every,
                session_limit=MAX_SESSIONS, byte_limit=MAX_LOG_BYTES, anchor='native-phase-event',
                anchor_sample=previous[1] if previous else self.sessions,
                basis='same-GPU event offsets; not a host clock or kernel-busy counter'))

    def anchor(self, trace, event):
        if self.active is not None and self.active['owner'] == id(trace) and self.active['next_anchor'] is None:
            self.active['next_anchor'] = event
            if self.active['anchor'] is None:
                self.active['anchor'] = event

    def inputs(self, seq, output):
        session = self._session(seq)
        if session is None:
            return
        try:
            entries = output.num_scheduled_tokens
            if (type(entries) is not dict or len(entries) > MAX_METADATA
                    or any(type(k) is not str or len(k) > 256 or type(v) is not int or v < 0
                           for k, v in entries.items())):
                raise ValueError('unavailable scheduled rows')
            if not entries:
                return
            meta = session['metadata']
            if len(meta) >= MAX_METADATA and seq not in meta:
                self._count('metadata_dropped', session)
                return
            meta[seq] = dict(actual_rows=len(entries), actual_tokens=sum(entries.values()),
                uid_digest=hashlib.sha256(json.dumps(sorted(entries), separators=(',', ':')).encode()).hexdigest())
            cached = getattr(output, 'scheduled_cached_reqs', None)
            ids, counts = getattr(cached, 'req_ids', ()), getattr(cached, 'num_computed_tokens', ())
            history = (dict(zip(ids, counts)) if type(ids) in (list, tuple) and type(counts) in (list, tuple)
                       and len(ids) == len(counts) <= MAX_METADATA else {})
            new = getattr(output, 'scheduled_new_reqs', ())
            for request in (new[:MAX_METADATA] if type(new) in (list, tuple) else ()):
                history[getattr(request, 'req_id', None)] = getattr(request, 'num_computed_tokens', None)
            meta[seq]['history_tokens'] = (sorted(history[k] for k in entries)
                if all(type(history.get(k)) is int and history[k] >= 0 for k in entries) else None)
        except Exception:
            self._count('metadata_errors', session)

    def descriptor(self, seq, args, stream):
        session = self._session(seq)
        if session is None:
            return
        try:
            meta = session['metadata'].get(seq)
            if meta is None:
                self._count('metadata_missing', session)
                return
            desc = next(x for x in args if hasattr(x, 'cg_mode') and hasattr(x, 'num_tokens'))
            mode = desc.cg_mode
            shape = {k: getattr(desc, k, None) for k in
                     ('num_tokens', 'num_reqs', 'uniform_token_count', 'max_query_len', 'num_ubatches')}
            if any(v is not None and type(v) is not int for v in shape.values()):
                raise ValueError('descriptor is not scalar metadata')
            native = self.runner._flash_s5_native
            split = self.runner._flash_s5_split
            bank = native.controller.route
            meta.update(bank=native.controller.route, route_generation=native.controller.route_generation,
                        decode_sms=split.decode_sms,
                        bank_sms=(split.decode_sms if bank == 'split' else
                                  getattr(split, 'total_sms', None) if bank == 'full' else None),
                        stream=stream.cuda_stream, cg_mode=str(getattr(mode, 'name', mode))[:32], **shape)
        except Exception:
            self._count('metadata_errors', session)

    def _offsets(self, first, last, session=None):
        session = self.active if session is None else session
        if self.timing_failed:
            self._count('timing_unavailable', session)
            return
        anchor = session['anchor']
        try:
            if anchor is None or not anchor.query() or not first.query() or not last.query():
                self._count('pending', session)
                return
            lo, hi = float(anchor.elapsed_time(first)), float(anchor.elapsed_time(last))
            if not math.isfinite(lo) or not math.isfinite(hi) or hi < lo:
                raise ValueError('invalid event interval')
            return round(lo, 6), round(hi, 6)
        except Exception:
            # Cross-green timing support is a device gate; never repair it with a wait.
            self._count('timing_errors', session)
            self.timing_failed = True
            self.previous = None

    def forward(self, seq, pair, *, device_ns, forward_host_ns, observer_ns, t_enter, t_exit,
                forward_calls, handler_flags):
        session = self._session(seq)
        if session is None:
            return
        try:
            completed_ns = time.monotonic_ns()  # Recorder already queried both local endpoints complete.
            meta = session['metadata'].pop(seq, None)
            if meta is None:
                return  # Earlier RPCs can retire after this sample began.
            if 'bank' not in meta:
                self._count('unpaired', session)
                return
            meta.update(forward_calls=forward_calls, handler_flags=handler_flags,
                        completion='own-forward-end-event-query')
            if session['forwards'] >= MAX_FORWARDS:
                self._count('forwards_dropped', session)
                return
            if session is self.outside:
                times = (0., round(device_ns / 1e6, 6))  # Already read by the owning handler, same context.
            else:
                times = self._offsets(*pair, session=session)
            if times is None:
                times = (None, None)
            session['trace'].record('d', seq, meta, *times, device_ns, forward_host_ns,
                                        observer_ns, t_enter, t_exit, completed_ns)
            session['forwards'] += 1
        except Exception:
            self._count('forward_errors', session)

    def finish(self, trace, *, source_complete):
        if self.active is None or self.active['owner'] != id(trace):
            return
        try:
            # Resolve before the native anchor is released, without waiting for late decode.
            self.recorder._resolve()
            phases = 0
            for name, phase, first, last in trace.pairs:
                if phase not in PHASES:
                    continue
                if phases >= MAX_PHASES:
                    self._count('phases_dropped')
                    continue
                times = self._offsets(first, last)
                if times is None:
                    continue
                bucket, _ = trace.pair_details[id(first)]
                stream = trace.operation.device.stream_token('copy' if phase in ('copy', 'prefetch_copy') else 'compute')
                self.active['trace'].record('p', phase, name[:160], trace.rows[name]['group'], bucket, stream, *times)
                phases += 1
            self.active['identity'].update(stage=trace.stage, route_generation=trace.route_generation,
                                           phases=phases, forwards=self.active['forwards'],
                                           cancelled=trace.operation._cancelled is not None,
                                           timing_backend_available=not self.timing_failed)
            if self.active['metadata']:
                self._count('unresolved_metadata')
            if (not source_complete or not phases or not self.active['forwards'] or trace.dropped or trace.errors or trace.pending
                    or trace.operation._cancelled is not None):
                self._count('incomplete')
            if source_complete and not self.timing_failed and self.active['next_anchor'] is not None:
                identity = self.active['identity']
                self.previous = (identity['engine_epoch'], identity['sample'], self.active['next_anchor'])
            self._emit(not self.active['counts'])
        except Exception:
            self._count('finish_errors')
            self._emit(False)

    def _emit(self, complete, session=None):
        if session is None:
            session, self.active = self.active, None
        if self.blocked:
            self.previous = None
            return
        try:
            session['trace'].record('s', complete, session['counts'])
            lines = session['trace'].lines(**session['identity'])
            size = sum(len(x.encode()) + 97 for x in lines)  # receipt identity + newline
            if self.bytes + size > MAX_LOG_BYTES - 4096:
                self.blocked = True
                small = TimelineTrace('worker', session['trace'].rank)
                small.record('s', False, dict(log_budget_exhausted=1))
                lines = small.lines(**session['identity'])
                size = sum(len(x.encode()) + 97 for x in lines)
            self.bytes += size
            logger = __import__('logging').getLogger(__name__)
            if emit_receipt_lines(lines, logger) is False:
                self.blocked = True
        except Exception:
            self.blocked = True
        if self.blocked:
            self.previous = None


class OutsideSampler:
    """Engine-owned sparse pair selection; does not borrow native progress ownership."""
    def __init__(self):
        self.enabled = False
        self.epoch = None
        self.steps = self.samples = 0
        self.pending = None

    def wire(self, kind, sequence, args, outside):
        if kind == 'sample_tokens' and self.pending is not None:
            parent, self.pending = self.pending, None
            return [sequence, self.epoch, time.monotonic_ns(), parent]
        if kind == 'execute_model':
            self.pending = None  # An intervening execute makes sample ownership ambiguous.
        if kind != 'execute_model' or not outside or not self.enabled or self.samples >= MAX_OUTSIDE:
            return None
        rows = getattr(args[0], 'num_scheduled_tokens', None) if args else None
        if type(rows) is not dict or not rows or not all(type(n) is int and n == 1 for n in rows.values()):
            return None
        self.steps += 1
        if (self.steps - 1) % OUTSIDE_EVERY:
            return None
        self.samples += 1
        self.pending = sequence
        return [sequence, self.epoch, time.monotonic_ns(), sequence]


def engine_step(engine):
    executor = getattr(engine, 'model_executor', None)
    if executor is None:
        return
    side = executor.__dict__.get('_flash_s5_decode_timeline')
    if side is None:
        side = OutsideSampler()
        executor._flash_s5_decode_timeline = side
    driver = getattr(engine, '_flash_s5_driver', None)
    policy = getattr(driver, 'auto_arm', None)
    side.epoch = getattr(driver, 'engine_epoch', None)
    route = getattr(driver, '_control_route', None)
    manual_split = (type(route) is dict and route.get('owner') == 'native-control'
                    and route.get('intent') == 'enter_split' and driver._split_ready is True
                    and route.get('engine_epoch') == driver.engine_epoch
                    and route.get('route_generation') == driver.route_generation)
    side.enabled = (policy is not None and (policy.manual_owned is False or manual_split)
                    and type(side.epoch) is int and side.epoch >= 0)


def worker_rpc(proc, method, tag, recorder):
    if recorder is None or recorder.timeline is None:
        return None
    observer = recorder.timeline
    observer.poll_outside()
    if tag is None:
        return None
    if type(tag) is not list or len(tag) != 4 or any(type(x) is not int or x < 0 for x in tag):
        recorder.counters['observer_errors'] += 1
        return None
    sequence, epoch, sent_ns, parent = tag
    if method == 'execute_model' and sequence == parent:
        observer.outside_begin(sequence, epoch, sent_ns)
    return sequence if method in ('execute_model', 'sample_tokens') else None
