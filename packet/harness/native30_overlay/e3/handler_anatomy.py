"""Stage-1 stock-handler anatomy: bounded, default-off seam timing on the worker.

Switch ``FLASH_S5_HANDLER_ANATOMY=1``; active only while the engine's progress
meta rides on the RPC (``FLASH_S5_NATIVE_PROGRESS`` set), keyed by the same
transport sequence. With the switch unset ``executor_hooks`` never imports this
module (see ``tools/stage1_anatomy_freeze.py``).

What it records, per stock RPC (``execute_model``, ``sample_tokens``, other) on
every worker: host enter/exit (the progress RPC state's own stamps), thread CPU
time over the RPC, and for every wrapped seam that fired inside it the summed
inclusive ns, exclusive ns (minus nested seams), thread CPU ns and the call
count. The forward launch and the runner's ``sample_tokens`` also carry a CUDA
event pair on the current stream (device time, read lazily) and the forward a
``Stream.query()`` before and after the launch. On the reply path it records the
main-thread ``handle_output`` time and, as separate rows keyed by the same
sequence, the async output thread's ``get_output`` and response-MQ write.

Hard rules (checked by ``harness/test_stage1_handler_anatomy.py``):

* the measured path adds no host sync: it calls only ``time.monotonic_ns``,
  ``time.thread_time_ns``, ``Event.record``, ``Event.query``, ``Stream.query``
  and, after both events of a pair query True, ``Event.elapsed_time``;
* everything is bounded: ``EVENT_PAIRS`` event pairs, ``MAX_ROWS`` rows per
  flush, drops counted, never blocking, never growing;
* wrappers are set on instances (worker's model runner, its model, graph
  managers, PP handler, model state, the PP group coordinator, the response MQ
  and the async output queue), never on classes; installed once per process;
  exceptions and return values pass through unchanged (PP comm handles come back
  wrapped in a delegating proxy, as briefed, so their ``wait`` can be timed).

Lines: ``flash-s5 handler `` + JSON, schema ``s5-handler-anatomy/1``, split into
parts below ``native_measurement.PROGRESS_LINE_BYTES`` like ``ProgressTrace``.
"""
import json
import os
import threading
import time
from collections import deque

ANATOMY_ENV = 'FLASH_S5_HANDLER_ANATOMY'
SCHEMA = 's5-handler-anatomy/1'
LINE_PREFIX = 'flash-s5 handler '
LINE_BYTES = 12000            # == native_measurement.PROGRESS_LINE_BYTES
MAX_ROWS = 4096
EVENT_PAIRS = 64
CAL_PROBES = 2000
CAL_EVENT_PAIRS = 100

KINDS = {'execute_model': 'e', 'sample_tokens': 's'}

# Seam names, by id. Nesting is resolved by the exclusive column; the analyser
# documents the expected parents (``tools/stage1_anatomy_report.py``).
SEAMS = ('pp_send_wait', 'pp_recv', 'pp_recv_wait', 'pp_recv_post', 'pp_send', 'execute', 'prep_states',
         'prep_inputs', 'prep_attn', 'prep_attn_meta', 'forward', 'logits', 'sample_tokens', 'sample',
         'bookkeeping', 'post_sampled', 'pp_tok_send', 'pp_tok_recv')
SEAM_ID = {name: index for index, name in enumerate(SEAMS)}

# (owner, attribute, seam, mode). Missing owners/attributes are skipped and named
# in the header: V1 (gpu_model_runner.GPUModelRunner) and V2
# (gpu/model_runner.GPUModelRunner) differ, and the S5 split bank has its own
# graph manager.
TARGETS = (
    ('pp', 'irecv_tensor_dict', 'pp_recv', 'recv'),
    ('pp', 'isend_tensor_dict', 'pp_send', 'send'),
    ('runner', 'execute_model', 'execute', 'plain'),
    ('runner', 'sample_tokens', 'sample_tokens', 'sample'),
    ('runner', '_model_forward', 'forward', 'forward'),                                   # V1
    ('cgm', 'run_fullgraph', 'forward', 'forward'),                                       # V2 FULL
    ('cgm', 'run_pw_graph', 'forward', 'forward'),                                        # V2 piecewise
    ('split_cgm', 'run_fullgraph', 'forward', 'forward'),                                 # S5 split bank
    ('model', 'compute_logits', 'logits', 'plain'),
    ('model', 'compute_logits_local', 'logits', 'plain'),
    ('runner', '_sample', 'sample', 'plain'),                                             # V1
    ('runner', 'sample', 'sample', 'plain'),                                              # V2
    ('runner', '_bookkeeping_sync', 'bookkeeping', 'plain'),                              # V1
    ('runner', 'postprocess_sampled', 'post_sampled', 'plain'),                           # V2
    ('runner', '_pp_broadcast_prev_sampled_token_ids', 'pp_tok_send', 'plain'),           # V1
    ('runner', '_pp_receive_prev_sampled_token_ids_to_input_batch', 'pp_tok_recv', 'plain'),  # V1
    ('pph', 'broadcast', 'pp_tok_send', 'plain'),                                         # V2
    ('pph', 'receive', 'pp_tok_recv', 'plain'),                                           # V2
    ('runner', '_prepare_inputs', 'prep_inputs', 'plain'),                                # V1
    ('runner', 'add_requests', 'prep_states', 'plain'),                                   # V2
    ('runner', 'update_requests', 'prep_states', 'plain'),                                # V2
    ('runner', 'prepare_inputs', 'prep_inputs', 'plain'),                                 # V2
    ('runner', 'prepare_attn', 'prep_attn', 'plain'),                                     # V2
    ('mstate', 'prepare_attn', 'prep_attn_meta', 'plain'),                                # V2
)

# Main row: one per stock RPC on this rank.
M_FIELDS = ('seq', 'kind', 'out', 'flags', 't_enter', 't_exit', 'wall_ns', 'cpu_ns', 'observer_ns',
            'dev_forward_ns', 'dev_sample_ns', 'idle_before_launch', 'idle_after_launch', 'handle_output_ns',
            'seams')
SEAM_FIELDS = ('seam', 'ns', 'excl_ns', 'cpu_ns', 'n')
# Output row: the reply path of one ordered stock RPC (async output thread, or the
# main thread when async scheduling is off).
A_FIELDS = ('seq', 'thread', 't_take', 'get_output_ns', 'get_output_cpu_ns', 'mq_write_ns', 't_emit',
            'observer_ns')
FLAG_FAILED, FLAG_ABORTED, FLAG_INSTALL, FLAG_DEV_UNRESOLVED, FLAG_NESTED = 1, 2, 4, 8, 16
COUNTERS = ('rows_dropped', 'events_pool_empty', 'events_unresolved_at_flush', 'events_late', 'outside_calls',
            'off_thread_calls', 'nested_calls', 'failed_rows', 'aborted_rows', 'log_failures', 'observer_errors',
            'get_output_unwrapped', 'async_unmatched')

# Row slots (main row is a list while open).
_SEQ, _KIND, _OUT, _FLAGS, _TE, _TX, _WALL, _CPU, _OBS, _DEVF, _DEVS, _QB, _QA, _HO, _SEAMS = range(15)
_C0 = 15  # thread CPU at begin (not written)


def enabled(env=None):
    return (env if env is not None else os.environ).get(ANATOMY_ENV, '') == '1'


def _load_torch():
    import torch
    return torch


def _load_groups():
    """(pp group, tp group) or (None, None) outside an initialised vLLM worker."""
    try:
        from vllm.distributed.parallel_state import get_pp_group, get_tp_group
        return get_pp_group(), get_tp_group()
    except Exception:  # noqa: BLE001 - no PP/TP group: those seams are skipped and named
        return None, None


class _Handle:
    """Delegating proxy over one comm handle; times ``wait`` into a seam."""

    __slots__ = ('_handle', '_recorder', '_seam')

    def __init__(self, handle, recorder, seam):
        self._handle, self._recorder, self._seam = handle, recorder, seam

    def wait(self, *args, **kwargs):
        return self._recorder._timed(self._seam, self._handle.wait, args, kwargs)

    def is_completed(self, *args, **kwargs):
        return self._handle.is_completed(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._handle, name)


class Recorder:
    """One per worker process; the main RPC thread owns rows, the stack and events."""

    def __init__(self, proc, torch, pp, tp, *, rows=MAX_ROWS, pairs=EVENT_PAIRS):
        self.proc, self.torch, self.pp, self.tp = proc, torch, pp, tp
        self.limit = rows
        self._lock = threading.Lock()
        self._local = threading.local()
        self.rows = []
        self.counters = dict.fromkeys(COUNTERS, 0)
        self.row = None          # the open main row (main thread)
        self.last = None         # the last returned main row, until its handle_output
        self.stack = []          # [seam_id, t_start, child_ns, cpu_start]
        self.active = [0] * len(SEAMS)
        self.main = None
        self.pending = deque()   # (row, slot, start, end)
        self.pool = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                     for _ in range(pairs)]
        self.installed, self.missing = [], []
        self.lines_out = 0
        self.fresh = True
        self.identity = {}
        self.calibration = {}

    # -- install -------------------------------------------------------------
    def install(self):
        self.calibration = self._calibrate()
        proc = self.proc
        worker = getattr(proc, 'worker', None)
        runner = getattr(worker, 'model_runner', None)
        router = getattr(runner, '_flash_e3_graph_router', None)
        owners = {
            'pp': self.pp, 'runner': runner, 'model': getattr(runner, 'model', None),
            'cgm': getattr(runner, 'cudagraph_manager', None),
            'split_cgm': getattr(getattr(router, 'backend', None), 'manager', None),
            'pph': getattr(runner, 'pp_handler', None), 'mstate': getattr(runner, 'model_state', None),
        }
        if owners['split_cgm'] is owners['cgm']:
            owners['split_cgm'] = None
        seen = set()
        for owner, attr, seam, mode in TARGETS:
            target = owners.get(owner)
            original = getattr(target, attr, None) if target is not None else None
            if not callable(original) or (id(target), attr) in seen:
                self.missing.append(owner + '.' + attr)
                continue
            seen.add((id(target), attr))
            try:
                setattr(target, attr, self._wrap(original, SEAM_ID[seam], mode))
            except (AttributeError, TypeError):
                self.missing.append(owner + '.' + attr + ':readonly')
                continue
            self.installed.append(owner + '.' + attr)
        mq = getattr(proc, 'worker_response_mq', None)
        if mq is not None and callable(getattr(mq, 'enqueue', None)):
            mq.enqueue = self._wrap_mq(mq.enqueue)
            self.installed.append('proc.worker_response_mq.enqueue')
        queue = getattr(proc, 'async_output_queue', None)
        if getattr(proc, 'use_async_scheduling', False) and queue is not None:
            queue.get = self._wrap_take(queue.get)
            self.installed.append('proc.async_output_queue.get')
        pp, tp = self.pp, self.tp
        self.identity = dict(
            rank=getattr(proc, 'rank', None),
            pp_rank=getattr(pp, 'rank_in_group', None), pp_size=getattr(pp, 'world_size', None),
            is_first_rank=getattr(pp, 'is_first_rank', None), is_last_rank=getattr(pp, 'is_last_rank', None),
            tp_rank=getattr(tp, 'rank_in_group', None), tp_size=getattr(tp, 'world_size', None),
            async_scheduling=bool(getattr(proc, 'use_async_scheduling', False)),
            runner_async_scheduling=getattr(runner, 'use_async_scheduling', None),
            runner=(type(runner).__module__ + '.' + type(runner).__name__) if runner is not None else None)
        return self

    def _calibrate(self):
        mono, thread = time.monotonic_ns, time.thread_time_ns
        sink = []
        t0 = mono()
        for _ in range(CAL_PROBES):
            sink.append((mono(), thread()))
        probe = (mono() - t0) / CAL_PROBES
        stream = self.torch.cuda.current_stream()
        start, end = self.torch.cuda.Event(enable_timing=True), self.torch.cuda.Event(enable_timing=True)
        t0 = mono()
        for _ in range(CAL_EVENT_PAIRS):
            start.record(stream)
            end.record(stream)
        pair = (mono() - t0) / CAL_EVENT_PAIRS
        return dict(probe_ns=round(probe, 1), probes=CAL_PROBES, event_pair_ns=round(pair, 1),
                    event_pairs=CAL_EVENT_PAIRS)

    # -- wrappers ------------------------------------------------------------
    def _wrap(self, original, seam, mode):
        if mode == 'recv':
            def recv(*args, **kwargs):
                out = self._timed(seam, original, args, kwargs)
                if self.row is None or threading.get_ident() != self.main or type(out) is not tuple or len(out) != 3:
                    return out
                tensors, handles, post = out
                return (tensors, [_Handle(h, self, SEAM_ID['pp_recv_wait']) for h in handles],
                        [self._post(fn) for fn in post])
            return recv
        if mode == 'send':
            def send(*args, **kwargs):
                out = self._timed(seam, original, args, kwargs)
                if self.row is None or threading.get_ident() != self.main or type(out) is not list:
                    return out
                return [_Handle(h, self, SEAM_ID['pp_send_wait']) for h in out]
            return send
        device = _DEVF if mode == 'forward' else _DEVS if mode == 'sample' else None
        query = mode == 'forward'

        def call(*args, **kwargs):
            return self._timed(seam, original, args, kwargs, device, query)
        return call

    def _post(self, fn):
        def post(*args, **kwargs):
            return self._timed(SEAM_ID['pp_recv_post'], fn, args, kwargs)
        return post

    def _timed(self, seam, fn, args, kwargs, device=None, query=False):
        t0 = time.monotonic_ns()
        row = self.row
        if row is None or threading.get_ident() != self.main:
            self.counters['outside_calls' if row is None else 'off_thread_calls'] += 1
            return fn(*args, **kwargs)
        if self.active[seam]:
            self.counters['nested_calls'] += 1
            row[_FLAGS] |= FLAG_NESTED
            return fn(*args, **kwargs)
        pair = stream = None
        try:
            if device is not None:
                stream = self.torch.cuda.current_stream()
                if query:
                    row[_QB] = int(stream.query())
                if self.pool:
                    pair = self.pool.pop()
                    pair[0].record(stream)
                else:
                    self.counters['events_pool_empty'] += 1
        except Exception:  # noqa: BLE001 - observation never changes the call
            self.counters['observer_errors'] += 1
        self.active[seam] += 1
        frame = [seam, 0, 0, time.thread_time_ns()]
        self.stack.append(frame)
        frame[1] = t1 = time.monotonic_ns()
        try:
            return fn(*args, **kwargs)
        finally:
            t2 = time.monotonic_ns()
            try:
                c2 = time.thread_time_ns()
                self.active[seam] -= 1
                if self.stack and self.stack[-1] is frame:
                    self.stack.pop()
                if pair is not None:
                    pair[1].record(stream)
                    self.pending.append((row, device, pair))
                if query and stream is not None:
                    row[_QA] = int(stream.query())
                inclusive = t2 - t1
                self._seam(row, seam, inclusive, inclusive - frame[2], c2 - frame[3])
                t3 = time.monotonic_ns()
                if self.stack:
                    self.stack[-1][2] += t3 - t0
                row[_OBS] += (t1 - t0) + (t3 - t2)
            except Exception:  # noqa: BLE001 - observation never changes the call
                self.counters['observer_errors'] += 1

    @staticmethod
    def _seam(row, seam, inclusive, exclusive, cpu):
        cell = row[_SEAMS].get(seam)
        if cell is None:
            row[_SEAMS][seam] = [inclusive, exclusive, cpu, 1]
        else:
            cell[0] += inclusive
            cell[1] += exclusive
            cell[2] += cpu
            cell[3] += 1

    def _wrap_mq(self, original):
        local = self._local

        def enqueue(*args, **kwargs):
            t0 = time.monotonic_ns()
            try:
                return original(*args, **kwargs)
            finally:
                local.mq = time.monotonic_ns() - t0
        return enqueue

    def _wrap_take(self, original):
        local = self._local

        def get(*args, **kwargs):
            out = original(*args, **kwargs)
            local.take = time.monotonic_ns()
            return out
        return get

    def _wrap_get_output(self, original, sequence):
        local = self._local

        def get_output(*args, **kwargs):
            t0, c0 = time.monotonic_ns(), time.thread_time_ns()
            try:
                return original(*args, **kwargs)
            finally:
                local.get = (sequence, time.monotonic_ns() - t0, time.thread_time_ns() - c0)
        return get_output

    # -- hook entry points (called by executor_hooks under the switch) --------
    def begin(self, sequence, name):
        t0 = time.monotonic_ns()
        if self.row is not None:
            self._close(self.row, None, FLAG_ABORTED)
            self.counters['aborted_rows'] += 1
        self.stack.clear()
        self.active = [0] * len(SEAMS)
        self.last = None
        self.main = threading.get_ident()
        self._resolve()
        if name.startswith('flash_s5_'):
            self.row = None
            return
        flags = FLAG_INSTALL if self.fresh else 0
        self.fresh = False
        row = [sequence, KINDS.get(name, 'o'), None, flags, None, None, None, None, 0, None, None, None, None,
               None, {}, 0]
        row[_OBS] = time.monotonic_ns() - t0
        row[_C0] = time.thread_time_ns()
        self.row = row

    def returned(self, rpc, output_rank):
        row = self.row
        if row is None or row[_SEQ] != rpc[0]:
            return
        c1 = time.thread_time_ns()
        row[_TE], row[_TX] = rpc[2], rpc[3]
        row[_CPU] = c1 - row[_C0]
        row[_OUT] = int(output_rank is None or self.proc.rank == output_rank)
        self._close(row, rpc[3], 0)
        self.last = row

    def handle_output(self, proc, rpc, output):
        """Time the stock ``handle_output`` for this RPC; wrap an async output's ``get_output``."""
        t0 = time.monotonic_ns()
        row = self.last if self.last is not None and self.last[_SEQ] == rpc[0] else None
        try:
            if isinstance(output, BaseException):
                if self.row is not None and self.row[_SEQ] == rpc[0]:
                    row = self.row
                    row[_OUT] = 1
                    row[_TE] = rpc[2]
                    row[_CPU] = time.thread_time_ns() - row[_C0]
                    self._close(row, t0, FLAG_FAILED)
                    self.counters['failed_rows'] += 1
            elif row is not None and callable(getattr(output, 'get_output', None)):
                try:
                    output.get_output = self._wrap_get_output(output.get_output, rpc[0])
                except (AttributeError, TypeError):
                    self.counters['get_output_unwrapped'] += 1
        except Exception:  # noqa: BLE001
            self.counters['observer_errors'] += 1
        t1 = time.monotonic_ns()
        try:
            return proc.handle_output(output)
        finally:
            t2 = time.monotonic_ns()
            if row is not None:
                row[_HO] = t2 - t1
                row[_OBS] += t1 - t0
            self.last = None

    def replied(self, rpc):
        """After an ordered reply's response-MQ write: one output row, keyed by sequence."""
        t_emit = time.monotonic_ns()
        local = self._local
        got = getattr(local, 'get', None)
        take = getattr(local, 'take', None)
        mq = getattr(local, 'mq', None)
        local.get = local.take = local.mq = None
        if got is not None and got[0] != rpc[0]:
            self.counters['async_unmatched'] += 1
            got = None
        thread = 1 if threading.get_ident() == self.main else 0
        self._append(('a', rpc[0], thread, take, got[1] if got else None, got[2] if got else None, mq, t_emit,
                      time.monotonic_ns() - t_emit))

    def _close(self, row, t_exit, flags):
        row[_FLAGS] |= flags
        if row[_TX] is None and t_exit is not None:
            row[_TX] = t_exit
        if row[_TE] is not None and row[_TX] is not None:
            row[_WALL] = row[_TX] - row[_TE]
        self.row = None
        self._append(row)

    def _append(self, row):
        with self._lock:
            if len(self.rows) >= self.limit:
                self.counters['rows_dropped'] += 1
                return
            self.rows.append(row)

    def _resolve(self):
        """Read every completed event pair; never waits."""
        kept = deque()
        while self.pending:
            row, slot, pair = self.pending.popleft()
            try:
                if pair[1].query() and pair[0].query():
                    row[slot] = int(round(pair[0].elapsed_time(pair[1]) * 1e6))
                    if row[_FLAGS] & FLAG_DEV_UNRESOLVED:
                        self.counters['events_late'] += 1
                    self.pool.append(pair)
                    continue
            except Exception:  # noqa: BLE001
                self.counters['observer_errors'] += 1
                continue
            kept.append((row, slot, pair))
        self.pending = kept

    # -- flush ---------------------------------------------------------------
    def flush(self, logger):
        """At flash_s5_release, after the progress lines: this rank's anatomy lines."""
        try:
            self._resolve()
            for row, _, _ in self.pending:
                if not row[_FLAGS] & FLAG_DEV_UNRESOLVED:
                    row[_FLAGS] |= FLAG_DEV_UNRESOLVED
                    self.counters['events_unresolved_at_flush'] += 1
            with self._lock:
                rows, self.rows = self.rows, []
            for text in self.lines(rows):
                logger.info(text)
        except Exception:  # noqa: BLE001 - logging failures are counted, never raised
            self.counters['log_failures'] += 1

    def lines(self, rows):
        main = [r for r in rows if type(r) is list]
        out_rows = [r for r in rows if type(r) is tuple]
        times = [t for r in main for t in (r[_TE], r[_TX]) if type(t) is int]
        times += [t for r in out_rows for t in (r[3], r[7]) if type(t) is int]
        base = min(times) if times else 0

        def us(value):
            return (value - base) // 1000 if type(value) is int else None
        encoded = []
        for r in main:
            seams = [[s] + cell for s, cell in sorted(r[_SEAMS].items())]
            encoded.append(('m', json.dumps([r[_SEQ], r[_KIND], r[_OUT], r[_FLAGS], us(r[_TE]), us(r[_TX]), r[_WALL],
                                             r[_CPU], r[_OBS], r[_DEVF], r[_DEVS], r[_QB], r[_QA], r[_HO], seams],
                                            separators=(',', ':'))))
        for r in out_rows:
            encoded.append(('a', json.dumps([r[1], r[2], us(r[3]), r[4], r[5], r[6], us(r[7]), r[8]],
                                            separators=(',', ':'))))
        head = dict(schema=SCHEMA, rank=self.identity.get('rank'), line=self.lines_out, base_ns=base)
        first = dict(identity=self.identity, calibration=self.calibration, counters=dict(self.counters),
                     installed=self.installed, missing=self.missing, seams=SEAMS,
                     fields=dict(m=M_FIELDS, a=A_FIELDS, seam=SEAM_FIELDS))
        first_text = json.dumps(first, separators=(',', ':'), sort_keys=True)
        budget = LINE_BYTES - len(LINE_PREFIX) - len(json.dumps(head)) - 256
        parts, current, size = [], {}, len(first_text) + 32
        for tag, text in encoded:
            if current and size + len(text) + 8 > budget:
                parts.append(current)
                current, size = {}, 0
            current.setdefault(tag, []).append(text)
            size += len(text) + 1
        parts.append(current)
        out = []
        for index, part in enumerate(parts):
            body = dict(head, part=index, parts=len(parts), rows='@ROWS@')
            if index == 0:
                body['head'] = '@HEAD@'
            text = json.dumps(body, separators=(',', ':'), sort_keys=True)
            rows_text = '{' + ','.join('"%s":[%s]' % (tag, ','.join(items)) for tag, items in sorted(part.items())) + '}'
            text = text.replace('"@ROWS@"', rows_text, 1)
            if index == 0:
                text = text.replace('"@HEAD@"', first_text, 1)
            out.append(LINE_PREFIX + text)
        self.lines_out += 1
        return out


def install(proc):
    """The process recorder: wrappers set once on this worker's instances."""
    pp, tp = _load_groups()
    return Recorder(proc, _load_torch(), pp, tp, rows=MAX_ROWS, pairs=EVENT_PAIRS).install()


__all__ = ['ANATOMY_ENV', 'A_FIELDS', 'COUNTERS', 'EVENT_PAIRS', 'LINE_PREFIX', 'MAX_ROWS', 'M_FIELDS', 'Recorder',
           'SCHEMA', 'SEAMS', 'SEAM_FIELDS', 'TARGETS', 'enabled', 'install']
