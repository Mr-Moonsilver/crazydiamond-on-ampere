"""Bounded, source-derived long-window native measurement (CPU only).

The raw ``NativeSegmentDriver`` journal is deliberately a *detailed* bounded
ring (``deque(maxlen=1024)``) and its loss counters are the loss guard for that
ring.  A 200k prompt needs ~196 chunks at the unchanged 1024-token size and many
accepted segments per chunk, so the beginning of a real W2 window leaves the
ring long before ``overlap_end``.  That guard must never be weakened.

This module owns a *separate*, independent, positive accounting of the same
engine-validated events.  It is fed from the same validated ``_record`` entries
as the raw journal but it never reads, repairs, enlarges or reclassifies the raw
ring.  Its completeness basis is its own bounded ingest census plus exact
per-chunk reconciliation:

* bounded capacities cover the real maximum request/plan shape (262144-token
  model limit / 1024-token chunks = 256 chunks per request; the driver's
  *unexpanded* plan is ``10 + layer_count`` segments, 58 for the 48-layer
  model).  A real chunk may contain more accepted segment events than the plan:
  a paused QSA continuation is re-issued and accepted with a **new stage-local
  op_seq** for the same descriptor, so segment counts are never equated to the
  plan length;
* ``OpSeqTracker`` is per worker (therefore per PP stage).  Two stages may both
  accept op_seq 0, 1, ... for different descriptors, so segment identity is
  ``(stage, op_seq)``, never ``op_seq`` alone;
* one compact record per accepted chunk retains the actual first accepted work
  time, the **first queried** proof observation for every logical proof, the
  actual completed op set, the commit, the source/PP retirement and the
  committed frontier;
* completeness of the segment census is a **source join**, not a count: for
  each declared stage, the retained accepted ``(stage, op_seq)`` set must equal
  every rank's actual ``completed_op_seqs`` set and its ``gpu_event`` proof set;
  TP peers in a stage must agree; the queried ``state_write`` proof must name a
  retained ``state_finalize`` segment, and a queried ``pp_consumer`` proof must
  name a retained ``pp_receive`` segment of the receiving stage.  The join is
  evaluated at reconciliation time, so a cumulative report observed before the
  final all-target acceptance is retained and joined later, never against a
  last-active or partial set;
* cumulative worker reports are the real source shape (``worker_bridge.
  _build_report`` repeats every completed proof on every poll).  A repeated
  identical proof is counted as a repeated observation, never as a new
  completion; it never changes the retained first-observation time and never
  grows retained state quadratically;
* request family is decided by the request generation's *initial* frontier:
  ``0`` is a fresh request, ``>0`` is a cached/append request.  Later chunks of
  the same fresh request keep ``fresh`` even though their ``committed_end`` is
  positive.  A UID reused with another generation is a distinct request;
* conflicting identity, epoch, route, proof, frontier or ordering facts are
  retained as reasons and refuse the chunk.  Cancelled/partial chunks stay
  visible and are never counted as completed throughput;
* unsupported, malformed or over-capacity input produces explicit
  ``INCONCLUSIVE`` evidence.  :meth:`NativeMeasurementCensus.ingest` and
  :meth:`NativeMeasurementCensus.evidence` never raise into the engine launch
  path.

No device, torch, CUDA, vLLM, network or box access is used.  Every retained
timestamp is a host ``CLOCK_MONOTONIC`` nanosecond value copied from the engine
entry; no duration is converted or invented.
"""

import json
import os
import sys
import time
from contextlib import contextmanager


CONTROL_PATH_FIELDS = ('prepare_ns', 'poll_ns', 'validate_ns', 'serialize_ns', 'ops_count')


class ControlPathCounters:
    """Exclusive host-wall counters; descriptive only, never a readiness gate.

    One engine tick owns this object. Nested spans subtract their elapsed time
    from the parent, so the four buckets can be added without double counting.
    Serialize includes bounded precheck/pickle and inline MQ enqueue; validate
    includes reply validation/application and its measurement bookkeeping.
    These are not CPU cycles, GPU durations or a complete engine wall clock.
    """
    def __init__(self, engine_epoch, clock=time.perf_counter_ns):
        self.engine_epoch = engine_epoch
        self.clock = clock
        self.values = dict.fromkeys(CONTROL_PATH_FIELDS, 0)
        self._stack = []

    @contextmanager
    def span(self, field):
        frame = [self.clock(), 0]
        self._stack.append(frame)
        try:
            yield
        finally:
            elapsed = max(0, self.clock() - frame[0])
            self._stack.pop()
            self.values[field] += max(0, elapsed - frame[1])
            if self._stack:
                self._stack[-1][1] += elapsed

    def snapshot(self):
        return dict(schema='s5-control-path-cost/1', engine_epoch=self.engine_epoch,
                    basis='exclusive EngineCore host wall; prepare includes admission/RPC prep; serialize is outbound precheck/enqueue; validate is reply application; worker CPU/response serialization excluded; ops_count is submitted RPCs',
                    **self.values)



class QsaControlCounters:
    """Bounded per-layer engine-lifetime telemetry, never a control predicate.

    A command/poll is one all-rank RPC, not one message per worker. Accepted
    boundaries count once after all active-stage peers agree; bucket commands
    include copy waves, pending retries and the host-only completion pull.
    Agreed bucket/wave counts describe the queried plan, not GPU completions.
    Rows are fixed by model geometry, so request/chunk count cannot grow storage.
    """
    FIELDS = ('layer_commands', 'layer_pending_rounds', 'boundary_commands',
              'profile_polls', 'profile_retry_polls', 'profile_pending_rounds',
              'bucket_commands', 'bucket_pending_rounds', 'bucket_accepted_commands',
              'tail_boundaries', 'tail_commands', 'tail_pending_rounds',
              'tail_accepted_commands',
              'agreed_buckets', 'agreed_copy_waves', 'profile_mismatches',
              'profile_timeouts', 'profile_worker_host_ns')

    def __init__(self, engine_epoch, stage_layers):
        self.engine_epoch = engine_epoch
        self.rows = {(stage, layer): dict.fromkeys(self.FIELDS, 0)
                     for stage, span in stage_layers.items() for layer in range(*span)}
        self.faults = 0

    def add(self, stage, layer, **counts):
        try:
            row = self.rows[(stage, layer)]
            if any(k not in row or type(v) is not int or v < 0 for k, v in counts.items()):
                raise ValueError('invalid telemetry increment')
            for key, value in counts.items():
                row[key] += value
        except Exception:
            # Descriptive loss remains visible without changing engine behavior.
            self.faults += 1

    def snapshot(self):
        return dict(schema='s5-qsa-control-counts/1', engine_epoch=self.engine_epoch,
            basis='engine-lifetime monotonic all-rank RPC/cohort counts and summed worker CPU profile-query nanoseconds; subtract matching status snapshots for a window; queried plan counts are not completion evidence or engine wall time',
            telemetry_faults=self.faults,
            layers=[dict(stage=stage, layer=layer, **values)
                    for (stage, layer), values in sorted(self.rows.items())])


def control_path_delta(begin, end):
    """Descriptive status-snapshot delta; absent telemetry changes no gate."""
    if (type(begin) is not dict or type(end) is not dict
            or begin.get('schema') != 's5-control-path-cost/1'
            or end.get('schema') != begin.get('schema')
            or begin.get('engine_epoch') != end.get('engine_epoch')):
        return {'available': False, 'reason': 'control-path-snapshots-unavailable'}
    result = {}
    for key in CONTROL_PATH_FIELDS:
        a, b = begin.get(key), end.get(key)
        if type(a) is not int or type(b) is not int or a < 0 or b < a:
            return {'available': False, 'reason': 'control-path-counter-regressed-or-invalid'}
        result[key] = b - a
    return dict(available=True, schema='s5-control-path-cost/1',
                engine_epoch=end['engine_epoch'], basis=end.get('basis'),
                interval='begin status snapshot to end evidence snapshot; includes boundary RPC coordination',
                **result)


# --- native progress receipts (held-f) ---------------------------------------
# ``FLASH_S5_NATIVE_PROGRESS`` unset/0 installs nothing.  ``trace`` records host
# timing receipts of native commands and stock RPCs while an auto-arm-owned
# native episode is active; ``burst`` additionally installs the switchable
# progress candidate.  Receipts never decide anything and read only
# ``time.monotonic_ns`` (one CLOCK_MONOTONIC domain for every process of the
# engine container); no CUDA call is made.

PROGRESS_ENV = 'FLASH_S5_NATIVE_PROGRESS'
PROGRESS_MODES = ('trace', 'burst')
PROGRESS_SCHEMA = 's5-native-progress/1'
PROGRESS_LINE = 'flash-s5 progress '
PROGRESS_MAX_ROWS = 4096
# One receipt record (one part) per atomic pipe write: at most this many bytes
# including the pid/seq/sha identity and the trailing newline (alpha-next PLAN 2).
PROGRESS_LINE_BYTES = 3968
PROGRESS_META_KEY = '_flash_s5'
# Fixed-width wire meta: transport sequence (u64) and the direct-reply bit.  The
# pickled size never depends on the value, so a size precheck is exact.
PROGRESS_META_BYTES = 9
# Worker methods whose reply may be written direct (the engine's segment,
# profile and poll kinds). Checked when the engine builds the envelope and
# again by the worker before dispatch; any other method refuses a direct bit.
PROGRESS_DIRECT_METHODS = frozenset(('flash_s5_submit', 'flash_s5_poll'))
PROGRESS_ROWS = {
    # engine: one native command, transport sequence through owner apply
    'n': ('seq', 'kind', 'segment', 'op_seq', 'stage', 'layer', 'origin', 't_submit', 't_sent',
          't_recv0', 't_recv1', 't_recv2', 't_recv3', 't_apply', 'outcome', 'direct'),
    # engine: one ordinary RPC (execute_model, sample_tokens, ...)
    's': ('seq', 'kind', 't_submit', 't_recv', 't_take'),
    # engine: one owner wait on an ordinary future
    'w': ('seq', 'kind', 't_start', 't_end', 'sends', 'capped'),
    # worker: one RPC on this rank, handler enter/exit and the response-MQ write
    'r': ('seq', 'kind', 't_enter', 't_exit', 't_emit', 'direct', 'replied'),
}


# Positions in an 'n' row (the engine fills it across submit, drain and apply).
PROGRESS_N = {name: index for index, name in enumerate(PROGRESS_ROWS['n'])}


# --- receipt writer (alpha-next PLAN 2: producer-side log fix) ------------------
# Every worker process and the engine share one container stdout pipe. A receipt
# line that went through ``logging`` -> vLLM's ``decorate_logs`` stdout wrapper
# -> the buffered ``sys.stdout`` left the process in several ``write(2)`` calls
# once it was longer than the buffer or ``PIPE_BUF``, and another process's line
# could land between them (alpha-perf ``engine-stream.log`` line 3832).
#
# ``ReceiptWriter`` writes each receipt record (one progress/anatomy/handler
# part, one ``native-alloc``, ``memory`` or ``stage-local`` line) with exactly
# one ``os.write`` on a private ``dup(1)``, at most ``RECEIPT_RECORD_BYTES``
# including the newline. POSIX makes a FIFO write of at most ``PIPE_BUF`` bytes
# atomic, so no other process's bytes can land inside a record. A short or
# failed write is counted, never raised and never retried (a retry would be a
# second, non-atomic write). Each record carries its identity:
#
# * ``pid``;
# * ``seq``: per process, strictly +1 across every record of every kind,
#   starting at 0 with the process's ``flash-s5 receipts`` path record;
# * ``sha``: sha256[:16] of the whole record before identity (all its parts,
#   joined by newlines), on every part.
#
# JSON records get ``"pid":..,"seq":..,"sha":"..",`` inserted after the opening
# brace; ``k=v`` records get `` pid=.. seq=.. sha=..`` appended. The prefixes
# are unchanged, so existing parsers still read every part.
#
# The writer is installed per process by the startup seams (``s5_startup``).
# A process where it was never installed (CPU tests that drive the emitters
# directly) keeps the old logger lines byte for byte; they carry no identity,
# and ``harness/receipt_lines.py`` reports them as ``unsafe_path`` legacy.

RECEIPT_RECORD_BYTES = 3968
# Room ``ProgressTrace.lines`` leaves for the identity and the newline.
RECEIPT_IDENTITY_BYTES = 96
RECEIPT_PATH_LINE = 'flash-s5 receipts '
RECEIPT_JSON_PREFIXES = (PROGRESS_LINE, 'flash-s5 anatomy ', 'flash-s5 handler ')
RECEIPT_KV_PREFIXES = (RECEIPT_PATH_LINE, 'flash-s5 native-alloc ', 'flash-s5 memory ',
                       'flash-s5 stage-local ')
RECEIPT_PIPE_BUF_MIN = 4096


def receipt_sha(texts):
    """sha256[:16] of one whole record: its pre-identity parts joined by newlines."""
    import hashlib
    return hashlib.sha256('\n'.join(texts).encode()).hexdigest()[:16]


def receipt_path(fd=1, *, limit=RECEIPT_RECORD_BYTES):
    """``atomic-fifo pipe_buf=<n>`` or ``unsafe:<reason>``; reads only fstat and PIPE_BUF."""
    import select
    import stat
    try:
        mode = os.fstat(fd).st_mode
    except OSError as exc:
        return 'unsafe:fstat-%s' % (exc.errno,)
    pipe_buf = getattr(select, 'PIPE_BUF', None)
    if not stat.S_ISFIFO(mode):
        return 'unsafe:not-fifo'
    if type(pipe_buf) is not int or pipe_buf < RECEIPT_PIPE_BUF_MIN:
        return 'unsafe:pipe-buf-%s' % (pipe_buf,)
    if limit > pipe_buf:
        return 'unsafe:record-%d-over-pipe-buf-%d' % (limit, pipe_buf)
    return 'atomic-fifo pipe_buf=%d' % pipe_buf


class ReceiptWriter:
    """One atomic ``os.write`` per receipt record on a private ``dup(1)``; never raises.

    ``fd`` defaults to a private duplicate of fd 1 made at the first write, so
    nothing is opened or written while receipts are off. ``write`` is
    ``os.write`` (a test seam). Records stay in ``seq`` order on the pipe: the
    lock covers numbering and writing. After a fork the child starts its own
    numbering (and its own path record) at 0.
    """

    def __init__(self, fd=None, *, write=os.write, limit=RECEIPT_RECORD_BYTES, getpid=os.getpid):
        import threading
        if type(limit) is not int or not 128 <= limit <= RECEIPT_RECORD_BYTES:
            raise ValueError('receipt record limit must be within the atomic budget')
        self._fd_given = fd
        self._write = write
        self._getpid = getpid
        self.limit = limit
        self._lock = threading.Lock()
        self.pid = None
        self.fd = None
        self.path = None
        self.seq = 0
        self.records = 0
        self.short_writes = 0
        self.write_errors = 0
        self.oversize = 0
        self.faults = 0

    def _start(self):
        """First write in this process: the private fd, the path check, record seq 0."""
        pid = self._getpid()
        if self.pid == pid:
            return
        self.pid = pid
        self.seq = 0
        self.fd = self._fd_given
        if self.fd is None:
            try:
                self.fd = os.dup(1)
            except OSError:
                self.fd = None
                self.write_errors += 1
        self.path = (receipt_path(self.fd, limit=self.limit) if self.fd is not None
                     else 'unsafe:no-fd')
        self._emit_kv(RECEIPT_PATH_LINE + 'receipt_path=' + self.path)

    def _next(self):
        seq = self.seq
        self.seq += 1
        return seq

    def _put(self, data):
        if len(data) > self.limit:
            self.oversize += 1
            return False
        if self.fd is None:
            self.write_errors += 1
            return False
        try:
            written = self._write(self.fd, data)
        except OSError:
            self.write_errors += 1
            return False
        if written != len(data):
            self.short_writes += 1
            return False
        self.records += 1
        return True

    def _emit_kv(self, text):
        seq = self._next()
        data = ('%s pid=%d seq=%d sha=%s\n' % (text, self.pid, seq, receipt_sha([text]))).encode()
        return self._put(data)

    def write_kv(self, text):
        """One ``k=v`` record (one line, no newline inside); False when not written whole."""
        try:
            with self._lock:
                self._start()
                if type(text) is not str or '\n' in text or not text.startswith(RECEIPT_KV_PREFIXES):
                    self._next()
                    self.faults += 1
                    return False
                return self._emit_kv(text)
        except Exception:
            self.faults += 1
            return False

    def write_lines(self, texts):
        """One JSON record of one or more parts (``ProgressTrace.lines`` output); one write per part."""
        try:
            texts = list(texts)
            if not texts:
                return True
            with self._lock:
                self._start()
                prefix = next((p for p in RECEIPT_JSON_PREFIXES if texts[0].startswith(p)), None)
                if (prefix is None or any(type(t) is not str or not t.startswith(prefix + '{')
                                          or '\n' in t for t in texts)):
                    for _ in texts:
                        self._next()
                    self.faults += 1
                    return False
                sha = receipt_sha(texts)
                ok = True
                cut = len(prefix) + 1
                for text in texts:
                    seq = self._next()
                    data = ('%s"pid":%d,"seq":%d,"sha":"%s",%s\n'
                            % (text[:cut], self.pid, seq, sha, text[cut:])).encode()
                    ok = self._put(data) and ok
                return ok
        except Exception:
            self.faults += 1
            return False

    def stats(self):
        return dict(pid=self.pid, path=self.path, seq=self.seq, records=self.records,
                    short_writes=self.short_writes, write_errors=self.write_errors,
                    oversize=self.oversize, faults=self.faults)


_RECEIPT_WRITER = None


def install_receipt_writer(writer=None):
    """This process's receipt writer (idempotent). Opens and writes nothing until a receipt."""
    global _RECEIPT_WRITER
    if writer is not None:
        _RECEIPT_WRITER = writer
    elif _RECEIPT_WRITER is None:
        _RECEIPT_WRITER = ReceiptWriter()
    return _RECEIPT_WRITER


def installed_receipt_writer():
    return _RECEIPT_WRITER


def emit_receipt_lines(texts, logger):
    """Every JSON receipt emit site: the installed writer, else the old logger lines."""
    writer = _RECEIPT_WRITER
    if writer is not None:
        return writer.write_lines(texts)
    for text in texts:
        logger.info(text)
    return True


def emit_receipt_kv(text, logger):
    """Every ``k=v`` receipt emit site: the installed writer, else the old logger line."""
    writer = _RECEIPT_WRITER
    if writer is not None:
        return writer.write_kv(text)
    logger.info(text)
    return True


def progress_mode(env):
    """None, 'trace' or 'burst'; refuses a value it does not know."""
    value = env.get(PROGRESS_ENV, '')
    if value in ('', '0'):
        return None
    if value not in PROGRESS_MODES:
        raise ValueError('invalid setting: ' + PROGRESS_ENV)
    if env.get('FLASH_S5_AUTO_ARM', '0') != '1':
        raise ValueError(PROGRESS_ENV + ' requires FLASH_S5_AUTO_ARM=1')
    return value


def encode_progress_meta(sequence, direct):
    if type(sequence) is not int or not 0 <= sequence < 1 << 64:
        raise ValueError('progress meta sequence out of range')
    return sequence.to_bytes(8, 'little') + (b'\x01' if direct else b'\x00')


def decode_progress_meta(meta):
    if type(meta) is not bytes or len(meta) != PROGRESS_META_BYTES or meta[8] not in (0, 1):
        raise ValueError('malformed progress meta')
    return int.from_bytes(meta[:8], 'little'), meta[8] == 1


class ProgressTrace:
    """Bounded per-process progress receipts, flushed as one log line per chunk.

    Rows are fixed tuples (``ROWS``, held-f's ``PROGRESS_ROWS``); time fields
    are absolute ``CLOCK_MONOTONIC`` ns and are written as microsecond offsets
    from the line's ``base_ns``.  A full buffer drops and counts, never blocks
    or grows. Workers record from two threads (RPC loop and async output),
    hence the lock. A subclass may carry its own row table, schema and line
    prefix (held-g's ``AnatomyTrace``).
    """
    ROWS = PROGRESS_ROWS
    SCHEMA = PROGRESS_SCHEMA
    LINE = PROGRESS_LINE

    def __init__(self, side, rank=None, *, limit=PROGRESS_MAX_ROWS):
        import threading
        if side not in ('engine', 'worker'):
            raise ValueError('progress side must be engine or worker')
        if type(limit) is not int or limit < 1:
            raise ValueError('progress row limit must be positive')
        self.side, self.rank, self.limit = side, rank, limit
        self._lock = threading.Lock()
        self.rows = []
        self.dropped = 0
        self.lines_out = 0
        self.dropped_total = 0

    def record(self, tag, *fields):
        if len(fields) != len(self.ROWS[tag]):
            raise ValueError('progress row shape differs from its schema')
        with self._lock:
            if len(self.rows) >= self.limit:
                self.dropped += 1
                self.dropped_total += 1
                return False
            self.rows.append((tag,) + fields)
            return True

    def take(self):
        with self._lock:
            rows, dropped = self.rows, self.dropped
            self.rows, self.dropped = [], 0
            return rows, dropped

    def lines(self, **identity):
        """This flush as compact log lines; [] when nothing was recorded or dropped.

        Each line leaves ``RECEIPT_IDENTITY_BYTES`` free below
        ``PROGRESS_LINE_BYTES``, so with the writer's pid/seq/sha and newline a
        part is one ``ReceiptWriter`` record below ``PIPE_BUF``. Parts share
        ``line``/``base_ns`` and carry ``part``/``parts``; ``fields`` and
        ``dropped`` ride on part 0 only.
        """
        rows, dropped = self.take()
        if not rows and not dropped:
            return []
        times = [value for row in rows for name, value in zip(self.ROWS[row[0]], row[1:])
                 if name.startswith('t_') and type(value) is int]
        base = min(times) if times else 0
        encoded = []
        for row in rows:
            item = [((value - base) // 1000 if type(value) is int else None)
                    if name.startswith('t_') else value
                    for name, value in zip(self.ROWS[row[0]], row[1:])]
            encoded.append((row[0], json.dumps(item, separators=(',', ':'), sort_keys=True)))
        head = dict(schema=self.SCHEMA, side=self.side, rank=self.rank, line=self.lines_out,
                    base_ns=base, **identity)
        budget = (PROGRESS_LINE_BYTES - RECEIPT_IDENTITY_BYTES - len(self.LINE)
                  - len(json.dumps(head)) - 256)
        first = len(json.dumps(self.ROWS, separators=(',', ':'))) + 32
        parts, current, size = [], {}, first
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
                body.update(fields=self.ROWS, dropped=dropped)
            text = json.dumps(body, separators=(',', ':'), sort_keys=True)
            rows_text = '{' + ','.join('"%s":[%s]' % (tag, ','.join(items))
                                       for tag, items in sorted(part.items())) + '}'
            out.append(self.LINE + text.replace('"@ROWS@"', rows_text, 1))
        self.lines_out += 1
        return out


# --- native anatomy receipts (held-g) ------------------------------------------
# The same FLASH_S5_NATIVE_PROGRESS setting adds one anatomy line per chunk per
# process, on a schema of its own (the held-f progress lines keep their format):
#
# * R1, each worker (DeviceSpans): every accepted lane command gets a start and
#   an end CUDA timing event on the lane prefill stream, from a pool allocated
#   at the worker's first transaction. After release's drain, one anchor event
#   recorded on that drained stream and synchronized maps them onto the host
#   CLOCK_MONOTONIC. Typed pending refusals are counted per reason, with the
#   host time they held each command.
# * R5, the engine (StepAnatomy): the engine step loop of the chunk, from the
#   existing step seams.
#
# Receipts never decide anything: a failed read or log line is counted, never
# raised into the lane or the engine.

ANATOMY_SCHEMA = 's5-native-anatomy/1'
ANATOMY_LINE = 'flash-s5 anatomy '
# Timed commands per chunk per worker (two events each, plus one anchor).
ANATOMY_POOL = 256
# One chunk in this many carries per-command (worker) and per-step (engine) rows.
ANATOMY_DETAIL_EVERY = 16
# Steps per chunk kept for the medians; later steps still count in the sums.
ANATOMY_MAX_STEPS = 8192
ANATOMY_ROWS = {
    # worker: this rank's chunk on its prefill stream, device times on the host clock
    'c': ('commands', 'timed', 'untimed', 'refused', 't_first', 't_last', 'busy_us', 'idle_us',
          'idle_qsa_us', 'read_us', 'cancelled', 'detail', 'errors'),
    # worker: the chunk's accepted commands of one class; refusals and the host
    # time they held a command count for the command they preceded
    'k': ('kind', 'count', 'busy_us', 'max_us', 'idle_before_us', 'refused', 'wait_us'),
    # worker: one QSA layer, from its pre to its tail
    'q': ('layer', 'commands', 'credit', 'credit_wait_us', 'grant', 'grant_wait_us',
          'lag_us', 'lag_max_us', 'busy_us', 'idle_us', 't_start', 't_end'),
    # worker: the chunk's typed pending refusals by reason
    'f': ('reason', 'count', 'wait_us'),
    # worker detail: one accepted command
    'x': ('op_seq', 'kind', 'layer', 'refused', 't_host', 't_start', 't_end'),
    # engine: the chunk's steps of one class
    'e': ('cls', 'steps', 'total_us', 'advance_us', 'schedule_us', 'wait_us', 'rest_us',
          'sends', 'rpcs', 'waits', 'median_us', 'max_us'),
    # engine detail: one step
    'p': ('cls', 't_begin', 'total_us', 'advance_us', 'schedule_us', 'wait_us', 'sends', 'rpcs'),
}
# The receipt class of an accepted LAYER command: the QsaLayerDriver phase it
# ran (its newest ``records`` entry), or 'layer' when no QSA phase ran (GDN).
ANATOMY_QSA = {'pre': 'qsa-pre', 'schedule_boundary': 'qsa-boundary', 'selection': 'qsa-selection',
               'copy_wave': 'qsa-copy', 'bucket': 'qsa-bucket', 'attended': 'qsa-attended',
               'pending': 'qsa-pending', 'tail': 'qsa-tail'}
# Typed pending refusals by the text the lane raises them with.
ANATOMY_REASONS = (('lane submission credit pending', 'credit'),
                   ('QSA schedule awaits active-stage TP agreement', 'grant'),
                   ('PP receive has not completed', 'pp'))


def anatomy_reason(exc):
    text = str(exc)
    for prefix, reason in ANATOMY_REASONS:
        if text.startswith(prefix):
            return reason
    return 'other'


class AnatomyTrace(ProgressTrace):
    ROWS = ANATOMY_ROWS
    SCHEMA = ANATOMY_SCHEMA
    LINE = ANATOMY_LINE


def _us(ns):
    return int(ns) // 1000


class DeviceSpans:
    """Held-g R1: device spans of one worker's lane commands on its prefill stream.

    The LaneDispatcher calls ``begin`` at open, ``start`` inside the lane scope
    before a command's handler, ``end`` after the command's completion event,
    ``refused``/``abandon`` when the command was not accepted, and ``release``
    after release's drain. ``event`` makes one CUDA timing event. A chunk with
    more than ``pool`` accepted commands times only its first ``pool``
    (``untimed`` counts the rest); nothing is ever refused for it.
    """

    def __init__(self, event, *, rank, emit, clock=time.monotonic_ns, pool=ANATOMY_POOL,
                 detail_every=ANATOMY_DETAIL_EVERY, emit_lines=None):
        if not callable(event) or not callable(emit) or not callable(clock):
            raise ValueError('device spans need event, emit and clock callables')
        if emit_lines is not None and not callable(emit_lines):
            raise ValueError('device spans emit_lines must be callable')
        # Alpha-next PLAN 2: given, one chunk's parts go out together as one
        # receipt record (``emit_receipt_lines``); else one ``emit`` per part.
        self.emit_lines = emit_lines
        if type(pool) is not int or pool < 1 or type(detail_every) is not int or detail_every < 1:
            raise ValueError('device span pool and detail cadence must be positive')
        self.event, self.emit, self.clock = event, emit, clock
        self.rank = rank
        self.pool_size, self.detail_every = pool, detail_every
        self.pool = None
        self.anchor = None
        self.trace = AnatomyTrace('worker', rank)
        self.chunks = 0
        self.errors = 0
        self.lines_emitted = 0
        self._clear(None, None)

    def _clear(self, transaction, stage):
        self.transaction = transaction
        self.stage = stage
        self.used = 0
        self.untimed = 0
        # (op_seq, kind, layer, slot, host_ns, refusals, {reason: [count, held_ns]})
        self.commands = []
        # op_seq -> [last reason, last refusal ns, {reason: [count, held_ns]}]
        self.waiting = {}
        # reason -> [count, held_ns], refusals of every command
        self.reasons = {}

    def begin(self, transaction, stage):
        if self.pool is None:
            self.pool = [(self.event(), self.event()) for _ in range(self.pool_size)]
            self.anchor = self.event()
        self.chunks += 1
        self._clear(transaction, stage)

    def start(self, stream, driver):
        """Before the handler's first launch: host stamp, start event, driver record count."""
        host = self.clock()
        records = getattr(driver, 'records', None)
        count = len(records) if type(records) is list else None
        slot = None
        if self.transaction is not None and self.used < self.pool_size:
            slot = self.used
            self.pool[slot][0].record(stream)
            self.used += 1
        return slot, host, count

    def _hold(self, episode, now):
        reason, last = episode[0], episode[1]
        if reason is not None:
            held = max(0, now - last)
            episode[2].setdefault(reason, [0, 0])[1] += held
            self.reasons.setdefault(reason, [0, 0])[1] += held

    def end(self, mark, stream, command, driver):
        """After the command's completion event: its end event and its row."""
        slot, host, count = mark
        if slot is not None:
            self.pool[slot][1].record(stream)
        else:
            self.untimed += 1
        kind = command.kind
        if kind == 'layer':
            records = getattr(driver, 'records', None)
            if type(records) is list and count is not None and len(records) > count:
                kind = ANATOMY_QSA.get(records[-1][0], 'qsa-other')
        episode = self.waiting.pop(command.op_seq, None)
        refusals, reasons = 0, {}
        if episode is not None:
            self._hold(episode, host)
            reasons = episode[2]
            refusals = sum(entry[0] for entry in reasons.values())
        self.commands.append((command.op_seq, kind, command.layer_start, slot, host, refusals, reasons))

    def refused(self, mark, command, exc):
        """A typed pending refusal: no op_seq consumed; the start slot is reused."""
        if mark is None:
            return
        slot, host, _count = mark
        if slot is not None and slot == self.used - 1:
            self.used -= 1
        reason = anatomy_reason(exc)
        episode = self.waiting.get(command.op_seq)
        if episode is None:
            episode = self.waiting[command.op_seq] = [None, None, {}]
        self._hold(episode, host)
        episode[0], episode[1] = reason, host
        episode[2].setdefault(reason, [0, 0])[0] += 1
        self.reasons.setdefault(reason, [0, 0])[0] += 1

    def abandon(self, mark):
        """A command that raised before acceptance gives its start slot back."""
        if mark is not None and mark[0] is not None and mark[0] == self.used - 1:
            self.used -= 1

    def release(self, stream, cancelled):
        """After release's drain: read the chunk's spans, log its line; never raises."""
        if self.transaction is None:
            return False
        try:
            began = self.clock()
            times = {}
            if self.used:
                anchor = self.anchor
                anchor.record(stream)
                anchor.synchronize()
                base = self.clock()
                for slot in range(self.used):
                    first, last = self.pool[slot]
                    times[slot] = (base - round(first.elapsed_time(anchor) * 1e6),
                                   base - round(last.elapsed_time(anchor) * 1e6))
            self._rows(times, self.clock() - began, cancelled)
            transaction = self.transaction
            texts = self.trace.lines(txn=transaction.txn_seq,
                                     request_generation=transaction.request_generation,
                                     chunk_start=transaction.chunk_start,
                                     chunk_end=transaction.chunk_end, stage=self.stage)
            if self.emit_lines is not None:
                self.emit_lines(texts)
                self.lines_emitted += len(texts)
            else:
                for text in texts:
                    self.emit(text)
                    self.lines_emitted += 1
        except Exception:
            self.errors += 1
            self.trace.take()
            return False
        finally:
            self._clear(None, None)
        return True

    def _rows(self, times, read_ns, cancelled):
        detail = (self.chunks - 1) % self.detail_every == 0
        kinds, layers = {}, {}
        busy = idle = idle_qsa = 0
        first = last = previous = None
        for op_seq, kind, layer, slot, host, refusals, reasons in self.commands:
            entry = kinds.setdefault(kind, [0, 0, 0, 0, 0, 0])
            entry[0] += 1
            entry[4] += refusals
            entry[5] += sum(item[1] for item in reasons.values())
            qsa = kind.startswith('qsa-')
            continuation = qsa and kind != 'qsa-pre'
            span = times.get(slot) if slot is not None else None
            gap = lag = None
            if span is not None:
                cost = max(0, span[1] - span[0])
                busy += cost
                entry[1] += cost
                entry[2] = max(entry[2], cost)
                if previous is not None:
                    gap = max(0, span[0] - previous)
                    lag = host - previous
                    idle += gap
                    entry[3] += gap
                    if continuation:
                        idle_qsa += gap
                first = span[0] if first is None else first
                last = span[1]
                previous = span[1]
            if qsa:
                row = layers.setdefault(layer, dict(commands=0, credit=0, credit_wait=0, grant=0,
                                                    grant_wait=0, lag=0, lag_max=0, busy=0,
                                                    idle=0, start=None, end=None))
                row['commands'] += 1
                for reason in ('credit', 'grant'):
                    count, held = reasons.get(reason, (0, 0))
                    row[reason] += count
                    row[reason + '_wait'] += held
                if span is not None:
                    row['busy'] += max(0, span[1] - span[0])
                    row['start'] = span[0] if row['start'] is None else row['start']
                    row['end'] = span[1]
                if continuation and gap is not None:
                    row['idle'] += gap
                    row['lag'] += max(0, lag)
                    row['lag_max'] = max(row['lag_max'], lag)
            if detail:
                self.trace.record('x', op_seq, kind, layer, refusals, host,
                                  None if span is None else span[0],
                                  None if span is None else span[1])
        refused = sum(entry[0] for entry in self.reasons.values())
        self.trace.record('c', len(self.commands), len(times), self.untimed, refused, first, last,
                          _us(busy), _us(idle), _us(idle_qsa), _us(read_ns),
                          int(cancelled is not None), int(detail), self.errors)
        for kind in sorted(kinds):
            count, cost, peak, before, refusals, held = kinds[kind]
            self.trace.record('k', kind, count, _us(cost), _us(peak), _us(before), refusals, _us(held))
        for layer in sorted(layers):
            row = layers[layer]
            self.trace.record('q', layer, row['commands'], row['credit'], _us(row['credit_wait']),
                              row['grant'], _us(row['grant_wait']), _us(row['lag']),
                              _us(max(0, row['lag_max'])), _us(row['busy']), _us(row['idle']),
                              row['start'], row['end'])
        for reason in sorted(self.reasons):
            count, held = self.reasons[reason]
            self.trace.record('f', reason, count, _us(held))


class StepAnatomy:
    """Held-g R5: the engine step loop of one auto-owned native chunk (owner thread).

    A step runs from one ``engine_idle_tick`` entry (the first seam inside
    ``step_with_batch_queue``) to the next. The chunk's first step starts at
    its begin (which runs inside an advance); its last ends at its flush.
    ``advanced`` stamps the before-step hook (``driver.advance``), ``scheduled``
    the return of ``scheduler.schedule`` (the park seam), ``waited`` one owner
    wait on an ordinary future. ``rest`` is the step's time outside those
    three: stock RPC submission, ``update_from_output``, the process loop and
    the idle tick's sleep. Classes: ``tick`` (the idle tick returned a bypass),
    ``idle`` (the held-g idle wait bypassed the step), ``advance`` (the driver's
    advance returned a bypass), ``empty`` (stock scheduled no token),
    ``stock`` (stock scheduled tokens), ``partial`` (the chunk ended inside the
    step before it was classified).
    """
    CLASSES = ('tick', 'idle', 'advance', 'empty', 'stock', 'partial')

    def __init__(self, counters, *, clock_ns=time.monotonic_ns, detail_every=ANATOMY_DETAIL_EVERY,
                 max_steps=ANATOMY_MAX_STEPS):
        self.counters = counters
        self.clock_ns = clock_ns
        self.detail_every = detail_every
        self.max_steps = max_steps
        self.trace = AnatomyTrace('engine', limit=max_steps + len(self.CLASSES))
        self.chunks = 0
        self.step = None
        self.closed = []
        self.sums = {}

    def reset(self):
        self.chunks += 1
        self.closed = []
        self.sums = {}
        self.step = self._open(self.clock_ns())

    def _open(self, now):
        # [t_begin, cls, advance, t_advanced, schedule, wait, waits, sends0, rpcs0]
        return [now, None, 0, None, 0, 0, 0, self.counters['commands'], self.counters['stock_rpcs']]

    def _close(self, now):
        step = self.step
        if step is None:
            return
        cls = step[1] or 'partial'
        total = max(0, now - step[0])
        sends = self.counters['commands'] - step[7]
        rpcs = self.counters['stock_rpcs'] - step[8]
        entry = self.sums.setdefault(cls, [0, 0, 0, 0, 0, 0, 0, 0, 0, []])
        entry[0] += 1
        entry[1] += total
        entry[2] += step[2]
        entry[3] += step[4]
        entry[4] += step[5]
        entry[5] += max(0, total - step[2] - step[4] - step[5])
        entry[6] += sends
        entry[7] += rpcs
        entry[8] += step[6]
        if len(entry[9]) < self.max_steps:
            entry[9].append(total)
        if (self.chunks - 1) % self.detail_every == 0 and len(self.closed) < self.max_steps:
            self.closed.append((cls, step[0], total, step[2], step[4], step[5], sends, rpcs))
        self.step = None

    def begin(self):
        now = self.clock_ns()
        self._close(now)
        self.step = self._open(now)

    def bypass(self, cls):
        if self.step is not None and self.step[1] is None:
            self.step[1] = cls

    def advanced(self, started, bypassed):
        step = self.step
        if step is None:
            return
        now = self.clock_ns()
        step[2] += max(0, now - max(started, step[0]))
        step[3] = now
        if bypassed:
            self.bypass('advance')

    def scheduled(self, scheduler_output):
        step = self.step
        if step is None:
            return
        now = self.clock_ns()
        if step[3] is not None:
            step[4] += max(0, now - step[3])
        tokens = getattr(scheduler_output, 'total_num_scheduled_tokens', None)
        self.bypass('empty' if tokens == 0 else 'stock')

    def waited(self, started, ended):
        step = self.step
        if step is not None:
            step[5] += max(0, ended - max(started, step[0]))
            step[6] += 1

    def lines(self, **identity):
        """Close the open step and return the chunk's engine anatomy line(s)."""
        self._close(self.clock_ns())
        detail = (self.chunks - 1) % self.detail_every == 0
        for cls in self.CLASSES:
            entry = self.sums.get(cls)
            if entry is None:
                continue
            totals = sorted(entry[9])
            median = totals[len(totals) // 2] if totals else None
            self.trace.record('e', cls, entry[0], _us(entry[1]), _us(entry[2]), _us(entry[3]),
                              _us(entry[4]), _us(entry[5]), entry[6], entry[7], entry[8],
                              None if median is None else _us(median),
                              _us(totals[-1]) if totals else None)
        for cls, began, total, advance, schedule, wait, sends, rpcs in self.closed:
            self.trace.record('p', cls, began, _us(total), _us(advance), _us(schedule), _us(wait),
                              sends, rpcs)
        self.closed = []
        self.sums = {}
        return self.trace.lines(detail=int(detail), **identity)


JOURNAL_VECTOR_SCHEMA = 's5-journal-vector/1'
JOURNAL_VECTOR_WIDTH = 64
MAX_JOURNAL_VECTOR_ITEMS = JOURNAL_VECTOR_WIDTH ** 3


def encode_report_vector(values):
    """Lossless paging of a validated worker vector; each list stays <=64.

    Small reports keep their original shape. No element, ordering, identity,
    completion sequence or timestamp is removed. The raw ring is unchanged.
    """
    if type(values) not in (list, tuple):
        raise ValueError('report vector must be a sequence')
    if len(values) > MAX_JOURNAL_VECTOR_ITEMS:
        raise ValueError('report vector exceeds the bounded page tree')
    if len(values) <= JOURNAL_VECTOR_WIDTH:
        return list(values)
    items = list(values)
    levels = 0
    while len(items) > JOURNAL_VECTOR_WIDTH:
        items = [items[i:i + JOURNAL_VECTOR_WIDTH]
                 for i in range(0, len(items), JOURNAL_VECTOR_WIDTH)]
        levels += 1
    return {'schema': JOURNAL_VECTOR_SCHEMA, 'length': len(values),
            'levels': levels, 'items': items}


def decode_report_vector(value):
    """Validate canonical pages before returning the exact original vector."""
    if type(value) is list:
        return value
    if (type(value) is not dict or value.get('schema') != JOURNAL_VECTOR_SCHEMA
            or set(value) != {'schema', 'length', 'levels', 'items'}
            or type(value['length']) is not int
            or not JOURNAL_VECTOR_WIDTH < value['length'] <= MAX_JOURNAL_VECTOR_ITEMS
            or type(value['levels']) is not int or not 1 <= value['levels'] <= 2):
        raise ValueError('invalid paged report vector')
    flat = []
    def visit(items, levels):
        if type(items) is not list or not 1 <= len(items) <= JOURNAL_VECTOR_WIDTH:
            raise ValueError('invalid report vector page')
        if levels:
            for child in items:
                visit(child, levels - 1)
        else:
            flat.extend(items)
            if len(flat) > value['length']:
                raise ValueError('report vector length overflow')
    visit(value['items'], value['levels'])
    if len(flat) != value['length'] or encode_report_vector(flat) != value:
        raise ValueError('noncanonical report vector')
    return flat


def _report_vector(value):
    # Preserve existing refusals for malformed legacy fields.
    if type(value) is dict and value.get('schema') == JOURNAL_VECTOR_SCHEMA:
        return decode_report_vector(value)
    return value

SCHEMA = 1

#: Named completeness basis.  Deliberately separate from the raw ring's
#: ``journal.complete_in_window`` flag, which stays exactly as it was.  The
#: v2 basis is a source join over per-stage segment ids and the ranks' actual
#: completed/proof sets; it is not a plan-length count.
MEASUREMENT_BASIS = 'bounded-native-measurement-census-v2-source-join'
RECONCILIATION_BASIS = ('per-chunk open/per-stage segment ids/report completed '
                        'op set/GPU_EVENT/STATE_WRITE role/PP_CONSUMER role/'
                        'commit/source-hold/committed-frontier/PP-receipt/cancel')
CLOCK_BASIS = 'clock-monotonic'
PROOF_TIME_BASIS = 'engine-tick-observation-of-queried-report'

#: Real model/workload shape, from ``deploy/vllm-flash.env`` and
#: ``reference/model-config.json`` (MAX_MODEL_LEN=262144, 48 layers) and the
#: unchanged 1024-token native chunk.
MAX_MODEL_TOKENS = 262144
NATIVE_CHUNK_TOKENS = 1024
MODEL_LAYER_COUNT = 48
MAX_CHUNKS_PER_REQUEST = (MAX_MODEL_TOKENS + NATIVE_CHUNK_TOKENS - 1) // NATIVE_CHUNK_TOKENS
SEGMENTS_PER_CHUNK = 10 + MODEL_LAYER_COUNT

#: Telemetry capacities.  ``max_requests`` covers three lanes x (fresh +
#: append) x four declared repeats with headroom; ``max_chunks_total`` is the
#: resulting hard ceiling.  All are state capacities, not workload parameters.
MAX_REQUESTS = 32
MAX_CHUNKS_TOTAL = MAX_REQUESTS * MAX_CHUNKS_PER_REQUEST          # 8192
MAX_SEGMENTS_PER_CHUNK = 128
MAX_PROOFS_PER_RANK = 128
MAX_PP_RECEIPTS_PER_CHUNK = 8
MAX_SOURCE_HOLDS_PER_CHUNK = 4
MAX_FRONTIERS_PER_CHUNK = 4
MAX_CANCELS_PER_CHUNK = 4
MAX_CONFLICT_REASONS = 24
MAX_ORPHAN_TRANSACTIONS = 128
MAX_STANDALONE_CANCELLATIONS = 128
MAX_FAULTS = 32
MAX_GLOBAL_REASONS = 32
HOST_BYTES_WALK_LIMIT = 4000000

MEASUREMENT_KINDS = (
    'chunk_open', 'segment', 'report', 'chunk_commit', 'pp_receipt',
    'source_hold', 'committed_frontier', 'cancel',
)
_MEASUREMENT_KIND_SET = frozenset(MEASUREMENT_KINDS)

_ACCEPTED_ROUTE = 'split'
_ACCEPTED_TRANSPORT = 'native-split'
_ALLOWED_PROOF_KINDS = frozenset(('gpu_event', 'state_write', 'pp_consumer'))
#: The real descriptor vocabulary produced by ``NativeSegmentDriver._build_plan``
#: (``SegmentKind`` values).  An unknown kind is a source-shape conflict, not a
#: new completion class.
_ALLOWED_SEGMENT_KINDS = frozenset((
    'prepare', 'prefix', 'layer', 'suffix', 'state_finalize', 'pp_send',
    'pp_receive'))
_DESCRIPTIVE_TRANSPORTS = frozenset(('serialized', 'uva', 'fallback'))


def _is_int(value):
    return type(value) is int


def _is_positive_int(value):
    return type(value) is int and value > 0


def _add_reason(bucket, reason, limit=MAX_CONFLICT_REASONS):
    """Append a bounded, de-duplicated reason and never raise."""
    if type(reason) is not str or not reason:
        reason = 'unspecified-conflict'
    if reason not in bucket and len(bucket) < limit:
        bucket.append(reason)


def _reason(bucket, reason, limit=MAX_CONFLICT_REASONS):
    return _add_reason(bucket, reason, limit)


def _transport_is_accepted(value):
    return value == _ACCEPTED_TRANSPORT


def _transport_is_descriptive(value):
    return value in _DESCRIPTIVE_TRANSPORTS


def _deep_size(value, budget):
    """Deterministic sys.getsizeof estimate with a hard node budget."""
    total = sys.getsizeof(value)
    if budget[0] <= 0:
        return total
    budget[0] -= 1
    if type(value) is dict:
        for key, item in value.items():
            total += _deep_size(key, budget)
            total += _deep_size(item, budget)
    elif type(value) in (list, tuple, set, frozenset):
        for item in value:
            total += _deep_size(item, budget)
    return total


class NativeMeasurementCensus:
    """Independent bounded accounting over validated native driver entries.

    The object is constructed once on the engine driver and lives across chunk
    resets and requests.  ``ingest`` is called from ``_record`` after the raw
    ring append; ``evidence`` is queried from ``overlap_evidence`` for one
    declared host-monotonic window.

    Configuration problems never raise: the object records them and every
    ``evidence`` result is then explicitly INCONCLUSIVE, with the configured
    owner (the driver) untouched.  Ingest/evidence also swallow internal faults
    into the same retained evidence rather than failing the engine step.
    """

    def __init__(self, *, engine_epoch=None, ranks=(), pp_senders=None,
                 stage_ranks=None, expected_segments_per_chunk=None,
                 max_requests=MAX_REQUESTS,
                 max_chunks_per_request=MAX_CHUNKS_PER_REQUEST,
                 max_chunks_total=None,
                 max_segments_per_chunk=MAX_SEGMENTS_PER_CHUNK,
                 max_proofs_per_rank=MAX_PROOFS_PER_RANK,
                 max_pp_receipts_per_chunk=MAX_PP_RECEIPTS_PER_CHUNK,
                 max_source_holds_per_chunk=MAX_SOURCE_HOLDS_PER_CHUNK,
                 max_frontiers_per_chunk=MAX_FRONTIERS_PER_CHUNK,
                 max_cancels_per_chunk=MAX_CANCELS_PER_CHUNK,
                 clock_basis=CLOCK_BASIS, pool_chunk_capacity=False):
        self._config_reasons = []
        self.engine_epoch = self._config_int(
            'engine-epoch', engine_epoch, minimum=0)
        self.ranks = self._config_ranks(ranks)
        self.stage_ranks = self._config_stages(stage_ranks, self.ranks)
        #: rank -> PP stage, from the declared stage partition.  The source join
        #: cannot be evaluated without it, so an absent partition is a config
        #: refusal (never an implicit single-stage assumption).
        self.rank_stage = self._config_rank_stages()
        if pp_senders is None:
            pp_senders = (self.stage_ranks.get(0, ()) if self.stage_ranks else ())
        self.pp_senders = self._config_ranks(pp_senders, allow_empty=True)
        if self.stage_ranks and self.pp_senders:
            for sender in self.pp_senders:
                if sender not in self.ranks:
                    _add_reason(self._config_reasons,
                                'config-pp-sender-outside-ranks')
        # Descriptive only: the unexpanded plan length is never a completeness
        # criterion.  Paused QSA continuations legitimately add stage-local op
        # sequences for the same descriptor.
        if expected_segments_per_chunk is not None:
            self.expected_segments_per_chunk = self._config_int(
                'expected-segments', expected_segments_per_chunk, minimum=1)
        else:
            self.expected_segments_per_chunk = None
        self.max_requests = self._config_int(
            'max-requests', max_requests, minimum=1, fallback=MAX_REQUESTS)
        self.max_chunks_per_request = self._config_int(
            'max-chunks-per-request', max_chunks_per_request, minimum=1,
            fallback=MAX_CHUNKS_PER_REQUEST)
        if max_chunks_total is None:
            max_chunks_total = self.max_requests * self.max_chunks_per_request
        self.max_chunks_total = self._config_int(
            'max-chunks-total', max_chunks_total, minimum=1,
            fallback=MAX_CHUNKS_TOTAL)
        self.max_segments_per_chunk = self._config_int(
            'max-segments', max_segments_per_chunk, minimum=1,
            fallback=MAX_SEGMENTS_PER_CHUNK)
        self.max_proofs_per_rank = self._config_int(
            'max-proofs', max_proofs_per_rank, minimum=1,
            fallback=MAX_PROOFS_PER_RANK)
        self.max_pp_receipts_per_chunk = self._config_int(
            'max-pp-receipts', max_pp_receipts_per_chunk, minimum=1,
            fallback=MAX_PP_RECEIPTS_PER_CHUNK)
        self.max_source_holds_per_chunk = self._config_int(
            'max-source-holds', max_source_holds_per_chunk, minimum=1,
            fallback=MAX_SOURCE_HOLDS_PER_CHUNK)
        self.max_frontiers_per_chunk = self._config_int(
            'max-frontiers', max_frontiers_per_chunk, minimum=1,
            fallback=MAX_FRONTIERS_PER_CHUNK)
        self.max_cancels_per_chunk = self._config_int(
            'max-cancels', max_cancels_per_chunk, minimum=1,
            fallback=MAX_CANCELS_PER_CHUNK)
        self.clock_basis = clock_basis if type(clock_basis) is str and clock_basis \
            else CLOCK_BASIS
        if self.clock_basis != CLOCK_BASIS:
            _add_reason(self._config_reasons, 'config-clock-basis-not-monotonic')

        self.pool_chunk_capacity = pool_chunk_capacity is True
        if type(pool_chunk_capacity) is not bool:
            _add_reason(self._config_reasons, 'config-invalid:pool-chunk-capacity')
        # Same aggregate capacities as the prior per-chunk allocation. Expanded
        # chunks borrow unused slots; no facts are dropped and no total grows.
        self._segment_total = 0
        self._proof_totals = {rank: 0 for rank in self.ranks}
        self._completed_totals = {rank: 0 for rank in self.ranks}
        self._chunks = {}
        self._groups = {}
        self._group_order = []
        self._orphans = {}
        self._standalone_cancels = []
        self._ingested = 0
        self._measurement_events = 0
        self._ignored_events = 0
        self._by_kind = {}
        self._overflow_reasons = []
        self._overflow_count = 0
        self._faults = []
        self._fault_count = 0
        self._global_reasons = []
        self._duplicate_observations = 0
        self._first_seen_proofs = 0

    # -- configuration -----------------------------------------------------
    def _config_int(self, label, value, *, minimum, fallback=None):
        if _is_int(value) and value >= minimum:
            return value
        _add_reason(self._config_reasons, 'config-invalid:' + label)
        return fallback if fallback is not None else minimum

    def _config_ranks(self, values, *, allow_empty=False):
        try:
            ranks = tuple(values)
        except Exception:
            _add_reason(self._config_reasons, 'config-ranks-not-iterable')
            return ()
        if not ranks and not allow_empty:
            _add_reason(self._config_reasons, 'config-ranks-empty')
            return ()
        if any(not _is_int(rank) or rank < 0 for rank in ranks):
            _add_reason(self._config_reasons, 'config-rank-not-a-nonnegative-int')
        if len(set(ranks)) != len(ranks):
            _add_reason(self._config_reasons, 'config-rank-duplicate')
        return ranks

    def _config_stages(self, value, ranks):
        if value is None:
            return {}
        if type(value) is not dict:
            _add_reason(self._config_reasons, 'config-stage-ranks-not-a-map')
            return {}
        out = {}
        for stage, members in value.items():
            if not _is_int(stage) or stage < 0:
                _add_reason(self._config_reasons, 'config-stage-invalid')
                continue
            try:
                members = tuple(members)
            except Exception:
                _add_reason(self._config_reasons, 'config-stage-members-not-iterable')
                continue
            if any(member not in ranks for member in members):
                _add_reason(self._config_reasons, 'config-stage-member-outside-ranks')
            out[stage] = members
        return out

    def _config_rank_stages(self):
        """Build rank -> stage from the declared partition.

        A rank in exactly one stage is required; the partition must also cover
        every configured rank.  Both are conditions of the source join, so a
        violation is a config refusal rather than a guessed mapping.
        """
        if not self.stage_ranks:
            _add_reason(self._config_reasons, 'config-stage-partition-missing')
            return {}
        mapping = {}
        for stage, members in self.stage_ranks.items():
            for rank in members:
                if rank in mapping:
                    _add_reason(self._config_reasons, 'config-rank-in-two-stages')
                    continue
                mapping[rank] = stage
        if self.ranks and set(mapping) != set(self.ranks):
            _add_reason(self._config_reasons,
                        'config-stage-partition-does-not-cover-ranks')
        return mapping

    @property
    def config_ok(self):
        return not self._config_reasons

    # -- bounded recording helpers ----------------------------------------
    def _conflict(self, bucket, reason):
        _add_reason(bucket, reason)

    def _overflow(self, reason, chunk=None):
        self._overflow_count += 1
        _add_reason(self._overflow_reasons, reason, limit=MAX_GLOBAL_REASONS)
        if chunk is not None:
            _add_reason(chunk['conflicts'], 'capacity-overflow:' + reason)

    def _fault(self, reason):
        self._fault_count += 1
        _add_reason(self._faults, reason, limit=MAX_FAULTS)

    def _global_conflict(self, reason):
        _add_reason(self._global_reasons, reason, limit=MAX_GLOBAL_REASONS)

    # -- ingest ------------------------------------------------------------
    def ingest(self, entry):
        """Ingest one validated ``_record`` entry.  Never raises.

        Non-measurement journal kinds are counted and ignored.  Any internal
        fault is retained and makes every later evidence result INCONCLUSIVE.
        """
        self._ingested += 1
        try:
            if type(entry) is not dict:
                self._fault('entry-not-an-object')
                return {'accepted': False, 'reason': 'entry-not-an-object'}
            kind = entry.get('kind')
            key = kind if type(kind) is str else '<missing>'
            self._by_kind[key] = self._by_kind.get(key, 0) + 1
            if kind not in _MEASUREMENT_KIND_SET:
                self._ignored_events += 1
                return {'accepted': True, 'ignored': True}
            self._measurement_events += 1
            return self._ingest_measurement(entry)
        except Exception as exc:       # never fail the engine step
            self._fault('ingest-fault:' + type(exc).__name__)
            return {'accepted': False, 'reason': 'ingest-fault'}

    def _ingest_measurement(self, entry):
        if not self.config_ok:
            # Keep counting; evidence is INCONCLUSIVE.  No event is merged into
            # a half-valid model.
            return {'accepted': False, 'reason': 'measurement-config-invalid'}
        kind = entry['kind']
        observed_ns = entry.get('observed_ns')
        if not _is_positive_int(observed_ns):
            self._fault('entry-without-positive-observed-ns:' + kind)
            return {'accepted': False, 'reason': 'observed-ns-invalid'}
        if entry.get('clock_basis') != self.clock_basis:
            self._global_conflict('clock-basis-differs')
            return {'accepted': False, 'reason': 'clock-basis-differs'}
        if entry.get('engine_epoch') != self.engine_epoch:
            self._global_conflict('engine-epoch-differs')
            return {'accepted': False, 'reason': 'engine-epoch-differs'}
        route = entry.get('route')
        if route not in ('split', 'full'):
            self._global_conflict('route-unproved')
            return {'accepted': False, 'reason': 'route-unproved'}
        route_generation = entry.get('route_generation')
        if not _is_int(route_generation) or route_generation < 0:
            self._global_conflict('route-generation-unproved')
            return {'accepted': False, 'reason': 'route-generation-unproved'}
        txn = entry.get('transaction')
        if not _is_int(txn) or txn < 0:
            self._global_conflict('transaction-unproved:' + kind)
            return {'accepted': False, 'reason': 'transaction-unproved'}
        if kind == 'chunk_open':
            return self._ingest_chunk_open(entry, txn, observed_ns)
        if kind == 'cancel':
            return self._ingest_cancel(entry, txn, observed_ns)
        chunk = self._chunks.get(txn)
        if chunk is None:
            return self._orphan(entry, txn, observed_ns)
        if observed_ns < chunk['open_ns']:
            self._conflict(chunk['conflicts'], 'event-before-chunk-open:' + kind)
        if chunk.get('malformed'):
            return {'accepted': False, 'reason': 'chunk-open-invalid'}
        if entry.get('route') != chunk['route']:
            self._conflict(chunk['conflicts'], 'route-conflict:' + kind)
        if route_generation != chunk['route_generation']:
            self._conflict(chunk['conflicts'], 'route-generation-conflict:' + kind)
        handler = {
            'segment': self._ingest_segment,
            'report': self._ingest_report,
            'chunk_commit': self._ingest_commit,
            'pp_receipt': self._ingest_pp_receipt,
            'source_hold': self._ingest_source_hold,
            'committed_frontier': self._ingest_frontier,
        }[kind]
        return handler(entry, chunk, observed_ns)

    def _orphan(self, entry, txn, observed_ns):
        record = self._orphans.get(txn)
        if record is None:
            if len(self._orphans) >= MAX_ORPHAN_TRANSACTIONS:
                self._overflow('orphan-capacity')
                return {'accepted': False, 'reason': 'orphan-capacity'}
            record = {'transaction': txn, 'first_ns': observed_ns,
                      'kinds': [], 'reasons': []}
            self._orphans[txn] = record
        if entry['kind'] not in record['kinds']:
            record['kinds'].append(entry['kind'])
        _add_reason(record['reasons'], 'event-before-chunk-open:' + entry['kind'])
        return {'accepted': False,
                'reason': 'event-before-chunk-open:' + entry['kind']}

    # -- chunk_open --------------------------------------------------------
    def _ingest_chunk_open(self, entry, txn, observed_ns):
        request = entry.get('request')
        generation = entry.get('request_generation')
        chunk_start = entry.get('chunk_start')
        chunk_end = entry.get('chunk_end')
        committed_end = entry.get('committed_end')
        prompt_end = entry.get('prompt_end')
        state_epoch = entry.get('state_epoch')
        remaining = entry.get('remaining_tokens')
        terminal = entry.get('terminal')
        fields_ok = (
            type(request) is str and bool(request)
            and _is_int(generation) and generation >= 1
            and _is_int(chunk_start) and chunk_start >= 0
            and _is_int(chunk_end) and chunk_end > chunk_start
            and _is_int(committed_end) and committed_end == chunk_start
            and _is_int(prompt_end) and prompt_end >= 2
            and chunk_end <= prompt_end - 1
            and _is_int(state_epoch) and state_epoch >= 1
            and _is_int(remaining) and remaining >= 1
            and remaining == prompt_end - chunk_end
            and type(terminal) is bool and terminal == (remaining == 1)
        )
        if not fields_ok:
            self._fault('chunk-open-invalid:transaction=%d' % txn)
            return {'accepted': False, 'reason': 'chunk-open-invalid'}
        prior = self._chunks.get(txn)
        if prior is not None:
            if self._same_open(prior, entry):
                self._duplicate_observations += 1
                return {'accepted': True, 'repeated': True}
            self._conflict(prior['conflicts'], 'duplicate-chunk-open-conflict')
            return {'accepted': False, 'reason': 'duplicate-chunk-open-conflict'}
        if len(self._chunks) >= self.max_chunks_total:
            self._overflow('chunk-capacity-total')
            return {'accepted': False, 'reason': 'chunk-capacity-total'}
        group_key = (request, generation)
        group = self._groups.get(group_key)
        if group is None:
            if len(self._groups) >= self.max_requests:
                self._overflow('request-capacity')
                return {'accepted': False, 'reason': 'request-capacity'}
            group = {
                'key': group_key,
                'request': request,
                'generation': generation,
                'initial_frontier': chunk_start,
                'first_ingested_start': chunk_start,
                'family': 'fresh' if chunk_start == 0 else 'append',
                'prompt_end': prompt_end,
                'chunks': [],
                'last_start': None,
                'last_end': None,
                'first_open_ns': observed_ns,
                'last_open_ns': observed_ns,
                'conflicts': [],
                'terminal_txn': None,
                'saw_cancel': False,
            }
            self._groups[group_key] = group
            self._group_order.append(group_key)
        else:
            if group['prompt_end'] != prompt_end:
                self._conflict(group['conflicts'], 'request-prompt-end-conflict')
            if chunk_start <= group['last_start']:
                if chunk_start == group['last_start']:
                    self._conflict(group['conflicts'], 'chunk-start-duplicate')
                else:
                    self._conflict(group['conflicts'], 'chunk-reordered')
            elif chunk_start != group['last_end']:
                if chunk_start > group['last_end']:
                    self._conflict(group['conflicts'], 'chunk-frontier-gap')
                else:
                    self._conflict(group['conflicts'], 'chunk-frontier-overlap')
            if len(group['chunks']) >= self.max_chunks_per_request:
                self._overflow('chunk-capacity-request', {'conflicts': []})
                return {'accepted': False, 'reason': 'chunk-capacity-request'}
            if chunk_start < group['first_ingested_start']:
                group['first_ingested_start'] = chunk_start
            if chunk_start < group['initial_frontier']:
                # The true initial frontier was observed later (reordered
                # ingress).  Classify from the actual minimum; the ordering
                # conflict above already refuses the group.
                group['initial_frontier'] = chunk_start
                group['family'] = 'fresh' if chunk_start == 0 else 'append'
            if observed_ns < group['last_open_ns']:
                self._conflict(group['conflicts'], 'chunk-open-ns-reordered')
            group['last_open_ns'] = observed_ns
        chunk = {
            'transaction': txn,
            'request': request,
            'generation': generation,
            'chunk_start': chunk_start,
            'chunk_end': chunk_end,
            'committed_end': committed_end,
            'prompt_end': prompt_end,
            'state_epoch': state_epoch,
            'remaining_tokens': remaining,
            'terminal': terminal,
            'route': entry.get('route'),
            'route_generation': entry.get('route_generation'),
            'open_ns': observed_ns,
            'malformed': False,
            'saw_orphan': txn in self._orphans,
            'cancelled': False,
            'cancels': [],
            'segments': {
                'attrs': {},
                'unique': 0,
                'events': 0,
                'first_ns': None,
                'last_ns': None,
                'transport': None,
                'serialized': False,
            },
            'ranks': {},
            'pp': {},
            'pp_observations': 0,
            'source_hold': None,
            'source_hold_observations': 0,
            'frontier': None,
            'frontier_observations': 0,
            'frontier_missing': False,
            'commit': None,
            'commit_observations': 0,
            'conflicts': [],
        }
        if chunk['saw_orphan']:
            self._conflict(chunk['conflicts'], 'event-before-chunk-open')
        if group['terminal_txn'] is not None:
            self._conflict(chunk['conflicts'], 'chunk-after-terminal')
        if terminal:
            if group['terminal_txn'] is not None:
                self._conflict(chunk['conflicts'], 'multiple-terminal-chunks')
            group['terminal_txn'] = txn
        group['chunks'].append(txn)
        group['last_start'] = chunk_start
        group['last_end'] = chunk_end
        self._chunks[txn] = chunk
        if txn in self._orphans:
            self._orphans.pop(txn, None)
        return {'accepted': True}

    @staticmethod
    def _same_open(chunk, entry):
        return (
            chunk['request'] == entry.get('request')
            and chunk['generation'] == entry.get('request_generation')
            and chunk['chunk_start'] == entry.get('chunk_start')
            and chunk['chunk_end'] == entry.get('chunk_end')
            and chunk['committed_end'] == entry.get('committed_end')
            and chunk['prompt_end'] == entry.get('prompt_end')
            and chunk['state_epoch'] == entry.get('state_epoch')
            and chunk['remaining_tokens'] == entry.get('remaining_tokens')
            and chunk['terminal'] == entry.get('terminal')
            and chunk['route_generation'] == entry.get('route_generation')
            and chunk['route'] == entry.get('route')
        )

    # -- segment -----------------------------------------------------------
    def _ingest_segment(self, entry, chunk, observed_ns):
        seg = chunk['segments']
        op_seq = entry.get('op_seq')
        stage = entry.get('stage')
        segment_kind = entry.get('segment_kind')
        targets = entry.get('targets')
        transport = entry.get('transport')
        if not _is_int(op_seq) or op_seq < 0:
            self._conflict(chunk['conflicts'], 'segment-op-seq-unproved')
            return {'accepted': False, 'reason': 'segment-op-seq-unproved'}
        if not self.pool_chunk_capacity and op_seq >= self.max_segments_per_chunk:
            self._overflow('segment-op-seq-capacity', chunk)
            return {'accepted': False, 'reason': 'segment-op-seq-capacity'}
        if not _is_int(stage) or not self.stage_ranks or stage not in self.stage_ranks:
            self._conflict(chunk['conflicts'], 'segment-stage-unproved')
            return {'accepted': False, 'reason': 'segment-stage-unproved'}
        if segment_kind not in _ALLOWED_SEGMENT_KINDS:
            # The real vocabulary is the driver's ``SegmentKind`` values; a new
            # spelling is not a new completion class.
            self._conflict(chunk['conflicts'], 'segment-kind-unproved')
            return {'accepted': False, 'reason': 'segment-kind-unproved'}
        layer_start = entry.get('layer_start')
        layer_end = entry.get('layer_end')
        if segment_kind == 'layer':
            if not _is_int(layer_start) or layer_start < 0 \
                    or not _is_int(layer_end) \
                    or not layer_start < layer_end <= layer_start + entry.get('layer_group', 1):
                self._conflict(chunk['conflicts'], 'segment-layer-shape-invalid')
                return {'accepted': False,
                        'reason': 'segment-layer-shape-invalid'}
        elif layer_start is not None or layer_end is not None:
            self._conflict(chunk['conflicts'], 'segment-layer-shape-invalid')
            return {'accepted': False, 'reason': 'segment-layer-shape-invalid'}
        try:
            target_tuple = tuple(targets)
        except Exception:
            self._conflict(chunk['conflicts'], 'segment-targets-unproved')
            return {'accepted': False, 'reason': 'segment-targets-unproved'}
        if not target_tuple or any(target not in self.ranks for target in target_tuple):
            self._conflict(chunk['conflicts'], 'segment-targets-outside-ranks')
            return {'accepted': False, 'reason': 'segment-targets-outside-ranks'}
        expected = self.stage_ranks.get(stage)
        if expected is None:
            self._conflict(chunk['conflicts'], 'segment-stage-not-declared')
            return {'accepted': False, 'reason': 'segment-stage-not-declared'}
        if set(target_tuple) != set(expected):
            self._conflict(chunk['conflicts'],
                           'segment-targets-differ-from-stage-ranks')
            return {'accepted': False,
                    'reason': 'segment-targets-differ-from-stage-ranks'}
        attributes = (segment_kind, layer_start, layer_end,
                      tuple(sorted(target_tuple)), transport)
        # ``OpSeqTracker`` is per worker and therefore per PP stage: the same
        # op_seq on two stages names two different descriptors.
        key = (stage, op_seq)
        prior = seg['attrs'].get(key)
        if prior is not None:
            if prior != attributes:
                self._conflict(chunk['conflicts'], 'segment-op-conflict')
                return {'accepted': False, 'reason': 'segment-op-conflict'}
            seg['events'] += 1
            return {'accepted': True, 'repeated': True}
        if ((self.pool_chunk_capacity and self._segment_total >= self.max_segments_per_chunk * self.max_chunks_total)
                or (not self.pool_chunk_capacity and seg['unique'] >= self.max_segments_per_chunk)):
            self._overflow('segment-capacity', chunk)
            return {'accepted': False, 'reason': 'segment-capacity'}
        seg['attrs'][key] = attributes
        seg['unique'] += 1
        self._segment_total += 1
        seg['events'] += 1
        if seg['first_ns'] is None or observed_ns < seg['first_ns']:
            seg['first_ns'] = observed_ns
        if seg['last_ns'] is None or observed_ns > seg['last_ns']:
            seg['last_ns'] = observed_ns
        if seg['transport'] is None:
            seg['transport'] = transport
        elif seg['transport'] != transport:
            self._conflict(chunk['conflicts'], 'segment-transport-mixed')
        if _transport_is_descriptive(transport):
            seg['serialized'] = True
        elif not _transport_is_accepted(transport):
            self._conflict(chunk['conflicts'], 'segment-transport-unproved')
        if chunk['commit'] is not None and observed_ns > chunk['commit']['observed_ns']:
            self._conflict(chunk['conflicts'], 'segment-after-commit')
        return {'accepted': True}

    # -- report ------------------------------------------------------------
    def _rank_record(self, chunk, rank):
        record = chunk['ranks'].get(rank)
        if record is None:
            record = {
                'reports': 0,
                'first_report_ns': None,
                'last_report_ns': None,
                'first_proof_ns': None,
                'completed_sequence': None,
                'watermark_regressed': False,
                'watermark_mismatch': False,
                #: Union of the cumulative ``completed_op_seqs`` facts this
                #: rank has actually reported.  A report is cumulative, so the
                #: union is the rank's complete op set as observed at the last
                #: accepted report; it is joined against the stage's accepted
                #: segment ids, never assumed.
                'completed_ops': None,
                'completed_set_regressed': False,
                'proofs': {},
                'unique_proofs': 0,
                'proof_repeats': 0,
                'state_write': None,
                'pp_consumer': None,
                'pending_samples': 0,
                'pending_after_completion': False,
                'failed_code': None,
                'state_flag_missing_proof': False,
                'proof_without_state_flag': False,
            }
            chunk['ranks'][rank] = record
        return record

    def _ingest_report(self, entry, chunk, observed_ns):
        rank = entry.get('worker_rank')
        if rank not in self.ranks:
            self._conflict(chunk['conflicts'], 'report-rank-outside-ranks')
            return {'accepted': False, 'reason': 'report-rank-outside-ranks'}
        record = self._rank_record(chunk, rank)
        record['reports'] += 1
        if record['first_report_ns'] is None:
            record['first_report_ns'] = observed_ns
        if record['last_report_ns'] is not None \
                and observed_ns < record['last_report_ns']:
            self._conflict(chunk['conflicts'], 'report-observation-regressed')
        record['last_report_ns'] = observed_ns
        if chunk['commit'] is not None \
                and observed_ns > chunk['commit']['observed_ns']:
            self._conflict(chunk['conflicts'], 'report-after-commit')
        completed_sequence = entry.get('completed_sequence')
        if _is_int(completed_sequence) and completed_sequence >= 0:
            prior = record['completed_sequence']
            if prior is not None and completed_sequence < prior:
                record['watermark_regressed'] = True
                self._conflict(chunk['conflicts'], 'report-watermark-regressed')
            else:
                record['completed_sequence'] = completed_sequence
        else:
            self._conflict(chunk['conflicts'], 'report-watermark-unproved')
        failed_code = entry.get('failed_code')
        if failed_code is not None:
            if type(failed_code) is not str:
                failed_code = 'unproved'
            record['failed_code'] = failed_code
            self._conflict(chunk['conflicts'], 'report-failed:' + failed_code)
        # The cumulative completed op set is the actual source fact the stage
        # join consumes.  It must be an int sequence, cumulative, and disjoint
        # from the rank's outstanding work.
        try:
            completed = _report_vector(entry.get('completed_op_seqs'))
            pending = _report_vector(entry.get('pending_op_seqs')) or []
            proofs = _report_vector(entry.get('proofs')) or []
        except ValueError:
            self._conflict(chunk['conflicts'], 'report-vector-invalid')
            return {'accepted': False, 'reason': 'report-vector-invalid'}
        # Held-g L4: a lean entry's completed/proof vectors are the rank's new
        # items only; its header counts what the rank reported before.
        lean = entry.get('lean')
        if lean is not None and not (
                type(lean) is dict
                and all(_is_int(lean.get(key)) and lean[key] >= 0
                        for key in ('base_completed', 'base_proofs', 'completed', 'proofs'))):
            self._conflict(chunk['conflicts'], 'report-lean-header-invalid')
            return {'accepted': False, 'reason': 'report-lean-header-invalid'}
        if type(completed) is not list:
            self._conflict(chunk['conflicts'], 'report-completed-not-a-list')
            completed_set = None
        else:
            completed_set = set()
            completed_ok = True
            for op in completed:
                if not _is_int(op) or op < 0:
                    completed_ok = False
                    break
                completed_set.add(op)
            if not completed_ok or len(completed_set) != len(completed):
                self._conflict(chunk['conflicts'],
                               'report-completed-set-unproved')
                completed_set = None
            elif lean is not None:
                prior = record['completed_ops'] or frozenset()
                if lean['base_completed'] != len(prior) or lean['base_proofs'] != len(record['proofs']):
                    self._conflict(chunk['conflicts'], 'report-lean-gap')
                if not prior.isdisjoint(completed_set):
                    self._conflict(chunk['conflicts'], 'report-lean-replay')
                # The rank's cumulative set: what every check below compares.
                completed_set = prior.union(completed_set)
                if lean['completed'] != len(completed_set):
                    self._conflict(chunk['conflicts'], 'report-lean-count-differs')
        pending_set = set()
        if type(pending) is not list:
            self._conflict(chunk['conflicts'], 'report-pending-not-a-list')
            pending = []
        else:
            for op in pending:
                if not _is_int(op) or op < 0:
                    self._conflict(chunk['conflicts'],
                                   'report-pending-op-seq-unproved')
                    pending_set = None
                    break
                pending_set.add(op)
        if pending_set is not None:
            if completed_set is not None and pending_set & completed_set:
                record['pending_after_completion'] = True
                self._conflict(chunk['conflicts'],
                               'report-pending-after-completion')
            if record['completed_ops'] is not None \
                    and pending_set & record['completed_ops']:
                record['pending_after_completion'] = True
                self._conflict(chunk['conflicts'],
                               'report-pending-after-completion')
        if pending:
            record['pending_samples'] += 1
            if record['state_write'] is not None:
                record['pending_after_completion'] = True
                self._conflict(chunk['conflicts'],
                               'report-pending-after-completion')
        if completed_set is not None:
            prior_count = len(record['completed_ops'] or ())
            exceeds = (self._completed_totals[rank] + max(0, len(completed_set) - prior_count)
                       > self.max_segments_per_chunk * self.max_chunks_total
                       if self.pool_chunk_capacity else len(completed_set) > self.max_segments_per_chunk)
            if exceeds:
                self._overflow('report-completed-capacity', chunk)
                completed_set = None
        if completed_set is not None:
            prior = record['completed_ops']
            if prior is not None and not prior.issubset(completed_set):
                record['completed_set_regressed'] = True
                self._conflict(chunk['conflicts'],
                               'report-completed-set-regressed')
                # Keep the earlier superset: a later cumulative report may not
                # retract an operation it already reported as complete.
            else:
                self._completed_totals[rank] += len(completed_set) - len(prior or ())
                record['completed_ops'] = frozenset(completed_set)
        state_written = entry.get('state_written') is True
        if type(proofs) is not list:
            self._conflict(chunk['conflicts'], 'report-proofs-not-a-list')
            proofs = []
        saw_state_proof = False
        for proof in proofs:
            if type(proof) is not dict:
                self._conflict(chunk['conflicts'], 'report-proof-not-an-object')
                continue
            kind = proof.get('kind')
            op_seq = proof.get('op_seq')
            completion_seq = proof.get('completion_seq')
            if kind not in _ALLOWED_PROOF_KINDS:
                self._conflict(chunk['conflicts'], 'report-proof-kind-unproved')
                continue
            if type(op_seq) is not int or op_seq < 0:
                self._conflict(chunk['conflicts'], 'report-proof-op-seq-unproved')
                continue
            if type(completion_seq) is not int or completion_seq < 0:
                self._conflict(chunk['conflicts'],
                               'report-proof-completion-seq-unproved')
                continue
            if kind == 'state_write':
                saw_state_proof = True
            key = (kind, op_seq)
            prior = record['proofs'].get(key)
            if prior is not None:
                if prior['completion_seq'] != completion_seq:
                    self._conflict(chunk['conflicts'], 'report-proof-conflict')
                else:
                    record['proof_repeats'] += 1
                    self._duplicate_observations += 1
                continue
            if ((self.pool_chunk_capacity and self._proof_totals[rank] >= self.max_proofs_per_rank * self.max_chunks_total)
                    or (not self.pool_chunk_capacity and record['unique_proofs'] >= self.max_proofs_per_rank)):
                self._overflow('proof-capacity', chunk)
                continue
            retained = {'completion_seq': completion_seq,
                        'observed_ns': observed_ns}
            record['proofs'][key] = retained
            record['unique_proofs'] += 1
            self._proof_totals[rank] += 1
            self._first_seen_proofs += 1
            # First queried observation, never a later cumulative duplicate.
            if record['first_proof_ns'] is None \
                    or observed_ns < record['first_proof_ns']:
                record['first_proof_ns'] = observed_ns
            if kind == 'state_write':
                if record['state_write'] is None:
                    record['state_write'] = {
                        'op_seq': op_seq,
                        'completion_seq': completion_seq,
                        'observed_ns': observed_ns,
                    }
                else:
                    self._conflict(chunk['conflicts'],
                                   'multiple-state-write-proofs')
            elif kind == 'pp_consumer':
                if record['pp_consumer'] is None:
                    record['pp_consumer'] = {
                        'op_seq': op_seq,
                        'completion_seq': completion_seq,
                        'observed_ns': observed_ns,
                    }
        if lean is not None:
            # A lean entry carries the STATE_WRITE proof once, in its delta.
            saw_state_proof = record['state_write'] is not None
            if lean['proofs'] != len(record['proofs']):
                self._conflict(chunk['conflicts'], 'report-lean-count-differs')
        if state_written and not saw_state_proof:
            # Cumulative reports repeat their proofs; a true state_written
            # report always carries the STATE_WRITE proof in the same list.
            record['state_flag_missing_proof'] = True
            self._conflict(chunk['conflicts'], 'state-written-without-proof')
        if saw_state_proof and not state_written:
            record['proof_without_state_flag'] = True
            self._conflict(chunk['conflicts'], 'state-proof-without-flag')
        if record['state_write'] is not None \
                and record['state_write']['observed_ns'] > observed_ns:
            self._conflict(chunk['conflicts'], 'state-write-observation-regressed')
        # The watermark is the worker's own ``_completion_seq`` at this poll:
        # it must equal the highest queried proof sequence in the same report.
        max_proof_seq = max((value['completion_seq']
                             for value in record['proofs'].values()), default=0)
        record['watermark_mismatch'] = False
        if record['completed_sequence'] is not None \
                and record['completed_sequence'] != max_proof_seq:
            record['watermark_mismatch'] = True
            self._conflict(chunk['conflicts'],
                           'report-watermark-differs-from-proof')
        return {'accepted': True}

    # -- PP receipt --------------------------------------------------------
    def _ingest_pp_receipt(self, entry, chunk, observed_ns):
        chunk['pp_observations'] += 1
        sender = entry.get('sender_rank')
        seq = entry.get('seq')
        completion_seq = entry.get('completion_seq')
        accepted = entry.get('accepted') is True
        if not accepted:
            self._conflict(chunk['conflicts'], 'pp-receipt-not-accepted')
            return {'accepted': False, 'reason': 'pp-receipt-not-accepted'}
        if not _is_int(sender) or sender not in self.ranks:
            self._conflict(chunk['conflicts'], 'pp-receipt-sender-unproved')
            return {'accepted': False, 'reason': 'pp-receipt-sender-unproved'}
        if self.pp_senders and sender not in self.pp_senders:
            self._conflict(chunk['conflicts'], 'pp-receipt-unexpected-sender')
            return {'accepted': False, 'reason': 'pp-receipt-unexpected-sender'}
        if not _is_int(seq) or seq < 0 or not _is_int(completion_seq) \
                or completion_seq < 0:
            self._conflict(chunk['conflicts'], 'pp-receipt-identity-unproved')
            return {'accepted': False, 'reason': 'pp-receipt-identity-unproved'}
        if chunk['commit'] is not None \
                and observed_ns > chunk['commit']['observed_ns']:
            self._conflict(chunk['conflicts'], 'pp-receipt-after-commit')
        prior = chunk['pp'].get(sender)
        if prior is None:
            if len(chunk['pp']) >= self.max_pp_receipts_per_chunk:
                self._overflow('pp-receipt-capacity', chunk)
                return {'accepted': False, 'reason': 'pp-receipt-capacity'}
            chunk['pp'][sender] = {'seq': seq, 'completion_seq': completion_seq,
                                   'observed_ns': observed_ns}
            return {'accepted': True}
        if prior['seq'] != seq or prior['completion_seq'] != completion_seq:
            self._conflict(chunk['conflicts'], 'pp-receipt-conflict')
            return {'accepted': False, 'reason': 'pp-receipt-conflict'}
        self._duplicate_observations += 1
        return {'accepted': True, 'repeated': True}

    # -- source hold -------------------------------------------------------
    def _check_source_description(self, chunk, description):
        """Check the driver's pre-commit ``hold.describe()`` identity.

        ``describe`` is captured **before** ``controller.commit``: ``released``
        is therefore expected to be false and is retained as observed, never
        required.  Retirement is carried by the entry's own ``retired`` flag.
        """
        if description.get('transaction_seq') != chunk['transaction']:
            self._conflict(chunk['conflicts'],
                           'source-hold-transaction-conflict')
        if description.get('state_epoch') != chunk['state_epoch']:
            self._conflict(chunk['conflicts'],
                           'source-hold-state-epoch-conflict')
        if description.get('request_uid') != chunk['request']:
            self._conflict(chunk['conflicts'], 'source-hold-request-conflict')
        if description.get('bound') is not True:
            self._conflict(chunk['conflicts'], 'source-hold-not-bound')
        if description.get('ranks') != len(self.ranks):
            self._conflict(chunk['conflicts'],
                           'source-hold-rank-count-differs')
        if description.get('owners') != len(self.ranks):
            self._conflict(chunk['conflicts'],
                           'source-hold-owner-count-differs')

    def _ingest_source_hold(self, entry, chunk, observed_ns):
        chunk['source_hold_observations'] += 1
        retired = entry.get('retired') is True
        description = entry.get('description')
        if description is not None and type(description) is not dict:
            self._conflict(chunk['conflicts'],
                           'source-hold-description-unproved')
            description = None
        if type(description) is dict:
            self._check_source_description(chunk, description)
        record = chunk['source_hold']
        if record is None:
            if chunk['source_hold_observations'] > self.max_source_holds_per_chunk:
                self._overflow('source-hold-capacity', chunk)
                return {'accepted': False, 'reason': 'source-hold-capacity'}
            record = {
                'retired': retired,
                'retired_ns': observed_ns if retired else None,
                'description': description,
                'first_ns': observed_ns,
            }
            chunk['source_hold'] = record
            return {'accepted': True}
        if record['retired'] and not retired:
            self._conflict(chunk['conflicts'], 'source-hold-regressed')
        if retired and not record['retired']:
            record['retired'] = True
            record['retired_ns'] = observed_ns
        if description is not None and record['description'] is None:
            record['description'] = description
        elif description is not None and record['description'] is not None \
                and description != record['description']:
            self._conflict(chunk['conflicts'],
                           'source-hold-description-conflict')
        self._duplicate_observations += 1
        return {'accepted': True, 'repeated': True}

    # -- committed frontier ------------------------------------------------
    def _ingest_frontier(self, entry, chunk, observed_ns):
        chunk['frontier_observations'] += 1
        frontier = entry.get('frontier')
        if frontier is None:
            chunk['frontier_missing'] = True
            return {'accepted': False, 'reason': 'committed-frontier-missing'}
        if type(frontier) is not dict:
            self._conflict(chunk['conflicts'], 'committed-frontier-not-an-object')
            return {'accepted': False, 'reason': 'committed-frontier-malformed'}
        retained = {
            'committed_end': frontier.get('committed_end'),
            'remaining': frontier.get('remaining'),
            'state_epoch': frontier.get('state_epoch'),
            'ranks': list(frontier.get('ranks') or []),
            'pp_seqs': list(frontier.get('pp_seqs') or []),
            'observed_ns': observed_ns,
        }
        prior = chunk['frontier']
        if prior is None:
            if chunk['frontier_observations'] > self.max_frontiers_per_chunk:
                self._overflow('frontier-capacity', chunk)
                return {'accepted': False, 'reason': 'frontier-capacity'}
            chunk['frontier'] = retained
            return {'accepted': True}
        for key in ('committed_end', 'remaining', 'state_epoch', 'ranks',
                    'pp_seqs'):
            if prior[key] != retained[key]:
                self._conflict(chunk['conflicts'], 'committed-frontier-conflict')
                return {'accepted': False, 'reason': 'committed-frontier-conflict'}
        self._duplicate_observations += 1
        return {'accepted': True, 'repeated': True}

    # -- chunk commit ------------------------------------------------------
    def _ingest_commit(self, entry, chunk, observed_ns):
        chunk['commit_observations'] += 1
        if chunk['commit'] is not None:
            prior = chunk['commit']
            if (prior['request'] == entry.get('request')
                    and prior['chunk_start'] == entry.get('chunk_start')
                    and prior['chunk_end'] == entry.get('chunk_end')
                    and prior['committed_end'] == entry.get('committed_end')
                    and prior['state_epoch'] == entry.get('state_epoch')
                    and prior['terminal'] == entry.get('terminal')):
                self._duplicate_observations += 1
                return {'accepted': True, 'repeated': True}
            self._conflict(chunk['conflicts'], 'duplicate-commit-conflict')
            return {'accepted': False, 'reason': 'duplicate-commit-conflict'}
        if entry.get('request') != chunk['request']:
            self._conflict(chunk['conflicts'], 'commit-request-conflict')
        if entry.get('chunk_start') != chunk['chunk_start'] \
                or entry.get('chunk_end') != chunk['chunk_end']:
            self._conflict(chunk['conflicts'], 'commit-frontier-conflict')
        if entry.get('committed_end') != chunk['committed_end']:
            self._conflict(chunk['conflicts'], 'commit-committed-end-conflict')
        if entry.get('state_epoch') != chunk['state_epoch']:
            self._conflict(chunk['conflicts'], 'commit-state-epoch-conflict')
        if entry.get('terminal') != chunk['terminal']:
            self._conflict(chunk['conflicts'], 'commit-terminal-conflict')
        releases = entry.get('worker_releases')
        if type(releases) is not list:
            self._conflict(chunk['conflicts'], 'commit-releases-not-a-list')
            releases = []
        by_rank = {}
        for row in releases:
            if type(row) is not dict:
                self._conflict(chunk['conflicts'], 'commit-release-not-an-object')
                continue
            rank = row.get('worker_rank')
            if rank not in self.ranks:
                self._conflict(chunk['conflicts'], 'commit-release-rank-unproved')
                continue
            if rank in by_rank:
                self._conflict(chunk['conflicts'],
                               'commit-release-rank-duplicate')
                continue
            by_rank[rank] = {'released': row.get('released') is True,
                             'closed': row.get('closed') is True,
                             'completed_sequence': row.get('completed_sequence')}
        if set(by_rank) != set(self.ranks):
            self._conflict(chunk['conflicts'], 'commit-release-ranks-incomplete')
        for rank, row in by_rank.items():
            if not row['released']:
                self._conflict(chunk['conflicts'], 'worker-not-released:' + str(rank))
            if chunk['terminal'] and not row['closed']:
                self._conflict(chunk['conflicts'],
                               'terminal-worker-not-closed:' + str(rank))
            if not chunk['terminal'] and row['closed']:
                self._conflict(chunk['conflicts'],
                               'nonterminal-worker-closed:' + str(rank))
            rank_record = chunk['ranks'].get(rank)
            if rank_record is not None \
                    and rank_record['completed_sequence'] is not None \
                    and row['completed_sequence'] != rank_record['completed_sequence']:
                self._conflict(chunk['conflicts'],
                               'release-watermark-differs-from-report:' + str(rank))
        chunk['commit'] = {
            'observed_ns': observed_ns,
            'request': entry.get('request'),
            'chunk_start': entry.get('chunk_start'),
            'chunk_end': entry.get('chunk_end'),
            'committed_end': entry.get('committed_end'),
            'state_epoch': entry.get('state_epoch'),
            'terminal': entry.get('terminal'),
            'releases': by_rank,
        }
        if observed_ns < chunk['open_ns']:
            self._conflict(chunk['conflicts'], 'commit-before-chunk-open')
        return {'accepted': True}

    # -- cancel ------------------------------------------------------------
    def _ingest_cancel(self, entry, txn, observed_ns):
        never_submitted = entry.get('never_submitted') is True
        closed_ranks = entry.get('closed_ranks')
        try:
            closed_ranks = sorted(closed_ranks or ())
        except Exception:
            closed_ranks = []
        reason = entry.get('reason')
        row = {
            'transaction': txn,
            'never_submitted': never_submitted,
            'closed_ranks': closed_ranks,
            'reason': reason if type(reason) is str else None,
            'observed_ns': observed_ns,
        }
        chunk = self._chunks.get(txn)
        if chunk is None:
            if not never_submitted:
                self._global_conflict('cancel-without-open')
            if len(self._standalone_cancels) >= MAX_STANDALONE_CANCELLATIONS:
                self._overflow('standalone-cancel-capacity')
                return {'accepted': False, 'reason': 'standalone-cancel-capacity'}
            self._standalone_cancels.append(row)
            return {'accepted': True}
        if len(chunk['cancels']) >= self.max_cancels_per_chunk:
            self._overflow('cancel-capacity', chunk)
            return {'accepted': False, 'reason': 'cancel-capacity'}
        if chunk['cancels']:
            prior = chunk['cancels'][-1]
            if (prior['never_submitted'] == never_submitted
                    and prior['closed_ranks'] == closed_ranks
                    and prior['reason'] == row['reason']):
                self._duplicate_observations += 1
                return {'accepted': True, 'repeated': True}
            self._conflict(chunk['conflicts'], 'cancel-conflict')
        chunk['cancels'].append(row)
        chunk['cancelled'] = True
        if chunk['commit'] is not None:
            self._conflict(chunk['conflicts'], 'cancel-after-commit')
        return {'accepted': True}

    # -- retention statistics ---------------------------------------------
    def stats(self):
        """Bounded capacity, retained counts and host-byte estimate."""
        host_bytes = self.host_bytes_estimate()
        retained_chunks = len(self._chunks)
        per_chunk = None
        if host_bytes.get('bytes') is not None and retained_chunks:
            per_chunk = host_bytes['bytes'] / retained_chunks
        projected = None
        if per_chunk is not None:
            projected = int(round(per_chunk * self.max_chunks_total))
        return {
            'basis': MEASUREMENT_BASIS,
            'capacity': {
                'max_requests': self.max_requests,
                'max_chunks_per_request': self.max_chunks_per_request,
                'max_chunks_total': self.max_chunks_total,
                'max_segments_per_chunk': None if self.pool_chunk_capacity else self.max_segments_per_chunk,
                'max_proofs_per_rank': None if self.pool_chunk_capacity else self.max_proofs_per_rank,
                'pool_chunk_capacity': self.pool_chunk_capacity,
                'segments_total_limit': self.max_segments_per_chunk * self.max_chunks_total,
                'proofs_per_rank_total_limit': self.max_proofs_per_rank * self.max_chunks_total,
                'completed_per_rank_total_limit': self.max_segments_per_chunk * self.max_chunks_total,
                'retained_segments_total': self._segment_total,
                'retained_proofs_per_rank': dict(self._proof_totals),
                'retained_completed_per_rank': dict(self._completed_totals),
                'max_pp_receipts_per_chunk': self.max_pp_receipts_per_chunk,
                'max_source_holds_per_chunk': self.max_source_holds_per_chunk,
                'max_frontiers_per_chunk': self.max_frontiers_per_chunk,
                'max_cancels_per_chunk': self.max_cancels_per_chunk,
                'max_orphan_transactions': MAX_ORPHAN_TRANSACTIONS,
                'max_standalone_cancellations': MAX_STANDALONE_CANCELLATIONS,
                'segment_identity': '(stage, op_seq)',
                'plan_segments_per_chunk': self.expected_segments_per_chunk,
                'plan_segments_basis': ('driver _build_plan = 10 + 48 layers '
                                        '= 58; descriptive, not a completeness '
                                        'count: a paused QSA continuation is '
                                        're-issued under a new stage-local id'),
                'shape_basis': ('MAX_MODEL_LEN=262144 / 1024-token chunks = 256 '
                                'chunks; plan = 10 + 48 layers = 58 segments; '
                                '32 requests = 3 lanes x (fresh+append) x 4 '
                                'declared repeats + headroom'),
            },
            'ingested': {
                'total': self._ingested,
                'measurement': self._measurement_events,
                'ignored_non_measurement': self._ignored_events,
                'by_kind': dict(sorted(self._by_kind.items())),
                'duplicate_observations': self._duplicate_observations,
                'first_seen_proofs': self._first_seen_proofs,
            },
            'retained': {
                'chunks': len(self._chunks),
                'requests': len(self._groups),
                'orphan_transactions': len(self._orphans),
                'standalone_cancellations': len(self._standalone_cancels),
            },
            'overflow': {
                'count': self._overflow_count,
                'reasons': list(self._overflow_reasons),
            },
            'faults': {
                'count': self._fault_count,
                'reasons': list(self._faults),
            },
            'global_reasons': list(self._global_reasons),
            'host_bytes': host_bytes,
            'projected_host_bytes_at_capacity': {
                'chunks': self.max_chunks_total,
                'per_chunk_bytes': (round(per_chunk, 1)
                                    if per_chunk is not None else None),
                'bytes': projected,
                'basis': ('measured retained bytes / retained chunks x '
                          'max_chunks_total; a stated bound, not an allocation'),
            },
        }

    def host_bytes_estimate(self):
        """Deterministic ``sys.getsizeof`` walk of the retained state."""
        payload = {
            'chunks': self._chunks,
            'groups': self._groups,
            'orphans': self._orphans,
            'standalone_cancels': self._standalone_cancels,
        }
        try:
            return {
                'bytes': _deep_size(payload, [HOST_BYTES_WALK_LIMIT]),
                'basis': ('sys.getsizeof recursive estimate over retained '
                          'compact state; excludes interned/immortal objects'),
                'walk_limit': HOST_BYTES_WALK_LIMIT,
            }
        except Exception as exc:
            return {'bytes': None,
                    'basis': 'unavailable:' + type(exc).__name__}

    # -- evidence ----------------------------------------------------------
    def evidence(self, start_ns, end_ns, *, now_ns=None):
        """Return this census's window evidence.  Never raises."""
        try:
            return self._evidence(start_ns, end_ns, now_ns=now_ns)
        except Exception as exc:
            self._fault('evidence-fault:' + type(exc).__name__)
            return self._inconclusive(start_ns, end_ns, now_ns,
                                      ['measurement-evidence-fault:'
                                       + type(exc).__name__])

    def _inconclusive(self, start_ns, end_ns, now_ns, reasons):
        return {
            'schema': SCHEMA,
            'start_ns': start_ns if _is_int(start_ns) else None,
            'end_ns': end_ns if _is_int(end_ns) else None,
            'observed_clock_ns': now_ns if _is_int(now_ns) else None,
            'clock_basis': self.clock_basis,
            'engine_epoch': self.engine_epoch,
            'segments': [],
            'completions': [],
            'source_holds': [],
            'committed_frontiers': [],
            'pp_consumer_receipts': [],
            'cancellations': [],
            'invalid_segments': [],
            'groups': [],
            'prefill': self._empty_prefill(),
            'serialized_transport': False,
            'route_generations': [],
            'reasons': sorted(set(reasons)),
            'construction_valid': False,
            'status': 'INCONCLUSIVE',
            'not_a_performance_pass': True,
            'completeness': self._completeness_receipt(
                None, None, reasons, {}, {}, {}),
        }

    @staticmethod
    def _empty_prefill():
        return {
            'useful_completed_tokens': 0,
            'completed_prefills': 0,
            'prefill_wall_ns': 0,
            'prefill_wall_s': None,
            'prompt_tokens': None,
            'prompt_tokens_basis': ('one window may span several requests; each '
                                    "request's chunk_open.prompt_end is retained "
                                    'in groups and segments'),
            'basis': ('measurement census: admitted chunk token spans and '
                      'first accepted work .. commit host-monotonic '
                      'observations; not device elapsed time'),
        }

    def _evidence(self, start_ns, end_ns, *, now_ns):
        reasons = []
        if not self.config_ok:
            reasons.extend(self._config_reasons)
        if not _is_int(start_ns) or not _is_int(end_ns) \
                or start_ns <= 0 or end_ns <= 0 or end_ns <= start_ns:
            reasons.append('measurement-window-invalid')
            return self._inconclusive(start_ns, end_ns, now_ns, reasons)
        in_window = []
        for chunk in self._chunks.values():
            if start_ns <= chunk['open_ns'] <= end_ns:
                in_window.append(chunk)
        in_window.sort(key=lambda item: (item['open_ns'], item['transaction']))
        if not in_window:
            reasons.append('no-measurement-chunks-in-window')

        # The census is all-or-nothing: an overflow, fault or global identity
        # conflict means it can no longer prove it retained the window.
        if self._overflow_count:
            reasons.append('measurement-capacity-overflow')
            reasons.extend('overflow:' + value for value in self._overflow_reasons)
        if self._fault_count:
            reasons.append('measurement-ingest-fault')
            reasons.extend('fault:' + value for value in self._faults)
        reasons.extend(self._global_reasons)

        reconcile = {}
        segments = []
        completions = []
        source_holds = []
        frontiers = []
        pp_receipts = []
        cancellations = []
        accepted_chunks = 0
        useful_tokens = 0
        prefill_wall_ns = 0
        route_generations = set()
        serialized = False
        for chunk in in_window:
            result = self._reconcile_chunk(chunk, end_ns)
            reconcile[chunk['transaction']] = result
            if result['accepted']:
                accepted_chunks += 1
                useful_tokens += result['chunk_tokens']
                if result['first_work_ns'] is not None \
                        and result['commit_ns'] is not None:
                    prefill_wall_ns += (result['commit_ns']
                                        - result['first_work_ns'])
            if result['transport'] != _ACCEPTED_TRANSPORT:
                serialized = True
            route_generations.add(chunk['route_generation'])
            segments.append(self._segment_row(chunk, result))
            completions.extend(self._completion_rows(chunk, result))
            if chunk['source_hold'] is not None:
                hold = chunk['source_hold']
                source_holds.append({
                    'transaction': chunk['transaction'],
                    'request': chunk['request'],
                    'request_generation': chunk['generation'],
                    'description': hold['description'],
                    'retired': hold['retired'],
                    'observed_ns': hold['retired_ns'] or hold['first_ns'],
                    'clock_basis': self.clock_basis,
                })
            if chunk['frontier'] is not None:
                frontier = chunk['frontier']
                frontiers.append({
                    'transaction': chunk['transaction'],
                    'request': chunk['request'],
                    'request_generation': chunk['generation'],
                    'frontier': {
                        'committed_end': frontier['committed_end'],
                        'remaining': frontier['remaining'],
                        'state_epoch': frontier['state_epoch'],
                        'ranks': list(frontier['ranks']),
                        'pp_seqs': list(frontier['pp_seqs']),
                    },
                    'observed_ns': frontier['observed_ns'],
                    'clock_basis': self.clock_basis,
                })
            for sender, receipt in sorted(chunk['pp'].items()):
                pp_receipts.append({
                    'transaction': chunk['transaction'],
                    'sender_rank': sender,
                    'seq': receipt['seq'],
                    'completion_seq': receipt['completion_seq'],
                    'observed_ns': receipt['observed_ns'],
                    'clock_basis': self.clock_basis,
                    'accepted': True,
                })
            for row in chunk['cancels']:
                cancellations.append(dict(row))
        for row in self._standalone_cancels:
            if start_ns <= row['observed_ns'] <= end_ns:
                cancellations.append(dict(row))

        groups = self._reconcile_groups(start_ns, end_ns, reconcile)
        groups_complete = bool(groups) and all(group['complete'] for group in groups)
        expected_chunks = len(in_window)
        cancelled_chunks = sum(1 for chunk in in_window if chunk['cancelled'])
        in_window_standalone = sum(
            1 for row in self._standalone_cancels
            if start_ns <= row['observed_ns'] <= end_ns)
        if expected_chunks == 0:
            reasons.append('no-measurement-chunks-in-window')
        if accepted_chunks != expected_chunks:
            reasons.append('measurement-chunks-not-reconciled')
        if not groups_complete:
            reasons.append('measurement-requests-incomplete')
        if cancelled_chunks or in_window_standalone:
            reasons.append('measurement-cancellation-in-window')

        # Partial/conflicted chunks stay visible with their own reasons.
        for chunk in in_window:
            for value in reconcile[chunk['transaction']]['reasons']:
                reasons.append('chunk-%d:%s' % (chunk['transaction'], value))
        for group in groups:
            for value in group['reasons']:
                reasons.append('request-%s#%s:%s'
                               % (group['request'],
                                  group['request_generation'], value))

        complete = (self.config_ok and not reasons
                    and expected_chunks > 0
                    and accepted_chunks == expected_chunks
                    and groups_complete)
        prefill = {
            'useful_completed_tokens': useful_tokens,
            'completed_prefills': accepted_chunks,
            'prefill_wall_ns': prefill_wall_ns,
            'prefill_wall_s': (round(prefill_wall_ns / 1e9, 9)
                               if prefill_wall_ns else None),
            'prompt_tokens': None,
            'prompt_tokens_basis': ('one window may span several requests; each '
                                    "request's chunk_open.prompt_end is retained "
                                    'in groups and segments'),
            'basis': ('measurement census: admitted chunk token spans and first '
                      'accepted work .. commit host-monotonic observations; not '
                      'device elapsed time'),
        }
        census = self.stats()
        completeness = self._completeness_receipt(
            start_ns, end_ns, sorted(set(reasons)), census,
            {'expected_chunks': expected_chunks,
             'accepted_chunks': accepted_chunks,
             'cancelled_chunks': cancelled_chunks,
             'standalone_cancellations': in_window_standalone,
             'groups': len(groups),
             'complete_groups': sum(1 for group in groups if group['complete']),
             'first_seen_proofs': self._first_seen_proofs},
            {'chunks': [self._chunk_completeness_row(chunk, reconcile)
                        for chunk in in_window],
             'groups': groups})
        result = {
            'schema': SCHEMA,
            'start_ns': start_ns,
            'end_ns': end_ns,
            'observed_clock_ns': now_ns if _is_int(now_ns) else None,
            'clock_basis': self.clock_basis,
            'engine_epoch': self.engine_epoch,
            'segments': segments,
            'completions': completions,
            'source_holds': source_holds,
            'committed_frontiers': frontiers,
            'pp_consumer_receipts': pp_receipts,
            'cancellations': cancellations,
            'invalid_segments': [row for row in segments
                                 if row.get('accepted') is not True],
            'groups': groups,
            'prefill': prefill,
            'serialized_transport': serialized,
            'route_generations': sorted(route_generations),
            'reasons': sorted(set(reasons)),
            'construction_valid': complete,
            'status': 'COMPLETE' if complete else 'INCONCLUSIVE',
            'not_a_performance_pass': True,
            'completeness': completeness,
        }
        try:
            result['completeness']['serialized_receipt_bytes'] = len(
                json.dumps(result, separators=(',', ':'), default=repr))
        except Exception:
            result['completeness']['serialized_receipt_bytes'] = None
        return result

    def _completeness_receipt(self, start_ns, end_ns, reasons, census,
                              reconciliation, detail):
        complete = (self.config_ok and not reasons
                    and reconciliation.get('expected_chunks', 0) > 0
                    and reconciliation.get('expected_chunks')
                    == reconciliation.get('accepted_chunks')
                    and reconciliation.get('groups', 0)
                    == reconciliation.get('complete_groups', 0))
        return {
            'schema': SCHEMA,
            'basis': MEASUREMENT_BASIS,
            'reconciliation_basis': RECONCILIATION_BASIS,
            'independent_of_raw_journal': True,
            'raw_journal_consulted': False,
            'raw_journal_flag_mutated': False,
            'status': 'COMPLETE' if complete else 'INCONCLUSIVE',
            'complete': complete,
            'not_a_performance_pass': True,
            'window': {'start_ns': start_ns if _is_int(start_ns) else None,
                       'end_ns': end_ns if _is_int(end_ns) else None},
            'reasons': list(reasons),
            'census': census,
            'reconciliation': dict(reconciliation),
            'detail': detail,
        }

    @staticmethod
    def _chunk_completeness_row(chunk, reconcile):
        result = reconcile[chunk['transaction']]
        return {
            'transaction': chunk['transaction'],
            'request': chunk['request'],
            'request_generation': chunk['generation'],
            'chunk_start': chunk['chunk_start'],
            'chunk_end': chunk['chunk_end'],
            'accepted': result['accepted'],
            'state_write_ranks': result['state_write_ranks'],
            'reasons': list(result['reasons']),
        }

    # -- per-chunk reconciliation -----------------------------------------
    @staticmethod
    def _stage_segments(seg):
        """Retained ``{stage: {op_seq: segment_kind}}`` from accepted events."""
        by_stage = {}
        for (stage, op_seq), attributes in seg['attrs'].items():
            by_stage.setdefault(stage, {})[op_seq] = attributes[0]
        return by_stage

    def _reconcile_chunk(self, chunk, end_ns):
        reasons = []
        seg = chunk['segments']
        commit = chunk['commit']
        first_work_ns = seg['first_ns']
        commit_ns = commit['observed_ns'] if commit is not None else None
        transport = seg['transport']
        if chunk['cancelled']:
            _reason(reasons, 'chunk-cancelled')
        if chunk['saw_orphan']:
            _reason(reasons, 'event-before-chunk-open')
        if chunk['route'] != _ACCEPTED_ROUTE:
            _reason(reasons, 'chunk-route-not-split')
        if seg['serialized'] or not _transport_is_accepted(transport):
            _reason(reasons, 'serialized-or-unproved-transport')
        if seg['unique'] == 0:
            _reason(reasons, 'no-accepted-native-segment')
        by_stage = self._stage_segments(seg)
        if set(by_stage) != set(self.stage_ranks):
            _reason(reasons, 'missing-pp-stage-segment')
        if first_work_ns is None:
            _reason(reasons, 'first-accepted-work-unproved')
        elif first_work_ns > end_ns:
            _reason(reasons, 'first-accepted-work-after-window')
        state_write_ranks = []
        reconciled_proofs = 0
        for rank in self.ranks:
            record = chunk['ranks'].get(rank)
            if record is None:
                _reason(reasons, 'missing-rank-report:%d' % rank)
                continue
            if record['failed_code'] is not None:
                _reason(reasons, 'rank-failed:%d' % rank)
            if record['watermark_regressed']:
                _reason(reasons, 'report-watermark-regressed:%d' % rank)
            if record['watermark_mismatch']:
                _reason(reasons, 'report-watermark-differs-from-proof:%d' % rank)
            if record['completed_set_regressed']:
                _reason(reasons, 'report-completed-set-regressed:%d' % rank)
            if record['unique_proofs'] == 0:
                _reason(reasons, 'missing-completion-proof:%d' % rank)
            if record['state_flag_missing_proof']:
                _reason(reasons, 'state-written-without-proof:%d' % rank)
            if record['proof_without_state_flag']:
                _reason(reasons, 'state-proof-without-flag:%d' % rank)
            if record['pending_after_completion']:
                _reason(reasons, 'pending-after-completion:%d' % rank)
            stage = self.rank_stage.get(rank)
            if stage is None:
                _reason(reasons, 'rank-stage-unproved:%d' % rank)
            else:
                stage_ops = by_stage.get(stage, {})
                expected_ops = set(stage_ops)
                if not expected_ops:
                    _reason(reasons, 'stage-without-accepted-segment:%d' % stage)
                completed_ops = record['completed_ops']
                if completed_ops is None:
                    _reason(reasons, 'missing-completed-op-set:%d' % rank)
                elif set(completed_ops) != expected_ops:
                    # The rank's own cumulative source fact and the accepted
                    # stage ids must describe the same operations.
                    _reason(reasons,
                            'segment-set-differs-from-report:%d' % rank)
                gpu_ops = {op_seq for (kind, op_seq) in record['proofs']
                           if kind == 'gpu_event'}
                if gpu_ops != expected_ops:
                    _reason(reasons,
                            'gpu-event-set-differs-from-report:%d' % rank)
                # Proof roles are typed by the descriptor they name: a
                # STATE_WRITE must belong to this stage's STATE_FINALIZE and a
                # PP_CONSUMER to its PP_RECEIVE.
                for (kind, op_seq) in record['proofs']:
                    if kind == 'state_write' \
                            and stage_ops.get(op_seq) != 'state_finalize':
                        _reason(reasons,
                                'state-write-role-mismatch:%d/%d'
                                % (rank, op_seq))
                    if kind == 'pp_consumer' \
                            and stage_ops.get(op_seq) != 'pp_receive':
                        _reason(reasons,
                                'pp-consumer-role-mismatch:%d/%d'
                                % (rank, op_seq))
            state_write = record['state_write']
            if state_write is None:
                _reason(reasons, 'missing-state-write:%d' % rank)
            else:
                state_write_ranks.append(rank)
                reconciled_proofs += 1
                if state_write['observed_ns'] > end_ns:
                    _reason(reasons, 'state-write-after-window:%d' % rank)
                if commit_ns is not None and state_write['observed_ns'] > commit_ns:
                    _reason(reasons, 'completion-after-commit:%d' % rank)
            for proof in record['proofs'].values():
                if commit_ns is not None and proof['observed_ns'] > commit_ns:
                    _reason(reasons, 'proof-after-commit:%d' % rank)
                    break
        # Every addressable state-finalize and pp-receive segment must carry
        # its typed queried proof on every rank of its stage.
        for stage, ops in sorted(by_stage.items()):
            members = self.stage_ranks.get(stage, ())
            state_ops = sorted(op for op, kind in ops.items()
                               if kind == 'state_finalize')
            receive_ops = sorted(op for op, kind in ops.items()
                                 if kind == 'pp_receive')
            if not state_ops:
                _reason(reasons, 'stage-without-state-finalize:%d' % stage)
            for rank in members:
                record = chunk['ranks'].get(rank)
                if record is None:
                    continue
                for op_seq in state_ops:
                    if ('state_write', op_seq) not in record['proofs']:
                        _reason(reasons, 'missing-state-write-proof:%d/%d'
                                % (rank, op_seq))
                for op_seq in receive_ops:
                    if ('pp_consumer', op_seq) not in record['proofs']:
                        _reason(reasons,
                                'missing-pp-consumer-proof:%d/%d'
                                % (rank, op_seq))
        for sender in self.pp_senders:
            receipt = chunk['pp'].get(sender)
            if receipt is None:
                _reason(reasons, 'missing-pp-consumer-proof:%d' % sender)
            elif receipt['observed_ns'] > end_ns:
                _reason(reasons, 'pp-consumer-proof-after-window:%d' % sender)
        if commit is None:
            _reason(reasons, 'missing-chunk-commit')
        else:
            if commit_ns > end_ns:
                _reason(reasons, 'chunk-commit-after-window')
            if first_work_ns is not None and commit_ns <= first_work_ns:
                _reason(reasons, 'commit-before-accepted-work')
            if set(commit['releases']) != set(self.ranks):
                _reason(reasons, 'release-ranks-incomplete')
        hold = chunk['source_hold']
        if hold is None:
            _reason(reasons, 'missing-source-hold')
        elif not hold['retired']:
            _reason(reasons, 'source-hold-not-retired')
        elif type(hold['description']) is not dict:
            # The real driver records ``hold.describe()``; without it there is
            # no source identity to join, only a flag.
            _reason(reasons, 'source-hold-description-unproved')
        elif commit_ns is not None and hold['retired_ns'] < commit_ns:
            _reason(reasons, 'source-released-before-commit')
        frontier = chunk['frontier']
        if frontier is None:
            _reason(reasons, 'missing-committed-frontier')
        else:
            if commit_ns is not None and frontier['observed_ns'] < commit_ns:
                _reason(reasons, 'frontier-before-commit')
            if frontier['committed_end'] != chunk['chunk_end']:
                _reason(reasons, 'frontier-committed-end-differs')
            if frontier['remaining'] != chunk['remaining_tokens']:
                _reason(reasons, 'frontier-remaining-differs')
            if frontier['state_epoch'] != chunk['state_epoch']:
                _reason(reasons, 'frontier-state-epoch-differs')
            if set(frontier['ranks']) != set(self.ranks):
                _reason(reasons, 'frontier-ranks-incomplete')
            if self.pp_senders:
                # One PP transfer seq is shared by the stage-0 senders for the
                # chunk; every accepted sender receipt must name it.
                if not frontier['pp_seqs']:
                    _reason(reasons, 'frontier-pp-seqs-incomplete')
                else:
                    for sender in self.pp_senders:
                        receipt = chunk['pp'].get(sender)
                        if receipt is not None \
                                and receipt['seq'] not in frontier['pp_seqs']:
                            _reason(reasons,
                                    'frontier-pp-seq-differs:%d' % sender)
        for value in chunk['conflicts']:
            _reason(reasons, value)
        accepted = not reasons
        if accepted and state_write_ranks:
            final_state_write_ns = max(
                chunk['ranks'][rank]['state_write']['observed_ns']
                for rank in state_write_ranks)
        else:
            final_state_write_ns = None
        return {
            'accepted': accepted,
            'reasons': reasons,
            'transport': (transport if _transport_is_accepted(transport)
                          else ('serialized'
                                if seg['serialized'] or _transport_is_descriptive(transport)
                                else 'unproved')),
            'first_work_ns': first_work_ns,
            'commit_ns': commit_ns,
            'final_state_write_ns': final_state_write_ns,
            'state_write_ranks': sorted(state_write_ranks),
            'chunk_tokens': chunk['chunk_end'] - chunk['chunk_start'],
            'proofs': reconciled_proofs,
        }

    # -- output rows -------------------------------------------------------
    def _segment_row(self, chunk, result):
        stage_segments = self._stage_segments(chunk['segments'])
        return {
            'transaction': chunk['transaction'],
            'request': chunk['request'],
            'request_generation': chunk['generation'],
            'prefill': self._family(chunk),
            'initial_frontier': self._group_initial_frontier(chunk),
            'committed_end': chunk['committed_end'],
            'prompt_end': chunk['prompt_end'],
            'chunk_start': chunk['chunk_start'],
            'chunk_end': chunk['chunk_end'],
            'chunk_tokens': result['chunk_tokens'],
            'terminal': chunk['terminal'],
            'start_monotonic_ns': result['first_work_ns'],
            'end_monotonic_ns': result['commit_ns'],
            'final_queried_state_write_ns': result['final_state_write_ns'],
            'transport': result['transport'],
            'accepted': result['accepted'],
            'state_write': (set(result['state_write_ranks']) == set(self.ranks)
                            and bool(self.ranks)),
            'state_write_ranks': list(result['state_write_ranks']),
            'report_count': sum(record['reports']
                                for record in chunk['ranks'].values()),
            'segment_count': chunk['segments']['unique'],
            'segment_events': chunk['segments']['events'],
            'segment_identity': '(stage, op_seq)',
            'stage_segment_counts': {
                str(stage): len(ops)
                for stage, ops in sorted(stage_segments.items())},
            'plan_segments_per_chunk': self.expected_segments_per_chunk,
            'plan_basis': ('driver _build_plan length; descriptive only, a '
                           'paused QSA continuation adds stage-local ids'),
            'proof_count': sum(record['unique_proofs']
                               for record in chunk['ranks'].values()),
            'proof_repeats': sum(record['proof_repeats']
                                 for record in chunk['ranks'].values()),
            'decode_progress': None,
            'decode_progress_basis': 'not-queried-engine-side',
            'worker_releases': (
                [{'worker_rank': rank,
                  'released': row['released'],
                  'closed': row['closed'],
                  'completed_sequence': row['completed_sequence']}
                 for rank, row in sorted(chunk['commit']['releases'].items())]
                if chunk['commit'] is not None else None),
            'clock_basis': self.clock_basis,
            'engine_epoch': self.engine_epoch,
            'route': chunk['route'],
            'route_generation': chunk['route_generation'],
            'reasons': list(result['reasons']),
        }

    def _completion_rows(self, chunk, result):
        rows = []
        for rank in self.ranks:
            record = chunk['ranks'].get(rank)
            if record is None or record['reports'] == 0:
                continue
            proofs = []
            for (kind, op_seq), proof in sorted(record['proofs'].items()):
                proofs.append({
                    'kind': kind,
                    'op_seq': op_seq,
                    'completion_seq': proof['completion_seq'],
                    'completed_ns': proof['observed_ns'],
                    'time_basis': PROOF_TIME_BASIS,
                })
            state_write = record['state_write']
            completed_ops = (sorted(record['completed_ops'])
                             if record['completed_ops'] is not None else None)
            rows.append({
                'request': chunk['request'],
                'request_generation': chunk['generation'],
                'transaction': chunk['transaction'],
                'worker_rank': rank,
                'stage': self.rank_stage.get(rank),
                'queried': True,
                'completed_sequence': record['completed_sequence'],
                'completed_ops': completed_ops,
                'completed_ops_basis': ('union of cumulative '
                                        'completed_op_seqs reports'),
                'state_written': state_write is not None,
                'state_write_ns': (state_write['observed_ns']
                                   if state_write is not None else None),
                # First queried proof observation, not a later cumulative
                # duplicate and not the first (possibly empty) poll.
                'observed_ns': (record['first_proof_ns']
                                if record['first_proof_ns'] is not None
                                else record['first_report_ns']),
                'clock_basis': self.clock_basis,
                'proofs': proofs,
                'proof_repeats': record['proof_repeats'],
                'pending_samples': record['pending_samples'],
                'pending_after_completion': record['pending_after_completion'],
                'reasons': list(result['reasons']) if not result['accepted']
                           else [],
            })
        return rows

    # -- request groups ----------------------------------------------------
    def _group_initial_frontier(self, chunk):
        group = self._groups[(chunk['request'], chunk['generation'])]
        return group['initial_frontier']

    def _family(self, chunk):
        group = self._groups[(chunk['request'], chunk['generation'])]
        return group['family']

    def _reconcile_groups(self, start_ns, end_ns, reconcile):
        groups = []
        for key in self._group_order:
            group = self._groups[key]
            chunk_rows = []
            reasons = list(group['conflicts'])
            outside = 0
            for txn in group['chunks']:
                chunk = self._chunks[txn]
                if start_ns <= chunk['open_ns'] <= end_ns:
                    chunk_rows.append(chunk)
                else:
                    outside += 1
            if not chunk_rows:
                continue
            chunk_rows.sort(key=lambda item: item['chunk_start'])
            if outside:
                _reason(reasons, 'request-extends-outside-window')
            accepted = [item for item in chunk_rows
                        if reconcile[item['transaction']]['accepted']]
            if len(accepted) != len(chunk_rows):
                _reason(reasons, 'request-chunks-not-accepted')
            initial = group['initial_frontier']
            if chunk_rows[0]['chunk_start'] != initial:
                _reason(reasons, 'request-initial-frontier-not-observed')
            previous_end = None
            terminal_seen = 0
            for chunk in chunk_rows:
                if previous_end is None:
                    if chunk['chunk_start'] != initial:
                        _reason(reasons, 'request-frontier-gap')
                elif chunk['chunk_start'] != previous_end:
                    _reason(reasons, 'request-frontier-gap')
                previous_end = chunk['chunk_end']
                if chunk['terminal']:
                    terminal_seen += 1
                    if chunk['remaining_tokens'] != 1 \
                            or chunk['chunk_end'] != chunk['prompt_end'] - 1:
                        _reason(reasons, 'request-terminal-coverage-invalid')
            if terminal_seen != 1:
                _reason(reasons, 'request-terminal-not-covered')
            last = chunk_rows[-1]
            if not last['terminal']:
                _reason(reasons, 'request-terminal-outside-window')
            if last['chunk_end'] != last['prompt_end'] - 1:
                _reason(reasons, 'request-final-frontier-incomplete')
            family = group['family']
            if family not in ('fresh', 'append'):
                _reason(reasons, 'request-family-unproved')
            internal = [reconcile[item['transaction']] for item in chunk_rows]
            start = None
            end = None
            useful = 0
            if accepted:
                starts = [row['first_work_ns'] for row in internal
                          if row['first_work_ns'] is not None]
                ends = [row['final_state_write_ns'] for row in internal
                        if row['final_state_write_ns'] is not None]
                if starts:
                    start = min(starts)
                if ends:
                    end = max(ends)
                useful = sum(row['chunk_tokens'] for row in internal
                             if row['accepted'])
            groups.append({
                'request': group['request'],
                'request_generation': group['generation'],
                'family': family,
                'initial_frontier': initial,
                'prompt_end': group['prompt_end'],
                'prompt_tokens': group['prompt_end'],
                'prompt_tokens_basis': ('chunk_open.prompt_end is '
                                        'request.num_tokens at admission'),
                'chunk_start': chunk_rows[0]['chunk_start'],
                'chunk_end': last['chunk_end'],
                'terminal': last['terminal'],
                'chunks': len(chunk_rows),
                'accepted_chunks': len(accepted),
                'useful_tokens': useful,
                'start_ns': start,
                'end_ns': end,
                'wall_ns': (end - start if start is not None and end is not None
                            and end > start else None),
                'wall_s': (round((end - start) / 1e9, 9)
                           if start is not None and end is not None
                           and end > start else None),
                'basis': ('first accepted native work .. final first-queried '
                          'STATE_WRITE observation'),
                'complete': not reasons,
                'reasons': sorted(set(reasons)),
            })
        return groups


def measurement_event_kinds():
    """The exact engine-validated entry kinds this owner ingests."""
    return MEASUREMENT_KINDS


__all__ = [
    'CLOCK_BASIS', 'MAX_CHUNKS_PER_REQUEST', 'MAX_CHUNKS_TOTAL',
    'MAX_MODEL_TOKENS', 'MAX_PROOFS_PER_RANK', 'MAX_REQUESTS',
    'MAX_SEGMENTS_PER_CHUNK', 'MEASUREMENT_BASIS', 'MEASUREMENT_KINDS',
    'NATIVE_CHUNK_TOKENS', 'NativeMeasurementCensus', 'PROOF_TIME_BASIS',
    'RECONCILIATION_BASIS', 'SCHEMA', 'SEGMENTS_PER_CHUNK',
    'measurement_event_kinds',
]


def select_window_evidence(raw, begin, start_ns, end_ns):
    """Bind the driver's independent census to this exact measured window.

    Return a view for calculations, keeping the caller's raw packet untouched.
    A COMPLETE flag alone never supplies data or an identity. The old raw ring
    remains descriptive even when its retained tail is insufficient.
    """
    binding = {'basis': MEASUREMENT_BASIS, 'complete': False,
               'route_proved': False, 'route_basis': None, 'reasons': []}
    reasons = binding['reasons']
    view = raw if type(raw) is dict else {}
    try:
        if type(raw) is not dict or type(begin) is not dict:
            raise ValueError('measurement-envelope-missing')
        measurement = raw.get('measurement')
        if type(measurement) is not dict:
            raise ValueError('measurement-census-missing')
        comp = measurement.get('completeness')
        if type(comp) is not dict or comp.get('basis') != MEASUREMENT_BASIS:
            raise ValueError('measurement-basis-unproved')
        for record in (raw, measurement):
            if record.get('engine_epoch') != begin.get('engine_epoch'):
                raise ValueError('measurement-engine-epoch-differs')
            if record.get('clock_basis') != CLOCK_BASIS:
                raise ValueError('measurement-clock-basis-differs')
            if record.get('start_ns') != start_ns or record.get('end_ns') != end_ns:
                raise ValueError('measurement-window-differs')
        if comp.get('window') != {'start_ns': start_ns, 'end_ns': end_ns}:
            raise ValueError('measurement-completeness-window-differs')
        if comp.get('raw_journal_flag_mutated') is not False:
            raise ValueError('measurement-raw-loss-guard-unproved')
        if comp.get('raw_journal_consulted') is not False:
            raise ValueError('measurement-raw-source-consulted')
        if comp.get('independent_of_raw_journal') is not True:
            raise ValueError('measurement-independent-source-unproved')
        generation = begin.get('route_generation')
        if type(generation) is not int or measurement.get('route_generations') != [generation]:
            raise ValueError('measurement-route-generation-differs')
        ranks = begin.get('ranks')
        if type(ranks) is not list or not ranks or any(type(r) is not int for r in ranks) or len(set(ranks)) != len(ranks):
            raise ValueError('measurement-rank-set-unproved')
        segments = measurement.get('segments')
        completions = measurement.get('completions')
        groups = measurement.get('groups')
        if not all(type(x) is list and x for x in (segments, completions, groups)):
            raise ValueError('measurement-source-rows-missing')
        txns = {row['transaction'] for row in segments}
        if len(txns) != len(segments):
            raise ValueError('measurement-duplicate-transaction')
        if any(row.get('state_write_ranks') != sorted(ranks) for row in segments):
            raise ValueError('measurement-state-write-ranks-differ')
        coordinates = {(row['transaction'], row['worker_rank']) for row in completions}
        if len(coordinates) != len(completions) or coordinates != {(t, r) for t in txns for r in ranks}:
            raise ValueError('measurement-completion-coordinates-differ')
        if not (comp.get('complete') is True and comp.get('status') == 'COMPLETE'
                and measurement.get('status') == 'COMPLETE'
                and measurement.get('construction_valid') is True
                and measurement.get('reasons') == [] and comp.get('reasons') == []
                and all(row.get('accepted') is True for row in segments)
                and all(row.get('complete') is True for row in groups)
                and all(row.get('queried') is True for row in completions)):
            raise ValueError('measurement-census-incomplete')
        view = dict(raw)
        for key in ('segments', 'completions', 'source_holds', 'committed_frontiers',
                    'pp_consumer_receipts', 'cancellations', 'prefill', 'serialized_transport'):
            view[key] = measurement[key]
        binding['complete'] = True
        if raw.get('route') == 'split' and raw.get('route_generation') == generation:
            binding.update(route_proved=True, route_basis='queried-split-route')
        elif raw.get('route') == 'full' and raw.get('route_generation') == generation + 1 and raw.get('native_owned') is False:
            # The frozen headline ends at the final queried STATE_WRITE.
            # A later FULL return is a terminal tail, but only actual all-rank
            # drain/transition ACKs can certify that; stale client status cannot.
            acks = raw.get('transition_acks')
            from wire_v1 import ROUTE_BANK_FULL
            final_commit = max(row['end_monotonic_ns'] for row in segments)
            if (type(acks) is list and len(acks) == len(ranks)
                    and {row.get('worker_rank') for row in acks} == set(ranks)
                    and all(row.get('typed') == 'TransitionAck'
                        and row.get('engine_epoch') == begin['engine_epoch']
                        and row.get('route_generation') == generation + 1
                        and row.get('applied_route_generation') == generation + 1
                        and row.get('expected_route_generation') == generation
                        and row.get('target_bank') == ROUTE_BANK_FULL
                        and row.get('applied_bank') == ROUTE_BANK_FULL
                        and row.get('clock_basis') == CLOCK_BASIS
                        and row.get('applied_storage_epochs') == [
                            (begin.get('storage_epochs_by_rank') or {}).get(rank,
                              (begin.get('storage_epochs_by_rank') or {}).get(str(rank)))
                            for rank in sorted(ranks)]
                        and type(row.get('drained_streams')) is int and row['drained_streams'] > 0
                        and type(row.get('observed_ns')) is int
                        and final_commit <= row['observed_ns'] <= raw['observed_clock_ns']
                        for row in acks)):
                binding.update(route_proved=True, route_basis='queried-all-rank-terminal-full-tail')
        if not binding['route_proved']:
            reasons.append('measurement-end-route-unproved')
    except (ValueError, TypeError, KeyError) as exc:
        reasons.append(str(exc) if type(exc) is ValueError else 'measurement-envelope-malformed')
    binding['valid'] = binding['complete'] and binding['route_proved'] and not reasons
    return view, binding
