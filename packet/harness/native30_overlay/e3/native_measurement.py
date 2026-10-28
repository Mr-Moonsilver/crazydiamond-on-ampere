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
                    or not _is_int(layer_end) or layer_end != layer_start + 1:
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
