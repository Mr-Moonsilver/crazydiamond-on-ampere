"""Worker side of native performance control (CPU contract; no CUDA on import).

The generated worker RPC ``flash_s5_native_overlap`` reaches :func:`handle` on
each GPU worker through the ordinary ordered RPC transport.  Every answer is a
row with the exact :data:`ROW_FIELDS` vocabulary, copied from the worker's own
constructed native objects and from the runner state that the stock
dispatch/sample path updates:

* ``runner._flash_s5_native`` -- E's :class:`native_startup.WorkerNative` with
  the persistent decode storage owner, route controller, prefill stream owners
  and the native graph router;
* ``runner.req_states`` -- the runner's live request rows, read at the same
  dispatch/sample seams the scheduler writes them;
* ``runner._flash_s5_native_census`` -- :class:`NativeCensus`, fed by
  ``worker_hooks.observe_scheduler`` (every real runner dispatch) and
  ``worker_hooks.observe_sample`` (every real sampler call).  This seam is
  independent of the decode-only probe: the probe refuses a runner that also
  carries the native lane, so native mode cannot reuse it.

Completions are only ever queried: native replay leases leave
``GraphRouter.poll_native_replays`` on a boolean event query, and request rows
are read, never inferred from submission counters.  A missing construction
raises :class:`NativeOverlapUnavailable`; it is never reported as a clean or
successful status.

No torch/CUDA call happens on import.  The memory census imports torch lazily
inside :func:`measure_memory` and reports ``None`` (with an explicit
``unavailable`` marker) when the allocator cannot be queried.  It never calls
``synchronize``.
"""
import copy
import os
import time

SCHEMA = 1
from numbers import Integral

MODE = 'native-overlap'
CLOCK_BASIS = 'clock-monotonic'

ACTIONS = ('identity', 'census', 'status', 'barrier', 'drain', 'poll_native',
           'query', 'cancel')
QUALIFICATION_ACTIONS = ('teacher_configure', 'teacher_evidence',
                         'observer_begin', 'observer_end',
                         'row1_configure', 'row1_evidence')
# attn-meta (FLASH_ATTN_META_GDN_BATCH=1, default off): the per-arm selector (patches/s5_engine/attn_meta_select.py),
# answered in the row's ``qualification`` mapping; known only where the switch is on.
ATTN_META_ACTIONS = ('attn_meta_select', 'attn_meta_read')
_ATTN_META_FIELDS = {'attn_meta_select': {'action', 'on', 'engine_epoch', 'route_generation'},
                     'attn_meta_read': {'action', 'engine_epoch', 'route_generation'}}


def attn_meta_actions():
    return ATTN_META_ACTIONS if os.environ.get('FLASH_ATTN_META_GDN_BATCH', '0') == '1' else ()

# One exact field vocabulary for every worker row.  The engine refuses a row
# whose key set differs, so a worker that silently adds or drops an observation
# cannot pass as a typed answer.
ROW_INT_FIELDS = (
    'schema', 'worker_rank', 'pid', 'engine_epoch', 'route_generation',
    'storage_epoch', 'observed_clock_ns', 'decode_replay_watermark',
    'live_decode_requests', 'live_batch_rows', 'running_requests',
    'decode_eligible', 'prefill_requests', 'retained_rows', 'retained_active_rows',
    'pending_native_replays', 'polled_replays', 'pending_prefill_streams',
    'pending_output_copies', 'pp_consumer_streams', 'drained_streams',
    'drained_retained_rows', 'drained_native_replays',
)
ROW_STR_FIELDS = ('action', 'route', 'worker_incarnation', 'clock_basis')
ROW_BOOL_FIELDS = ('engine_epoch_ok', 'barrier_clean')
ROW_LIST_FIELDS = ('retained_request_uids', 'unavailable')
ROW_NULLABLE = ('allocator', 'memory', 'live_decode', 'qualification', 'native_inventory')
ROW_FIELDS = (ROW_INT_FIELDS + ROW_STR_FIELDS + ROW_BOOL_FIELDS
              + ROW_LIST_FIELDS + ROW_NULLABLE)


class NativeOverlapUnavailable(RuntimeError):
    """A required native construction or observation is absent."""


class NativeCensus:
    """CPU census written at the real runner dispatch and sample seams."""

    def __init__(self):
        self.scheduled = {}
        self.last_batch_req_ids = frozenset()
        self.steps = 0
        self.samples = 0
        self.empty_noops = 0
        self.rejected = []
        self.dispatches = 0
        self.decode_dispatches = 0
        self.non_decode_dispatches = 0
        self.full_graph_dispatches = 0
        self.dispatch_errors = 0

    def observe_scheduler(self, scheduler_output):
        scheduled = getattr(scheduler_output, 'num_scheduled_tokens', None)
        if type(scheduled) is not dict:
            self._reject('scheduler-tokens-unavailable')
            return
        rows = {}
        for request, tokens in scheduled.items():
            if type(request) is not str or type(tokens) is not int or tokens < 0:
                self._reject('scheduler-token-row-invalid')
                return
            rows[request] = tokens
        self.scheduled = rows
        self.last_batch_req_ids = frozenset(rows)
        self.steps += 1
        if not rows:
            # An empty dispatched batch is a meaningful no-op, not a failure.
            self.empty_noops += 1

    def observe_dispatch(self, num_reqs, num_tokens, uniform_tokens, batch_state, full_graph):
        """Observe the unpadded CPU batch at the existing graph route seam.

        The same source facts as the probe are observed without installing a
        probe, changing a dispatch or making a GPU read. Counters never reset.
        Missing metadata is a non-decode observation, never an assumed zero.
        """
        valid = (type(num_reqs) is int and num_reqs > 0
                 and type(num_tokens) is int and num_tokens == num_reqs
                 and uniform_tokens == 1 and batch_state is not None
                 and getattr(batch_state, 'has_prefill', None) is False)
        try:
            rows = getattr(batch_state, 'num_scheduled_tokens', None)
            prefills = getattr(batch_state, 'is_prefilling_np', None)
            valid = bool(valid and rows is not None and len(rows) == num_reqs
                         and all(isinstance(n, Integral) and not isinstance(n, bool)
                                 and int(n) == 1 for n in rows)
                         and prefills is not None and len(prefills) == num_reqs
                         and not any(bool(n) for n in prefills))
        except (TypeError, ValueError, OverflowError):
            self.dispatch_errors += 1
            valid = False
        self.dispatches += 1
        self.decode_dispatches += int(valid)
        self.non_decode_dispatches += int(not valid)
        self.full_graph_dispatches += int(full_graph is True)

    def observe_sample(self):
        self.samples += 1

    def _reject(self, reason):
        self.rejected.append(reason)
        del self.rejected[:-8]

    def record(self):
        return {'basis': 'worker-native-dispatch-sampler-v1',
                'steps': self.steps, 'samples': self.samples,
                'last_batch_member_count': len(self.last_batch_req_ids),
                'empty_noops': self.empty_noops,
                'dispatches': self.dispatches, 'decode_dispatches': self.decode_dispatches,
                'non_decode_dispatches': self.non_decode_dispatches,
                'full_graph_dispatches': self.full_graph_dispatches,
                'dispatch_errors': self.dispatch_errors,
                'rejected': list(self.rejected)}


def install_census(runner):
    """Install the dispatch/sample census once per runner."""
    existing = getattr(runner, '_flash_s5_native_census', None)
    if existing is not None:
        return existing
    census = NativeCensus()
    runner._flash_s5_native_census = census
    return census


def observe_scheduler(worker, scheduler_output):
    """Feed the census from the real runner dispatch seam (no probe involved)."""
    census = getattr(getattr(worker, 'model_runner', None),
                     '_flash_s5_native_census', None)
    if census is not None:
        census.observe_scheduler(scheduler_output)


def observe_sample(worker):
    census = getattr(getattr(worker, 'model_runner', None),
                     '_flash_s5_native_census', None)
    if census is not None:
        census.observe_sample()


def _worker_native(worker):
    runner = getattr(worker, 'model_runner', None)
    if runner is None:
        raise NativeOverlapUnavailable('native-overlap-runner-not-constructed')
    native = getattr(runner, '_flash_s5_native', None)
    if native is None or type(native) is dict:
        raise NativeOverlapUnavailable('native-overlap-lane-not-constructed')
    if getattr(native, 'controller', None) is None:
        raise NativeOverlapUnavailable('native-overlap-lane-not-constructed')
    return runner, native


def _state_identity(native):
    state = getattr(native, 'state', None)
    if state is None:
        raise NativeOverlapUnavailable('native-overlap-worker-state-unavailable')
    handle = getattr(getattr(state, 'decode', None), 'handle', None)
    storage_epoch = getattr(handle, 'storage_epoch', None)
    incarnation = getattr(state, 'incarnation', None)
    if type(storage_epoch) is not int or storage_epoch < 0:
        raise NativeOverlapUnavailable('native-overlap-retained-storage-unavailable')
    if type(incarnation) is not str or not incarnation:
        raise NativeOverlapUnavailable('native-overlap-incarnation-unavailable')
    return state, storage_epoch, incarnation


def _request_rows(runner):
    states = getattr(runner, 'req_states', None)
    rows = getattr(states, 'req_id_to_index', None)
    if type(rows) is not dict or not all(type(key) is str for key in rows):
        return None
    return rows


def live_census(runner, census):
    """The worker's own live-batch / decode-eligibility census, or ``None``.

    A partial subset of a decode cohort is expected under PP/parity: the last
    dispatched batch may be a strict subset of the live rows, and an empty
    batch is a real no-op.  Both are reported, never repaired into a full set.
    """
    rows = _request_rows(runner)
    if rows is None:
        return None
    states = runner.req_states
    lengths = getattr(getattr(states, 'prefill_len', None), 'np', None)
    computed = getattr(states, 'num_computed_prefill_tokens', None)
    if lengths is None or computed is None:
        return None
    prefill = 0
    for slot in rows.values():
        if type(slot) is not int or slot < 0:
            return None
        try:
            need, done = lengths[slot], computed[slot]
        except (IndexError, TypeError):
            return None
        if type(need) is bool or type(done) is bool:
            return None
        if not isinstance(need, Integral) or not isinstance(done, Integral):
            return None
        need, done = int(need), int(done)
        if need < 0 or done < 0:
            return None
        prefill += done < need
    live = len(rows)
    batch = frozenset(census.last_batch_req_ids) if census is not None else frozenset()
    scheduled = census.scheduled if census is not None else {}
    return {
        'basis': 'worker-native-live-decode-v1',
        'available': True,
        'live_requests': live,
        'prefill_requests': prefill,
        'decode_eligible': live - prefill,
        'batch_is_live_subset': bool(batch) and batch.issubset(rows),
        'last_batch_member_count': (len(batch) if census is not None else None),
        'empty_noop': bool(census is not None and not batch),
        'scheduled_requests': len(scheduled),
        'scheduled_tokens': sum(scheduled.values()),
        'live_request_uids': sorted(rows)[:64],
        'census': (census.record() if census is not None else None),
    }


def measure_memory(runner, *, torch_module=None):
    """Queried CUDA allocator census; ``None`` when it cannot be measured."""
    try:
        torch = torch_module if torch_module is not None else __import__('torch')
        if not torch.cuda.is_available() or not torch.cuda.is_initialized():
            return None, None, 'cuda-not-initialized'
        device = getattr(runner, 'device', None)
        index = device if type(device) is int else getattr(device, 'index', None)
        free, total = torch.cuda.mem_get_info(index)
        stats = torch.cuda.memory_stats(index)
        allocator = {
            'basis': 'torch-cuda-allocator-queried-v1',
            'device_index': index,
            'allocated_bytes': int(torch.cuda.memory_allocated(index)),
            'reserved_bytes': int(torch.cuda.memory_reserved(index)),
            'allocated_peak_bytes': int(stats.get('allocated_bytes.all.peak', 0)),
            'reserved_peak_bytes': int(stats.get('reserved_bytes.all.peak', 0)),
            'num_alloc_retries': int(stats.get('num_alloc_retries', 0)),
        }
        memory = {'basis': 'torch-cuda-mem-get-info-v1',
                  'device_index': index,
                  'free_bytes': int(free), 'total_bytes': int(total)}
        return allocator, memory, None
    except BaseException as exc:
        return None, None, 'memory-unqueried:' + type(exc).__name__


def _bridge_rows(runner):
    """Query the bridge's real open transaction and retained nonterminal row.

    ``active`` is true only while a transaction object is open.  A nonterminal
    release closes the transaction but deliberately keeps the request row in
    ``_retained`` for the next chunk, so an inactive bridge does *not* prove
    the row was released.  Both facts are read; neither certifies on its own.
    """
    bridge = getattr(runner, '_flash_s5_bridge', None)
    if bridge is None:
        raise NativeOverlapUnavailable('native-overlap-bridge-unavailable')
    active = getattr(bridge, 'active', None)
    if type(active) is not bool:
        raise NativeOverlapUnavailable('native-overlap-bridge-active-unavailable')
    retained = getattr(bridge, '_retained', None)
    uids = []
    pending = 0
    if retained is not None:
        if type(retained) is not dict:
            raise NativeOverlapUnavailable('native-overlap-retained-row-invalid')
        transaction = retained.get('transaction')
        report = retained.get('report')
        if transaction is None or report is None:
            raise NativeOverlapUnavailable('native-overlap-retained-row-incomplete')
        uid = getattr(transaction, 'request_uid', None)
        if type(uid) is not str or not uid:
            raise NativeOverlapUnavailable('native-overlap-retained-request-unavailable')
        ops = getattr(report, 'pending_op_seqs', None)
        if type(ops) not in (tuple, list):
            raise NativeOverlapUnavailable('native-overlap-retained-report-incomplete')
        uids.append(uid)
        pending = len(ops)
    return {'active': active,
            'retained_rows': int(active) + len(uids),
            'retained_active_rows': int(active),
            'retained_uids': uids,
            'retained_pending': pending}


def _poll_replays(native, limit):
    router = getattr(native, 'router', None)
    if router is None:
        return 0, 0
    completed = router.poll_native_replays(limit=limit)
    pending = getattr(router, 'native_pending', None)
    if type(pending) is not int or pending < 0:
        raise NativeOverlapUnavailable('native-overlap-replay-census-unavailable')
    return len(tuple(completed)), pending


def _drain(native):
    drained_prefill = native.drain_prefill_streams()
    drained = native.controller.drain()
    if type(drained_prefill) is not int or drained_prefill < 0:
        raise NativeOverlapUnavailable('native-overlap-prefill-drain-unproved')
    if type(drained) is not int or drained <= 0:
        raise NativeOverlapUnavailable('native-overlap-consumer-drain-unproved')
    return drained_prefill, drained


def _row(worker, native, action, command, clock, *, census, allocator, memory,
         unavailable, drained_streams=0, drained_retained=0,
         drained_native=0, polled=0, qualification=None):
    from native_overlap_metrics import clock_identity

    runner = getattr(worker, 'model_runner', None)
    state, storage_epoch, incarnation = _state_identity(native)
    controller = native.controller
    router = getattr(native, 'router', None)
    rows = _request_rows(runner)
    census_row = live_census(runner, census)
    if census_row is None:
        unavailable = list(unavailable) + ['live-census-unavailable']
        census_row = {'available': False, 'live_requests': 0,
                      'prefill_requests': 0, 'decode_eligible': 0,
                      'live_request_uids': []}
    engine_epoch = command.get('engine_epoch')
    bound = getattr(state, 'engine_bound', False)
    epoch_ok = (not bound) or (getattr(state, 'engine_epoch', None) == engine_epoch)
    identity = clock_identity()
    unavailable = sorted(set(unavailable)
                         | set(identity.get('unavailable') or []))
    if not epoch_ok:
        unavailable = sorted(set(unavailable) | {'engine-epoch-differs'})
    if census_row['available'] is False:
        unavailable = sorted(set(unavailable) | {'live-decode-census-unavailable'})
    pending_replays = getattr(router, 'native_pending', 0) if router is not None else 0
    if type(pending_replays) is not int or pending_replays < 0:
        raise NativeOverlapUnavailable('native-overlap-replay-census-unavailable')
    prefill_streams = getattr(native, 'prefill_streams', None)
    pending_prefill = int(bool(getattr(prefill_streams, '_in_scope', False)))
    pending_output = int(getattr(runner, 'execute_model_state', None) is not None)
    handler = getattr(runner, 'pp_handler', None)
    pp_streams = sum(1 for name in ('main_stream', 'broadcast_stream')
                     if handler is not None
                     and getattr(handler, name, None) is not None)
    bridge_row = _bridge_rows(runner)
    batch = (census.last_batch_req_ids if census is not None else frozenset())
    # The last dispatched batch may be a partial subset (or empty); it is one
    # observation of stock scheduling, never the definition of a retained row.
    active_rows = sum(1 for request in batch if rows is not None and request in rows)
    clean = (bridge_row['retained_rows'] == 0 and pending_replays == 0
             and pending_prefill == 0 and pending_output == 0
             and drained_streams > 0)
    if bridge_row['active']:
        unavailable = sorted(set(unavailable) | {'bridge-transaction-open'})
    if bridge_row['retained_uids']:
        unavailable = sorted(set(unavailable) | {'bridge-retained-row'})
    rank = getattr(worker, 'rank', None)
    if type(rank) is not int:
        rank = getattr(state, 'worker_rank', None)
    if type(rank) is not int:
        raise NativeOverlapUnavailable('native-overlap-worker-rank-unavailable')
    return {
        'schema': SCHEMA,
        'action': action,
        'worker_rank': int(rank),
        'pid': int(os.getpid()),
        'engine_epoch': (engine_epoch if type(engine_epoch) is int
                         else int(getattr(state, 'engine_epoch', 0))),
        'route_generation': int(controller.route_generation),
        'storage_epoch': int(storage_epoch),
        'observed_clock_ns': int(clock()),
        'decode_replay_watermark': int(getattr(router, 'replay_watermark', 0)
                                       if router is not None else 0),
        'live_decode_requests': int(census_row['live_requests']),
        'live_batch_rows': int(active_rows),
        'running_requests': int(census_row['live_requests']),
        'decode_eligible': int(census_row['decode_eligible']),
        'prefill_requests': int(census_row['prefill_requests']),
        'retained_rows': int(bridge_row['retained_rows']),
        'retained_active_rows': int(bridge_row['retained_active_rows']),
        'pending_native_replays': int(pending_replays),
        'polled_replays': int(polled),
        'pending_prefill_streams': int(pending_prefill),
        'pending_output_copies': int(pending_output),
        'pp_consumer_streams': int(pp_streams),
        'drained_streams': int(drained_streams),
        'drained_retained_rows': int(drained_retained),
        'drained_native_replays': int(drained_native),
        'route': str(controller.route),
        'worker_incarnation': incarnation,
        'clock_basis': CLOCK_BASIS,
        'engine_epoch_ok': bool(epoch_ok),
        'barrier_clean': bool(clean),
        'retained_request_uids': list(bridge_row['retained_uids']),
        'unavailable': unavailable,
        'allocator': allocator,
        'memory': memory,
        'live_decode': copy.deepcopy(census_row),
        'native_inventory': (_native_inventory(native) if action == 'identity' else None),
        'qualification': (copy.deepcopy(qualification)
                          if qualification is not None else None),
    }


def _native_inventory(native):
    """Queried construction inventory, not a readiness attestation."""
    if not callable(getattr(native, 'inventory', None)):
        return None
    inventory = native.inventory()
    if os.environ.get('FLASH_S5_AUTO_ARM', '0') == '1':
        from native_auto_arm import worker_capability
        inventory['auto_arm_decode'] = worker_capability(native)
    state = native.state
    owner = native.decode_owner
    inventory.update({
        'schema': 's5-native-construction/1',
        'worker_incarnation': state.incarnation,
        'engine_bound': state.engine_bound, 'engine_epoch': state.engine_epoch,
        'worker_rank': state.worker_rank, 'storage_epoch': owner.storage_epoch,
        'decode_owner_incarnation': owner.incarnation,
        'decode_lane': {'context_id': owner.lane_id.context_id,
                        'storage_epoch': owner.storage_epoch,
                        'worker_rank': owner.worker_rank},
        'router': native.router.native_census(),
    })
    return inventory


def _qualification(runner, command):
    """Delegate the exact teacher/observer action; never invent a verdict."""
    try:
        from native_qualification import handle_worker
    except ImportError:
        raise NativeOverlapUnavailable(
            'native-qualification-not-constructed') from None
    if command.get('action') not in QUALIFICATION_ACTIONS:
        raise NativeOverlapUnavailable('native-qualification-action-unknown')
    record = handle_worker(runner, dict(command))
    if type(record) is not dict:
        raise NativeOverlapUnavailable('native-qualification-returned-no-record')
    result = copy.deepcopy(record)
    result['graded_by'] = 'native_qualification.handle_worker'
    result['engine_supplied_verdict'] = False
    return result


def handle(worker, command, *, clock=time.monotonic_ns, torch_module=None):
    """One bounded worker action; every answer is bound to this worker."""
    if type(command) is not dict:
        raise NativeOverlapUnavailable('native-overlap-command-not-object')
    action = command.get('action')
    if action not in ACTIONS and action not in QUALIFICATION_ACTIONS and action not in attn_meta_actions():
        raise NativeOverlapUnavailable('native-overlap-unknown-action')
    runner, native = _worker_native(worker)
    census = getattr(runner, '_flash_s5_native_census', None)
    engine_epoch = command.get('engine_epoch')
    if type(engine_epoch) is not int or engine_epoch < 1:
        raise NativeOverlapUnavailable('native-overlap-invalid-engine-epoch')
    allocator, memory, memory_reason = measure_memory(runner,
                                                      torch_module=torch_module)
    unavailable = [] if memory_reason is None else [memory_reason]
    common = dict(census=census, allocator=allocator, memory=memory,
                  unavailable=unavailable)
    if action in ('identity', 'census', 'status'):
        return _row(worker, native, action, command, clock, **common)
    if action in ('barrier', 'drain'):
        polled, _pending = _poll_replays(native, 32)
        _drained_prefill, drained = _drain(native)
        row = _row(worker, native, action, command, clock,
                   drained_streams=drained, drained_native=polled, **common)
        # ``_row`` already refuses cleanliness for an open transaction or a
        # retained nonterminal row; the marker keeps the reason explicit.
        if _bridge_rows(runner)['active']:
            row['barrier_clean'] = False
        return row
    if action == 'poll_native':
        limit = command.get('limit', 8)
        if type(limit) is not int or not 1 <= limit <= 64:
            raise NativeOverlapUnavailable('native-overlap-poll-limit-invalid')
        polled, pending = _poll_replays(native, limit)
        row = _row(worker, native, action, command, clock, polled=polled, **common)
        row['pending_native_replays'] = pending
        return row
    if action in QUALIFICATION_ACTIONS:
        return _row(worker, native, 'qualification', command, clock,
                    qualification=_qualification(runner, command), **common)
    if action in attn_meta_actions():
        return _row(worker, native, 'qualification', command, clock,
                    qualification=_attn_meta(runner, command), **common)
    raise NativeOverlapUnavailable('native-overlap-action-not-implemented')


INWAIT_RANK_LEVERS = ('allocator_flush', 'slots_fast', 'prepare_slices', 'score_warm', 'score_warm_rank0', 'cow_fast', 'reserved_cap', 'cow_host_filter')    # the flush first: the cap is sized after it


def _attn_meta(runner, command):
    """The attn-meta selector's typed select / read on this rank; counters and wrapper order, never a verdict."""
    action = command.get('action')
    fields = set(command)
    levers = {}
    if action == 'attn_meta_select' and os.environ.get('FLASH_S5_INWAIT', '0') == '1':
        # inwait build-2 and build-3 (DESIGN-2, DESIGN-3): this rank's levers, the same pause as the attn-meta selection.
        for name in INWAIT_RANK_LEVERS:
            if name in fields:
                fields.discard(name)
                levers[name] = command[name]
                if type(levers[name]) is not bool:
                    raise NativeOverlapUnavailable('native-attn-meta-' + name.replace('_', '-') + '-not-bool')
    if fields != _ATTN_META_FIELDS[action]:
        raise NativeOverlapUnavailable('native-attn-meta-fields-differ')
    import attn_meta_select
    try:
        if action == 'attn_meta_select':
            if type(command['on']) is not bool:
                raise NativeOverlapUnavailable('native-attn-meta-select-not-bool')
            record = attn_meta_select.select(runner, command['on'])
        else:
            record = attn_meta_select.read(runner)
    except attn_meta_select.AttnMetaSelectError as error:
        raise NativeOverlapUnavailable('native-' + str(error)) from None
    result = copy.deepcopy(record)
    if os.environ.get('FLASH_S5_INWAIT', '0') == '1':
        # Inside ``after`` and ``counters``: the rows the box's attn-meta reader keeps.
        import lane_chunk
        row = result['after' if action == 'attn_meta_select' else 'counters']
        selects = dict(slots_fast=(lane_chunk.select_slots_fast, lane_chunk.SLOTS_FAST),
                       prepare_slices=(lane_chunk.select_prepare_slices, lane_chunk.PREPARE_SLICES),
                       score_warm=(lane_chunk.select_score_warm, lane_chunk.SCORE_WARM),
                       score_warm_rank0=(lane_chunk.select_score_warm_rank0, lane_chunk.SCORE_WARM_RANK0),
                       allocator_flush=(lane_chunk.select_allocator_flush, lane_chunk.ALLOCATOR_FLUSH),
                       cow_fast=(lane_chunk.select_cow_fast, lane_chunk.COW_FAST),
                       cow_host_filter=(lane_chunk.select_cow_host_filter, lane_chunk.COW_HOST_FILTER),
                       reserved_cap=(lane_chunk.select_reserved_cap, lane_chunk.RESERVED_CAP))
        for name in INWAIT_RANK_LEVERS:
            select, state = selects[name]
            row[name] = select(levers[name]) if name in levers else dict(state)
    result.update(worker_action=action, graded_by='attn_meta_select', engine_supplied_verdict=False)
    return result


__all__ = [
    'ACTIONS', 'ATTN_META_ACTIONS', 'CLOCK_BASIS', 'NativeCensus', 'attn_meta_actions', 'NativeOverlapUnavailable',
    'QUALIFICATION_ACTIONS', 'ROW_BOOL_FIELDS', 'ROW_FIELDS', 'ROW_INT_FIELDS',
    'ROW_LIST_FIELDS', 'ROW_NULLABLE', 'ROW_STR_FIELDS', 'SCHEMA',
    'handle', 'install_census', 'live_census', 'measure_memory',
    'observe_sample', 'observe_scheduler',
]
