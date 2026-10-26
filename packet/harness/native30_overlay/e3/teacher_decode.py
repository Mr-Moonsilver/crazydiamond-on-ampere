"""Bounded, authenticated decode-only teacher-forcing instrument.

The global vLLM trace-replay mode stays off. A private controller plan injects
trace IDs AFTER normal request validation and BEFORE engine serialization.
The stock sampler applies those IDs after sampling and before computing top-k
logprobs from the original logits. No logits/weights/kernel settings are changed.
Tokens exist only in private request state; public records contain hashes/counts.

Row evidence is captured at the actual sampler invocation, from the batch that
reached it, before trace application and before post_update. It never retains
logits, hidden states, token payloads or GPU pointers.
"""
import hashlib
import json
import math

MEASURED_STEPS = 256
ENTRY_STEPS = 256
TAIL_STEPS = 256
STEPS = ENTRY_STEPS + MEASURED_STEPS + TAIL_STEPS
MAX_COHORT = 8
SCHEMA = 's5-teacher-decode/1'
EVIDENCE_SCHEMA = 's5-teacher-decode/2'
ROW_SCHEMA = 's5-teacher-rows/1'
ROW_BASIS = 'sampler-invocation/1'
MAX_ROWS = MAX_COHORT * MEASURED_STEPS
MAX_CALLS = MAX_COHORT * MEASURED_STEPS
MAX_INDEX = MAX_COHORT
MAX_CU = MAX_COHORT + 1


class TeacherDecodeError(RuntimeError):
    pass


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':'), sort_keys=True).encode()).hexdigest()


def logprob_digest(trace_sha, position, token_ids, values):
    """Pinned completion serialization: sampled column then top20, dedup in order.

    This digest binds collector data, not model equality. No logits or token
    payload leaves the worker. The API's existing -9999 clamp is reproduced.
    """
    _require(type(token_ids) is list and type(values) is list
             and len(token_ids) == len(values) == 21, 'teacher-logprob-shape')
    top = {}
    for token, value in zip(token_ids, values):
        _require(type(token) is int and token >= 0 and type(value) in (int, float)
                 and not math.isnan(value) and value != float('inf'), 'teacher-logprob-value')
        top['token_id:' + str(token)] = max(float(value), -9999.0)
    _require(20 <= len(top) <= 21, 'teacher-logprob-coverage')
    return digest([trace_sha, position, list(top.items())])


def observe_logprobs(state, result, start):
    """Bind newly measured physical rows to the actual returned sampler data."""
    rows = state.positions[start:]
    if not rows:
        return
    try:
        output = result[0].logprobs_tensors
        ids, values = output.logprob_token_ids, output.logprobs
        shape = tuple(ids.shape)
        _require(len(shape) == 2 and 0 < shape[0] <= MAX_COHORT and shape[1] == 21
                 and tuple(values.shape) == shape, 'teacher-logprob-shape')
        ids, values = ids.cpu().tolist(), values.cpu().tolist()
        for row in rows:
            index = row['batch_row']
            _require(type(index) is int and 0 <= index < len(ids), 'teacher-logprob-row')
            row['logprob_sha256'] = logprob_digest(row['trace_sha256'], row['position'],
                                                 ids[index], values[index])
    except Exception:
        state.capture_defects.add('sampler-logprob-binding-unproved')
        state.capture_overflow = True


def _require(ok, code):
    if not ok:
        raise TeacherDecodeError(code)


def validate_plan(plan):
    _require(type(plan) is dict and set(plan) in ({'schema', 'requests'}, {'schema','requests','snapshot_max_bytes'})
             and plan['schema'] == SCHEMA, 'invalid-teacher-plan')
    cap=plan.get('snapshot_max_bytes',2*1024**3)
    _require(type(cap) is int and 64*1024**2<=cap<=2*1024**3, 'invalid-snapshot-budget')
    rows = plan['requests']
    _require(type(rows) is list and len(rows) == MAX_COHORT, 'invalid-teacher-cohort')
    found = {}
    traces = set()
    for row in rows:
        _require(type(row) is dict and set(row) == {'request_id', 'prompt_sha256', 'continuation'},
                 'invalid-teacher-row')
        name, sha, tokens = row['request_id'], row['prompt_sha256'], row['continuation']
        _require(type(name) is str and 1 <= len(name) <= 96
                 and all(c.isalnum() or c in '-_' for c in name), 'invalid-teacher-request-label')
        _require(type(sha) is str and len(sha) == 64
                 and all(c in '0123456789abcdef' for c in sha), 'invalid-teacher-prompt-digest')
        _require(type(tokens) is list and len(tokens) == STEPS
                 and all(type(t) is int and 0 <= t < 2**31 for t in tokens), 'invalid-teacher-trace')
        # Pinned completion serving creates precisely this external request ID
        # for one prompt, n=1. AsyncLLM randomizes it only AFTER this hook.
        external = 'cmpl-' + name + '-0'
        _require(external not in found, 'duplicate-teacher-request')
        trace_sha = digest(tokens)
        _require(trace_sha not in traces, 'teacher-traces-must-identify-streams')
        traces.add(trace_sha)
        found[external] = (sha, tuple(tokens))
    return found


class FrontendPlan:
    """One cohort, no retry/picking or admission by arrival order."""
    def __init__(self, processor):
        original = getattr(processor, 'process_inputs', None)
        _require(callable(original), 'teacher-input-seam-unavailable')
        _require(getattr(processor, '_flash_s5_teacher', None) is None, 'teacher-already-installed')
        self.processor, self.pending, self.plan_sha = processor, None, None
        self.admitted = 0

        def process(*args, **kwargs):
            # Public trace fields still encounter stock validation with the
            # unchanged enable_trace_replay=False configuration.
            request = original(*args, **kwargs)
            return self.inject(request)
        processor.process_inputs = process
        original_async = getattr(processor, 'process_inputs_async', None)
        if callable(original_async):
            # The pinned constructor binds the OLD sync method into make_async.
            # Wrapping only process_inputs would miss the raw-prompt path.
            async def process_async(*args, **kwargs):
                return self.inject(await original_async(*args, **kwargs))
            processor.process_inputs_async = process_async
        processor._flash_s5_teacher = self

    def configure(self, plan):
        _require(self.pending is None or not self.pending, 'teacher-cohort-not-admitted')
        self.pending = None if plan is None else validate_plan(plan)
        self.plan_sha = None if plan is None else digest(plan)
        self.admitted = 0
        return self.status()

    def status(self):
        return {'schema': SCHEMA, 'enabled': self.pending is not None,
                'plan_sha256': self.plan_sha, 'admitted': self.admitted,
                'pending': len(self.pending) if self.pending is not None else 0,
                'steps': STEPS, 'entry_steps': ENTRY_STEPS,
                'measured_steps': MEASURED_STEPS, 'tail_steps': TAIL_STEPS}

    def inject(self, request):
        if self.pending is None:
            return request
        entry = self.pending.get(request.request_id)
        _require(entry is not None, 'teacher-request-not-declared')
        sha, trace = entry
        prompt = getattr(request, 'prompt_token_ids', None)
        params = getattr(request, 'sampling_params', None)
        _require(type(prompt) is list and all(type(t) is int and t >= 0 for t in prompt)
                 and digest(prompt) == sha, 'teacher-prompt-differs')
        _require(params is not None and getattr(params, 'trace_decode_token_ids', None) is None,
                 'teacher-trace-already-set')
        _require(getattr(params, 'n', None) == 1 and getattr(params, 'max_tokens', None) == STEPS
                 and getattr(params, 'min_tokens', None) == STEPS
                 and getattr(params, 'temperature', None) == 0
                 and getattr(params, 'ignore_eos', None) is True
                 and getattr(params, 'logprobs', None) == 20
                 and not getattr(params, 'stop', None)
                 and not getattr(params, 'stop_token_ids', None), 'teacher-request-settings-differ')
        _require(getattr(request, 'lora_request', None) is None
                 and not getattr(request, 'mm_features', None), 'teacher-request-mode-unsupported')
        # Reuse the source's exact compatibility and vocabulary validation,
        # without the global trace-enable check or its normalizing mutations.
        params.trace_decode_token_ids = list(trace)
        try:
            params._validate_trace_replay(self.processor.model_config,
                                          self.processor.speculative_config)
        except BaseException:
            params.trace_decode_token_ids = None
            raise TeacherDecodeError('teacher-trace-compatibility-refused') from None
        del self.pending[request.request_id]
        self.admitted += 1
        return request


def _bounded(value, limit, allow_bool=False):
    """Return (items, overflow). Oversize/unreadable values keep a digest+length only."""
    try:
        items = value.cpu().tolist() if hasattr(value, 'cpu') else list(value)
    except (TypeError, ValueError, AttributeError):
        return None, True
    if type(items) is not list or len(items) > limit:
        payload = {'overflow': True, 'length': len(items) if type(items) is list else -1}
        if type(items) is list:
            payload['sha256'] = digest([str(item) for item in items])
        return payload, True
    kind = bool if allow_bool else int
    if any(type(item) is not kind for item in items):
        return None, True
    return items, False


def _snapshot(batch):
    """Bounded CPU mirror of one InputBatch; returns (fields, overflow)."""
    if batch is None:
        return None, True
    fields, overflow = {}, False
    for name, limit, allow_bool in (
            ('idx_mapping', MAX_INDEX, False), ('expanded_idx_mapping', MAX_INDEX, False),
            ('logits_indices', MAX_INDEX, False), ('cu_num_logits', MAX_CU, False),
            ('expanded_local_pos', MAX_INDEX, False), ('num_scheduled_tokens', MAX_INDEX, False),
            ('is_prefilling_np', MAX_INDEX, True)):
        items, bad = _bounded(getattr(batch, name, None), limit, allow_bool)
        fields[name], overflow = items, overflow or bad
    for name in ('num_reqs', 'num_reqs_after_padding', 'num_tokens',
                 'num_tokens_after_padding', 'num_draft_tokens'):
        value = getattr(batch, name, None)
        fields[name] = value if type(value) is int else None
        overflow = overflow or type(value) is not int
    # The sampler's own selection: positions[logits_indices] is the input
    # position of every logit it scores (one extra bounded GPU readback).
    try:
        selected = batch.positions[batch.logits_indices]
    except (AttributeError, IndexError, TypeError):
        selected = None
    items, bad = _bounded(selected, MAX_INDEX)
    fields['logit_input_positions'] = items
    return fields, overflow or bad


def _cohort_order(idx_mapping, traces):
    """Ordered cohort membership as trace digests; None marks a non-cohort row."""
    if type(idx_mapping) is not list:
        return None
    return [traces[slot][0] if type(slot) is int and slot in traces else None
            for slot in idx_mapping]


def _optional_int(value):
    return int(value) if type(value) is int else None


def _descriptor(descriptor):
    """Plain summary of the selected BatchExecutionDescriptor (no object retained)."""
    if descriptor is None:
        return None
    mode = getattr(descriptor, 'cg_mode', None)
    name = getattr(mode, 'name', None)
    return {'cg_mode': name.lower() if type(name) is str else None,
            'cg_mode_index': _optional_int(getattr(mode, 'value', None)),
            'num_tokens': _optional_int(getattr(descriptor, 'num_tokens', None)),
            'num_reqs': _optional_int(getattr(descriptor, 'num_reqs', None)),
            'uniform_token_count': _optional_int(getattr(descriptor, 'uniform_token_count', None)),
            'max_query_len': _optional_int(getattr(descriptor, 'max_query_len', None)),
            'num_active_loras': _optional_int(getattr(descriptor, 'num_active_loras', None)),
            'num_ubatches': _optional_int(getattr(descriptor, 'num_ubatches', None))}


def _descriptor_matches(graph, snapshot):
    if type(graph) is not dict or type(snapshot) is not dict:
        return False
    padded = snapshot.get('num_tokens_after_padding')
    reqs = snapshot.get('num_reqs_after_padding')
    return (type(graph.get('num_tokens')) is int and graph['num_tokens'] == padded
            and (graph.get('num_reqs') is None or graph['num_reqs'] == reqs))


def _shadow(owner, name, wrapper):
    """Install a callable shadow and remember whether the instance owned it."""
    original = getattr(owner, name)
    entry = {'owner': owner, 'name': name, 'original': original, 'wrapper': wrapper,
             'had_instance_attr': name in getattr(owner, '__dict__', {})}
    setattr(owner, name, wrapper)
    return entry


def _unshadow(entry):
    """Exact inverse of _shadow: a class method is unshadowed, not rebound."""
    owner, name = entry['owner'], entry['name']
    _require(getattr(owner, name) is entry['wrapper'], 'teacher-observer-drift')
    if entry['had_instance_attr']:
        setattr(owner, name, entry['original'])
    else:
        delattr(owner, name)


def _install_observer(runner, state):
    """Reversible runtime wrappers; exact inverse is _remove_observer."""
    _require(getattr(runner, '_flash_s5_teacher_observer', None) is None,
             'teacher-observer-already-installed')
    sample = getattr(runner, 'sample', None)
    _require(callable(sample), 'teacher-sampler-call-unavailable')
    sharder = getattr(runner, 'batch_sharder', None)
    shard = getattr(sharder, 'shard_sampler_inputs', None) if sharder is not None else None
    manager = getattr(runner, 'cudagraph_manager', None)
    dispatch = getattr(manager, 'dispatch', None) if manager is not None else None

    def sample_wrapper(hidden_states, input_batch, grammar_output):
        state.call_depth += 1
        outer = state.call_depth == 1
        if not outer:
            state.capture_defects.add('nested-sampler-call')
            state.capture_overflow = True
        else:
            if shard is not None:
                state.capture_defects.add('batch-sharded-sampling-unqualified')
                state.capture_overflow = True
            state.call_global = input_batch
            state.call_local = None if shard is not None else input_batch
            state.call_graph = state.last_graph
            state.call_graph_stale = state.dispatch_seq == state.consumed_dispatch_seq
            state.consumed_dispatch_seq = state.dispatch_seq
        start = len(getattr(state, 'positions', ()))
        try:
            result = sample(hidden_states, input_batch, grammar_output)
            if outer and hasattr(state, 'positions'):
                observe_logprobs(state, result, start)
            return result
        finally:
            state.call_depth -= 1
            if state.call_depth == 0:
                state.call_global = None
                state.call_local = None
                state.call_graph = None
                state.call_graph_stale = False

    def shard_wrapper(input_batch, grammar_output):
        result = shard(input_batch, grammar_output)
        state.call_local = result[0]
        return result

    def dispatch_wrapper(*args, **kwargs):
        result = dispatch(*args, **kwargs)
        state.last_graph = _descriptor(result)
        state.dispatch_seq += 1
        return result

    shadows = [_shadow(runner, 'sample', sample_wrapper)]
    if shard is not None:
        shadows.append(_shadow(sharder, 'shard_sampler_inputs', shard_wrapper))
    if dispatch is not None:
        shadows.append(_shadow(manager, 'dispatch', dispatch_wrapper))
    runner._flash_s5_teacher_observer = {'state': state, 'shadows': shadows}


def _remove_observer(runner):
    owner = getattr(runner, '_flash_s5_teacher_observer', None)
    _require(owner is not None, 'teacher-observer-not-installed')
    for entry in reversed(owner['shadows']):
        _unshadow(entry)
    del runner._flash_s5_teacher_observer


def _reset_state(state):
    state.traces = {}
    state.calls = []
    state.positions = []
    state.sampler_calls = 0
    state.call_overflow = False
    state.proof_overflow = False
    state.list_overflow = False
    state.capture_overflow = False
    state.capture_defects = set()
    state.last_graph = None
    state.dispatch_seq = 0
    state.consumed_dispatch_seq = -1


def capture_call(*, seq, slots, total, prompt, traces, local, global_batch, graph,
                 graph_stale, identity, calls_used, rows_used):
    """Capture one measured sampler call before trace application.

    Pure and bounded: the caller supplies the sampler's own batch objects and
    the probe identity. Returns None when this call has no measured rows,
    otherwise a dict with any of 'call', 'rows', 'row_traces' plus defects and
    overflow flags. Nothing here retains a tensor or GPU pointer.
    """
    selected = []
    for row, (slot, end, begin) in enumerate(zip(slots, total, prompt)):
        if type(slot) is not int or slot < 0 or slot not in traces:
            continue
        if type(end) is not int or type(begin) is not int:
            continue
        position = end - begin
        if not ENTRY_STEPS <= position < ENTRY_STEPS + MEASURED_STEPS:
            continue
        selected.append((row, slot, position, end, begin, traces[slot]))
    if not selected:
        return None
    defects = set()
    flags = {'call_overflow': False, 'rows_overflow': False, 'list_overflow': False}
    if calls_used >= MAX_CALLS:
        flags['call_overflow'] = True
        return dict(flags, defects=defects)
    if rows_used + len(selected) > MAX_ROWS:
        flags['rows_overflow'] = True
        return dict(flags, defects=defects)
    if local is None or global_batch is None:
        defects.add('batch-not-captured')
        return dict(flags, defects=defects)
    local_snapshot, local_bad = _snapshot(local)
    global_snapshot, global_bad = _snapshot(global_batch)
    flags['list_overflow'] = bool(local_bad or global_bad)
    if graph is None:
        defects.add('graph-descriptor-missing')
    elif graph_stale:
        defects.add('graph-descriptor-stale')
    matches = _descriptor_matches(graph, global_snapshot)
    if graph is not None and not matches:
        defects.add('graph-descriptor-batch-mismatch')
    rank = _optional_int(identity.get('rank'))
    engine_epoch = _optional_int(identity.get('engine_epoch'))
    route_generation = _optional_int(identity.get('route_generation'))
    storage_epoch = _optional_int(identity.get('storage_epoch'))
    route = identity.get('route')
    if type(rank) is not int:
        defects.add('rank-missing')
    if type(engine_epoch) is not int:
        defects.add('engine-epoch-missing')
    if route not in ('full', 'split'):
        defects.add('route-missing')
    if type(route_generation) is not int:
        defects.add('route-generation-missing')
    if type(storage_epoch) is not int:
        defects.add('storage-epoch-missing')
    call = dict(rank=rank, engine_epoch=engine_epoch, route=route,
                route_generation=route_generation, storage_epoch=storage_epoch,
                sampler_call_seq=seq, graph=dict(graph) if type(graph) is dict else None,
                descriptor_stale=bool(graph_stale), descriptor_matches_batch=matches,
                global_snapshot=global_snapshot, local_snapshot=local_snapshot,
                global_cohort_order=_cohort_order(global_snapshot.get('idx_mapping'), traces),
                local_cohort_order=_cohort_order(local_snapshot.get('idx_mapping'), traces),
                rows=len(selected))
    cu_num_logits = local_snapshot.get('cu_num_logits')
    num_reqs = local_snapshot.get('num_reqs')
    local_idx = local_snapshot.get('idx_mapping')
    input_positions = local_snapshot.get('logit_input_positions')
    local_order = call['local_cohort_order']
    rows, row_traces = [], {}
    for row, slot, position, end, begin, (trace_sha, trace) in selected:
        logits_row = None
        if (type(cu_num_logits) is list and type(num_reqs) is int
                and row < num_reqs and len(cu_num_logits) == num_reqs + 1):
            logits_row = cu_num_logits[row]
        else:
            defects.add('logits-row-unavailable')
        if (type(local_idx) is not list or row >= len(local_idx) or local_idx[row] != slot):
            defects.add('batch-row-slot-mismatch')
        if (type(local_order) is not list or row >= len(local_order)
                or local_order[row] != trace_sha):
            defects.add('batch-row-trace-mismatch')
        input_position = None
        if (type(input_positions) is list and type(logits_row) is int
                and 0 <= logits_row < len(input_positions)):
            input_position = input_positions[logits_row]
        else:
            defects.add('input-position-unavailable')
        rows.append({'sampler_call_seq': seq, 'batch_row': row, 'logits_row': logits_row,
                     'slot': slot, 'trace_sha256': trace_sha, 'position': position,
                     'prompt_len': begin, 'total_len': end,
                     'input_position': input_position, 'forced_token_exact': None})
        row_traces[row] = trace
    return dict(flags, defects=defects, call=call, rows=rows, row_traces=row_traces)


def make_trace_state(req_states):
    """Reuse pinned stock application kernel; allocate the fixed entry/measure/tail columns, not maxlen."""
    import torch
    from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor, UvaBackedTensor
    from vllm.v1.worker.gpu.sample.trace_replay import TraceReplayState
    class BoundedTraceReplayState(TraceReplayState):
        def __init__(self, states):
            self.max_num_reqs, self.device, self.req_states = (
                states.max_num_reqs, states.device, states)
            self.trace_token_ids = StagedWriteTensor(
                (states.max_num_reqs, STEPS), dtype=torch.int32,
                device=states.device, uva_instead_of_gpu=True)
            self.trace_len = UvaBackedTensor(states.max_num_reqs, dtype=torch.int32)
            self.traces = {}
            self.calls = []
            self.positions = []
            self.sampler_calls = 0
            self.proof_overflow = False
            self.call_overflow = False
            self.list_overflow = False
            self.capture_overflow = False
            self.capture_defects = set()
            self.rank = None
            self.probe = None
            self.call_global = None
            self.call_local = None
            self.call_graph = None
            self.call_graph_stale = False
            self.last_graph = None
            self.dispatch_seq = 0
            self.consumed_dispatch_seq = -1
            self.call_depth = 0

        def add_request(self, slot, params):
            trace = params.trace_decode_token_ids
            _require(trace is None or (type(trace) is list and len(trace) == STEPS
                     and all(type(t) is int and 0 <= t < 2**31 for t in trace)),
                     'teacher-worker-trace-invalid')
            if trace is None:
                self.traces.pop(slot, None)
            else:
                self.traces[slot] = (digest(trace), tuple(trace))
            super().add_request(slot, params)

        def _begin(self, seq, slots, total, prompt):
            local = self.call_local if self.call_local is not None else self.call_global
            probe = self.probe
            identity = {'rank': self.rank,
                        'engine_epoch': _optional_int(getattr(probe, '_rpc_engine_epoch', None)),
                        'route': getattr(probe, 'route', None),
                        'route_generation': _optional_int(getattr(probe, 'route_generation', None)),
                        'storage_epoch': _optional_int(getattr(probe, 'storage_epoch', None))}
            captured = capture_call(
                seq=seq, slots=slots, total=total, prompt=prompt, traces=self.traces,
                local=local, global_batch=self.call_global, graph=self.call_graph,
                graph_stale=self.call_graph_stale, identity=identity,
                calls_used=len(self.calls), rows_used=len(self.positions))
            if captured is None:
                return None
            self.capture_defects.update(captured['defects'])
            if captured['defects']:
                self.capture_overflow = True
            if captured['list_overflow']:
                self.list_overflow = True
            if captured['call_overflow'] or captured['rows_overflow']:
                self.call_overflow = True
                self.proof_overflow = True
                return None
            if 'call' not in captured:
                return None
            self.calls.append(captured['call'])
            return captured['rows'], captured['row_traces']

        def _finish(self, pending, forced):
            rows, traces = pending
            if type(forced) is not list:
                self.capture_defects.add('forced-readback-failed')
                self.capture_overflow = True
                return
            for row in rows:
                batch_row = row['batch_row']
                token = forced[batch_row] if 0 <= batch_row < len(forced) else None
                trace = traces.get(batch_row)
                row['forced_token_exact'] = (
                    type(token) is int and type(trace) is tuple
                    and 0 <= row['position'] < len(trace)
                    and token == trace[row['position']])
            self.positions.extend(rows)

        def apply_trace(self, sampled, idx_mapping):
            # Synchronous diagnostic readback of the actual sampler batch,
            # before trace application; no model/logit buffer is changed.
            self.sampler_calls += 1
            seq = self.sampler_calls
            pending = None
            try:
                slots = idx_mapping.cpu().tolist()
                total = self.req_states.total_len.gpu[idx_mapping].cpu().tolist()
                prompt = self.req_states.prompt_len.gpu[idx_mapping].cpu().tolist()
                pending = self._begin(seq, slots, total, prompt)
            except BaseException:
                self.capture_defects.add('metadata-readback-failed')
                self.capture_overflow = True
                pending = None
            try:
                super().apply_trace(sampled, idx_mapping)
            finally:
                if pending is not None:
                    try:
                        self._finish(pending, sampled.cpu().tolist())
                    except BaseException:
                        self.capture_defects.add('forced-readback-failed')
                        self.capture_overflow = True
    return BoundedTraceReplayState(req_states)


def configure_worker(runner, enabled, *, rank=None, factory=make_trace_state):
    """Caller must hold a drained, empty, full-route worker boundary."""
    _require(type(enabled) is bool, 'teacher-enable-invalid')
    _require(type(getattr(runner, 'is_last_pp_rank', None)) is bool, 'teacher-pp-role-unavailable')
    if not runner.is_last_pp_rank:
        _require(getattr(runner, 'sampler', None) is None, 'teacher-unexpected-pp-sampler')
        return {'enabled': enabled, 'sampler_owner': False, 'table_bytes': 0,
                'row_schema': None, 'row_basis': None}
    sampler = getattr(runner, 'sampler', None)
    _require(sampler is not None and getattr(sampler, 'logprobs_mode', None) == 'raw_logprobs',
             'teacher-needs-raw-logprobs-sampler')
    prior = getattr(runner, '_flash_s5_teacher_trace', None)
    _require(sampler.trace_replay_state is prior, 'teacher-sampler-ownership-differs')
    if enabled and prior is None:
        prior = factory(runner.req_states)
        prior.rank = rank
        runner._flash_s5_teacher_trace = prior
        sampler.trace_replay_state = prior
        prior.probe = runner._flash_s5_decode_probe
    elif enabled:
        _reset_state(prior)
        prior.rank = rank
    elif not enabled:
        if getattr(runner, '_flash_s5_teacher_observer', None) is not None:
            _remove_observer(runner)
        sampler.trace_replay_state = None
        runner._flash_s5_teacher_trace = None
    if enabled:
        owner = getattr(runner, '_flash_s5_teacher_observer', None)
        if owner is None:
            _install_observer(runner, prior)
        else:
            _require(owner.get('state') is prior, 'teacher-observer-owns-other-state')
    return {'enabled': enabled, 'sampler_owner': True,
            'table_bytes': runner.req_states.max_num_reqs * (STEPS + 1) * 4 if enabled else 0,
            'row_schema': ROW_SCHEMA if enabled else None,
            'row_basis': ROW_BASIS if enabled else None}


def worker_evidence(runner):
    state = getattr(runner, '_flash_s5_teacher_trace', None)
    if not getattr(runner, 'is_last_pp_rank', False):
        return {'sampler_owner': False, 'positions': [], 'calls': [], 'overflow': False,
                'call_overflow': False, 'list_overflow': False, 'capture_defects': [],
                'row_schema': None, 'row_basis': None}
    _require(state is not None, 'teacher-sampler-evidence-unavailable')
    defects = sorted(state.capture_defects)
    return {'sampler_owner': True, 'rank': state.rank, 'row_schema': ROW_SCHEMA,
            'row_basis': ROW_BASIS,
            'calls': [dict(call) for call in state.calls],
            'positions': [dict(row) for row in state.positions],
            'overflow': bool(state.proof_overflow or state.call_overflow
                             or state.list_overflow or state.capture_overflow or defects),
            'call_overflow': bool(state.call_overflow or state.proof_overflow),
            'list_overflow': bool(state.list_overflow), 'capture_defects': defects,
            'basis': 'sampler-invocation readbacks before trace application and state update'}
