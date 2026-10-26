"""Bounded, authenticated decode-only teacher-forcing instrument.

The global vLLM trace-replay mode stays off. A private controller plan injects
trace IDs AFTER normal request validation and BEFORE engine serialization.
The stock sampler applies those IDs after sampling and before computing top-k
logprobs from the original logits. No logits/weights/kernel settings are changed.
Tokens exist only in private request state; public records contain hashes/counts.
"""
import hashlib
import json

MEASURED_STEPS = 256
ENTRY_STEPS = 256
TAIL_STEPS = 256
STEPS = ENTRY_STEPS + MEASURED_STEPS + TAIL_STEPS
MAX_COHORT = 8
SCHEMA = 's5-teacher-decode/1'


class TeacherDecodeError(RuntimeError):
    pass


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':'), sort_keys=True).encode()).hexdigest()


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
            self.proofs = []
            self.proof_overflow = False
            self.probe = None

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

        def apply_trace(self, sampled, idx_mapping):
            # Deliberately synchronous *diagnostic* readback. These are real
            # sampler input positions before post_update changes total_len.
            # No model/logit buffer is changed; no client clock inference.
            slots = idx_mapping.cpu().tolist()
            total = self.req_states.total_len.gpu[idx_mapping].cpu().tolist()
            prompt = self.req_states.prompt_len.gpu[idx_mapping].cpu().tolist()
            super().apply_trace(sampled, idx_mapping)
            forced = sampled.cpu().tolist()
            batch = dict(self.probe.batch_observations.get('last') or {})
            for slot, end, begin, token in zip(slots, total, prompt, forced):
                if slot < 0 or slot not in self.traces:
                    continue
                position = end - begin
                if not ENTRY_STEPS <= position < ENTRY_STEPS + MEASURED_STEPS:
                    continue
                trace_sha, trace = self.traces[slot]
                if len(self.proofs) >= MAX_COHORT * MEASURED_STEPS:
                    self.proof_overflow = True
                    continue
                self.proofs.append(dict(trace_sha256=trace_sha, position=position,
                    worker_route=self.probe.route, route_generation=self.probe.route_generation,
                    full_graph=batch.get('full_graph'), decode_only=batch.get('decode_only'),
                    forced_token_exact=type(token) is int and token == trace[position]))
    return BoundedTraceReplayState(req_states)


def configure_worker(runner, enabled, *, factory=make_trace_state):
    """Caller must hold a drained, empty, full-route worker boundary."""
    _require(type(enabled) is bool, 'teacher-enable-invalid')
    _require(type(getattr(runner, 'is_last_pp_rank', None)) is bool, 'teacher-pp-role-unavailable')
    if not runner.is_last_pp_rank:
        _require(getattr(runner, 'sampler', None) is None, 'teacher-unexpected-pp-sampler')
        return {'enabled': enabled, 'sampler_owner': False, 'table_bytes': 0}
    sampler = getattr(runner, 'sampler', None)
    _require(sampler is not None and getattr(sampler, 'logprobs_mode', None) == 'raw_logprobs',
             'teacher-needs-raw-logprobs-sampler')
    prior = getattr(runner, '_flash_s5_teacher_trace', None)
    _require(sampler.trace_replay_state is prior, 'teacher-sampler-ownership-differs')
    if enabled and prior is None:
        prior = factory(runner.req_states)
        runner._flash_s5_teacher_trace = prior
        sampler.trace_replay_state = prior
        prior.probe = runner._flash_s5_decode_probe
    elif enabled:
        prior.proofs = []
        prior.proof_overflow = False
        prior.traces = {}
    elif not enabled:
        sampler.trace_replay_state = None
        runner._flash_s5_teacher_trace = None
    return {'enabled': enabled, 'sampler_owner': True,
            'table_bytes': runner.req_states.max_num_reqs * (STEPS + 1) * 4 if enabled else 0}


def worker_evidence(runner):
    state = getattr(runner, '_flash_s5_teacher_trace', None)
    if not getattr(runner, 'is_last_pp_rank', False):
        return {'sampler_owner': False, 'positions': [], 'overflow': False}
    _require(state is not None, 'teacher-sampler-evidence-unavailable')
    return {'sampler_owner': True, 'positions': [dict(row) for row in state.proofs],
            'overflow': state.proof_overflow, 'basis': 'sampler GPU positions and forced-token readback'}
