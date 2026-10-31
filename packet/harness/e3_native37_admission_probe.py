#!/usr/bin/env python3
"""Native37 T1/T2 in the pinned image: native admission on an armed split route.

Runs against whichever overlay is mounted (the native37 candidate, or held-c as
the negative control) and reports PASS/FAIL per test without retrying.

Shared construction (production geometry, CPU only): the actual pinned
Scheduler built by its own __init__ (generated core/scheduler.py from the
mounted overlay), the real KV-cache configuration path, SchedulerLeases, the
real PrefillAdmissionController and admission-metadata provider exactly as
s5_startup builds them, and a real NativeSegmentDriver installed on the
scheduler. Requests come from the real EngineCore.preprocess_add_request.
Declared leaves (native37_fixture_leaves.json): the reduced model config, the
EngineCore pause query, the executor (never called), the split-route transition
outcome (``_split_ready``/``route_generation``) and the empty sampler output.

T1  residue sweep through the real schedule(): every length either gets a
    native plan or a typed ``no-staged-plan`` refusal whose stop and gap match
    the band rule, and no refused request is stock-scheduled on the armed split
    route.  Includes the W2 lengths (197315 refused, 201306 planned) and row4.
T2  one 197315-token request through the real scheduler lifecycle: held
    unallocated while armed (memoised, counted), typed in status and in the
    overlap evidence prefill_end reads, abort-safe while held, stock-scheduled
    1024 tokens after disarm + FULL; positive control 201306 admitted natively.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import time
import traceback
from types import SimpleNamespace

BLOCK = 4048          # production mamba/attention block
HASH = 176            # production prefix_match_unit -> hash block
BUDGET = 1024         # PREFILL_CHUNK=1024 -> max_num_batched_tokens
CHUNK = 1024          # FLASH_S5_CHUNK_TOKENS default (async_handoff.DEFAULT_CHUNK_TOKENS)
MAX_SEQS = 8          # launcher default MAX_SEQS
MAX_LEN = 262144      # launcher default MAX_MODEL_LEN
MIN_ROWS = 64         # QSA_STAGE_MIN_ROWS
BLOCKS = 4096         # scheduler block-id pool; geometry only, no tensors
EPOCH = 7
W2 = {'fresh': 197315, 'append': 201306}
ROW4 = 4167
SWEEP_SEED, SWEEP_COUNT = 7, 300


def band(n, start=0):
    """The diagnosed refusal rule for a fresh request (start 0).

    The native lane ends at n-1 (one-token handback).  Stock alignment forces
    a stop at the last 4048-aligned position and, with partial mamba cache
    hits (176 < 4048), at the prompt's last 176 boundary.  A tail of 1..63
    tokens after the last stop cannot run the staged attention path.
    """
    assert start == 0
    lcp = n - n % BLOCK
    tail = n // HASH * HASH
    stop = max(lcp, tail if lcp < tail < n else 0)
    gap = (n - 1) - stop
    return {'stop': stop, 'gap': gap, 'refused': 1 <= gap <= MIN_ROWS - 1}


def sweep_lengths():
    rows = [('w2-fresh', W2['fresh']), ('w2-append', W2['append']), ('row4', ROW4),
            ('native15-probe', 98312), ('block-multiple', BLOCK * 48)]
    for k in (12, 48):
        lcp = BLOCK * k
        for g in (0, 1, 2, 62, 63, 64, 65):
            rows.append(('lcp-edge-k%d-g%d' % (k, g), lcp + 1 + g))
            rows.append(('tail-edge-k%d-g%d' % (k, g), lcp + 5 * HASH + 1 + g))
    rng = random.Random(SWEEP_SEED)
    rows += [('random-%d' % i, n) for i, n in enumerate(rng.sample(range(20000, 262000), SWEEP_COUNT))]
    return rows


def production_config():
    from vllm.config import (ModelConfig, VllmConfig, CacheConfig, SchedulerConfig,
                             ParallelConfig, DeviceConfig, CompilationConfig)
    from vllm.models.qwen4_exp.config import Qwen4ExpTextConfig, Qwen4ExpConfig
    path = Path(tempfile.mkdtemp(dir='/tmp', prefix='native37-qwen-'))
    # The reduced Qwen4Exp family of e3_native_transitions_lifecycle_probe.config,
    # with max_position_embeddings raised to the production max_model_len.
    text = Qwen4ExpTextConfig(vocab_size=256, hidden_size=256, intermediate_size=512,
        num_hidden_layers=3, num_attention_heads=12, num_key_value_heads=1, head_dim=256,
        layer_types=['linear_attention', 'full_attention', 'linear_attention'],
        linear_num_key_heads=4, linear_num_value_heads=4, linear_key_head_dim=64, linear_value_head_dim=64,
        num_experts=0, shared_expert_intermediate_size=0, max_position_embeddings=MAX_LEN,
        hc_count=4, hc_lowrank=16, ple_layer_ids=[1], ple_embed_dim=64, heads_per_ngram=4,
        ngram_vocab_size_base=1024, make_ngram_vocab_size_divisible_by=128,
        indexer_n_heads=4, indexer_kv_heads=1, indexer_head_dim=64, indexer_budget=2048, indexer_compress_ratio=4,
        partial_rotary_factor=.25, eos_token_id=255)
    Qwen4ExpConfig(text_config=text, architectures=['Qwen4ExpForConditionalGeneration']).save_pretrained(path)
    model = ModelConfig(model=str(path), skip_tokenizer_init=True, language_model_only=True,
                        dtype='bfloat16', max_model_len=MAX_LEN, enforce_eager=True)
    return VllmConfig(model_config=model,
        cache_config=CacheConfig(block_size=BLOCK, mamba_block_size=BLOCK, num_gpu_blocks_override=BLOCKS,
            gpu_memory_utilization=0.08, enable_prefix_caching=True, prefix_match_unit=HASH,
            mamba_cache_mode='align'),
        scheduler_config=SchedulerConfig(max_model_len=MAX_LEN, is_encoder_decoder=False,
            max_num_batched_tokens=BUDGET, max_num_seqs=MAX_SEQS, async_scheduling=True),
        parallel_config=ParallelConfig(distributed_executor_backend='mp'),
        device_config=DeviceConfig(device='cpu'), compilation_config=CompilationConfig(mode=0))


def production_host():
    # As native18_scheduler_fixture: the CPU platform refuses the hybrid KV
    # manager at VllmConfig validation; the CUDA production engine runs it.
    # The scheduler only reads these fields, so the scheduler config is set
    # after validation (declared leaf, native37_fixture_leaves.json).
    import copy
    from vllm.config import SchedulerConfig
    host = copy.copy(production_config())
    host.scheduler_config = SchedulerConfig(max_model_len=MAX_LEN, is_encoder_decoder=False,
        max_num_batched_tokens=BUDGET, max_num_seqs=MAX_SEQS, async_scheduling=True,
        enable_chunked_prefill=True, disable_hybrid_kv_cache_manager=False)
    return host


class NoExecutor:
    """Leaf for the E3 executor wrapper: T1/T2 never dispatch to workers."""

    def __getattr__(self, name):
        raise AssertionError('native37 probe touched the executor: ' + name)


class Stack:
    """Real scheduler + leases + controller + metadata + driver, s5_startup order."""

    def __init__(self):
        import torch
        from vllm.config import set_current_vllm_config
        from vllm.distributed import init_distributed_environment, initialize_model_parallel
        from vllm.models.qwen4_exp.nvidia.model import Qwen4ExpForConditionalGeneration as Model
        from vllm.v1.kv_cache_interface import MambaSpec, FullAttentionSpec
        from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum as Kind
        from vllm.v1.attention.backends.gdn_attn import GDNAttentionBackend
        from vllm.v1.attention.backends.short_conv_attn import PleShortConvAttentionBackend
        from vllm.models.qwen4_exp.nvidia.qsa import Qwen4ExpQSAFlashAttentionBackend
        from vllm.models.qwen4_exp.common.qsa_cache import QSAKeyStateCache, QSACompressedKeyCache, QSAStateBackend
        from vllm.v1.core.kv_cache_utils import (generate_scheduler_kv_cache_config, resolve_kv_cache_block_sizes,
                                                 update_kv_cache_capacity, init_none_hash, get_request_block_hasher)
        from vllm.v1.structured_output import StructuredOutputManager
        from vllm.utils.hashing import get_hash_fn_by_name
        from vllm.v1.flash_e3 import scheduler_hooks as sh
        from vllm.v1.flash_e3 import admission_metadata as am
        from vllm.v1.flash_e3 import async_handoff as ah
        from vllm.v1.flash_e3 import engine_driver as ed
        from e3_native_transitions_lifecycle_probe import cache_configuration
        self.sh, self.ed = sh, ed
        host = production_host()
        with set_current_vllm_config(host):
            init_distributed_environment(world_size=1, rank=0, local_rank=0,
                                         distributed_init_method='file:///tmp/native37-cpu-dist', backend='gloo')
            initialize_model_parallel(1, 1, backend='gloo')
            gdn, ple = Model.get_mamba_specs_from_config(host)

            def state(s, k):
                return MambaSpec(block_size=BLOCK, shapes=tuple(s.shapes), dtypes=tuple(s.dtypes), mamba_type=k,
                                 mamba_cache_mode='align', tp_replicated=s.tp_replicated)
            specs = {'shape.gdn0': state(gdn, Kind.GDN_ATTN), 'shape.gdn2': state(gdn, Kind.GDN_ATTN),
                     'shape.ple': state(ple, Kind.SHORT_CONV),
                     'shape.qsa': Qwen4ExpQSAFlashAttentionBackend.customize_spec(FullAttentionSpec(block_size=BLOCK,
                         num_kv_heads=1, head_size=256, head_size_v=256, dtype=torch.bfloat16))}
            caches = [QSAKeyStateCache(head_size=64, dtype=torch.bfloat16, cache_config=host.cache_config,
                          prefix='shape.indexer.raw_key_cache', vllm_config=host, compress_ratio=4,
                          cache_rope_positions=host.model_config.uses_mrope),
                      QSACompressedKeyCache(head_size=64, dtype=torch.bfloat16, cache_config=host.cache_config,
                          prefix='shape.indexer.compressed_key_cache', vllm_config=host, compress_ratio=4)]
            for owner in caches:
                specs[owner.prefix] = owner.get_kv_cache_spec(host)
            worker_kv, self.cache_record = cache_configuration(
                host, specs, (GDNAttentionBackend, PleShortConvAttentionBackend,
                              Qwen4ExpQSAFlashAttentionBackend, QSAStateBackend))
            kv = generate_scheduler_kv_cache_config([worker_kv])
            host.cache_config.num_gpu_blocks = kv.num_blocks
            host.cache_config.block_size = min(g.kv_cache_spec.block_size for g in kv.kv_cache_groups
                                               if g.kv_cache_spec.prefix_cacheable)
            update_kv_cache_capacity(host, kv)
            host.validate_block_size()
            block, hash_block = resolve_kv_cache_block_sizes(kv, host)
            Scheduler = host.scheduler_config.get_scheduler_cls()
            self.scheduler = Scheduler(host, kv, StructuredOutputManager(host), block_size=block,
                                       hash_block_size=hash_block)
        S = self.scheduler
        self.host = host
        self.leases = sh.SchedulerLeases().attach(S, S.kv_cache_manager.block_pool)
        hash_fn = get_hash_fn_by_name('sha256')
        init_none_hash(hash_fn)
        self.core = SimpleNamespace(mm_receiver_cache=None,
                                    request_block_hasher=get_request_block_hasher(hash_block, hash_fn),
                                    hash_block_size=hash_block)
        # s5_startup.start_engine order: controller, admission metadata, driver.
        self.controller = sh.PrefillAdmissionController(
            S, leases=self.leases, frontier=ah.CommittedFrontier(), engine_epoch=EPOCH,
            chunk_tokens=CHUNK, ranks=(0, 1, 2, 3), clock=time.monotonic,
            minimum_stage_rows=sh.QSA_STAGE_MIN_ROWS).attach()
        am.install_admission_metadata(S)
        self.paused = False
        # Leaf for EngineCore: the two attributes the driver reads (pause query, scheduler).
        self.engine = SimpleNamespace(is_scheduler_paused=lambda: self.paused, scheduler=S)
        self.driver = ed.NativeSegmentDriver(
            self.controller, executor=NoExecutor(), scheduler=S, engine=self.engine,
            layer_count=48, layer_split=24, stage_of=((0, 0), (1, 0), (2, 1), (3, 1)),
            clock=time.monotonic, reply_timeout_s=300.0, monotonic_ns=time.monotonic_ns).install()
        self.geometry = dict(block_size=block, hash_block_size=hash_block,
            max_num_scheduled_tokens=S.max_num_scheduled_tokens, max_num_running_reqs=S.max_num_running_reqs,
            mamba_partial_cache_hit=getattr(S, 'mamba_partial_cache_hit', None),
            scheduler_reserve_full_isl=getattr(S, 'scheduler_reserve_full_isl', None),
            max_model_len=host.model_config.max_model_len, num_gpu_blocks=kv.num_blocks,
            scheduler_class=type(S).__module__ + '.' + type(S).__name__,
            controller_chunk_tokens=self.controller.chunk_tokens,
            minimum_stage_rows=self.controller.minimum_stage_rows)
        self.salt = 0

    # -- declared leaves ---------------------------------------------------
    def split_route(self, on):
        """Leaf: the all-rank split/FULL TransitionAck outcome (no worker RPC)."""
        self.driver._split_ready = bool(on)
        self.driver.route_generation += 1

    def arm(self, phase='w2-overlap'):
        self.split_route(True)
        self.paused = True
        reply = self.driver.submit_control({'action': 'arm_admission', 'phase': phase})
        self.paused = False
        return reply

    def disarm(self):
        self.paused = True
        reply = self.driver.submit_control({'action': 'disarm_admission'})
        self.paused = False
        self.split_route(False)
        return reply

    # -- real engine paths -------------------------------------------------
    def request(self, name, n):
        from e3_native_cow_lease_probe import engine_request, host_preprocess
        self.salt += 1
        tokens = [(i * 7 + self.salt) % 251 for i in range(n)]
        request = host_preprocess(self.core, engine_request(name, tokens, 'native37-%d' % self.salt, 4))
        self.scheduler.add_request(request)
        return request

    def step(self):
        """One EngineCore step: schedule, then update with the declared empty sampler leaf."""
        from vllm.v1.outputs import ModelRunnerOutput
        started = time.monotonic_ns()
        out = self.scheduler.schedule()
        wall = time.monotonic_ns() - started
        ids = list(out.num_scheduled_tokens)
        self.scheduler.update_from_output(out, ModelRunnerOutput(
            req_ids=ids, req_id_to_index={k: i for i, k in enumerate(ids)},
            sampled_token_ids=[[] for _ in ids]))
        return out, wall

    def abort(self, request):
        from vllm.v1.request import RequestStatus
        self.scheduler.finish_requests([request.request_id], RequestStatus.FINISHED_ABORTED)

    def free_blocks(self):
        return self.scheduler.kv_cache_manager.block_pool.get_num_free_blocks()


def typed(controller):
    """Controller typed-refusal surface, or None on an engine without one (held-c)."""
    if not callable(getattr(controller, 'native_admission_record', None)):
        return None
    return controller.native_admission_record()


def t1(stack):
    rows, failures = [], []
    sh = stack.sh
    stack.arm()
    started = time.monotonic()
    for label, n in sweep_lengths():
        rule = band(n)
        before = typed(stack.controller)
        refused_before = stack.controller.refused
        request = stack.request('t1-' + label, n)
        out, wall = stack.step()
        parked = stack.controller.parked
        admitted = parked is not None and parked.transaction.request_uid == request.request_id
        stock = request.request_id in out.num_scheduled_tokens
        after = typed(stack.controller)
        row = dict(label=label, n=n, rule=rule, admitted=admitted, stock_scheduled_on_split=stock,
                   schedule_wall_ns=wall, refused_delta=stack.controller.refused - refused_before)
        if admitted:
            row['first_chunk'] = parked.transaction.chunk_end - parked.transaction.chunk_start
            back = stack.controller.cancel(parked.transaction, never_submitted=True)
            stack.scheduler.waiting.prepend_request(back)
        elif after is None:
            row['receipt'] = None
        else:
            last = after['last_refusal']
            row['receipt'] = (None if last is None or before is not None and last['seq'] == (before['last_refusal'] or {}).get('seq')
                              else {k: last.get(k) for k in ('reason', 'furthest_boundary', 'gap', 'native_end',
                                                              'queries', 'exhausted', 'memoised')})
        stack.abort(request)
        problems = []
        if stock:
            problems.append('stock-scheduled-new-prefill-on-armed-split-route')
        if admitted == rule['refused']:
            problems.append('band-rule-mismatch')
        if not admitted:
            receipt = row.get('receipt')
            if receipt is None:
                problems.append('refusal-untyped')
            elif (receipt['reason'] != 'no-staged-plan' or receipt['furthest_boundary'] != rule['stop']
                  or receipt['gap'] != rule['gap'] or receipt['native_end'] != n - 1):
                problems.append('refusal-receipt-differs-from-rule')
        row['problems'] = problems
        if problems:
            failures.append(dict(label=label, n=n, problems=problems))
        rows.append(row)
    stack.disarm()
    assert not stack.scheduler.requests, 'sweep left requests in the scheduler'
    w2 = {name: next(dict(n=r['n'], admitted=r['admitted'], stop=r['rule']['stop'], gap=r['rule']['gap'],
                          receipt=r.get('receipt')) for r in rows if r['label'] == 'w2-' + name) for name in W2}
    long_rows = [r for r in rows if r['label'].startswith('random-')]
    return dict(verdict='FAIL' if failures else 'PASS', cases=len(rows), failures=failures[:40],
                failure_count=len(failures),
                refused=sum(not r['admitted'] for r in rows), planned=sum(r['admitted'] for r in rows),
                random_refused=sum(not r['admitted'] for r in long_rows), random_cases=len(long_rows),
                stock_scheduled_on_split=sum(r['stock_scheduled_on_split'] for r in rows),
                untyped_refusals=sum('refusal-untyped' in r['problems'] for r in rows),
                rule_mismatches=sum('band-rule-mismatch' in r['problems'] for r in rows),
                w2_plannability=w2, wall_s=round(time.monotonic() - started, 3), rows=rows)


def t2(stack):
    sh = stack.sh
    checks = []

    def check(name, ok, **detail):
        checks.append(dict(name=name, verdict='PASS' if ok else 'FAIL', **detail))
        if not ok:
            raise _Stop(name)

    record = dict(checks=checks)
    free0 = stack.free_blocks()
    S, controller, driver = stack.scheduler, stack.controller, stack.driver
    try:
        stack.arm()
        begin = driver.submit_control({'action': 'status'})
        base = typed(controller)  # engine-lifetime counts: T2 reads deltas from here
        start_ns = time.monotonic_ns()
        a = stack.request('t2-fresh', W2['fresh'])
        walls, deferrals = [], []
        for step in range(3):
            out, wall = stack.step()
            walls.append(wall)
            check('held-unscheduled-step-%d' % step, a.request_id not in out.num_scheduled_tokens,
                  scheduled=dict(out.num_scheduled_tokens),
                  new=[r.req_id for r in out.scheduled_new_reqs])
        queued = a in list(S.waiting) or a in list(getattr(S, 'skipped_waiting', ()))
        check('held-unallocated', a.num_computed_tokens == 0 and stack.free_blocks() == free0
              and queued and controller.parked is None,
              computed=a.num_computed_tokens, free=stack.free_blocks(), free0=free0)
        rec = typed(controller)
        check('typed-record-exported', rec is not None and driver._status_record().get('native_admission') == rec)
        last = rec['last_refusal']
        rule = band(W2['fresh'])
        check('typed-no-staged-plan-at-the-forced-stop',
              last['reason'] == 'no-staged-plan' and last['furthest_boundary'] == rule['stop'] == 197296
              and last['gap'] == rule['gap'] == 18 and last['native_end'] == 197314 and last['start'] == 0
              and last['cap'] == 1024 and last['minimum_rows'] == 64 and last['memoised'] is True,
              last_refusal=last)
        delta = {k: rec[k] - base[k] for k in ('plan_searches', 'plan_memo_hits', 'refused',
                                               'split_stock_deferrals', 'admitted', 'refusal_seq')}
        check('memo-one-search-per-request', delta == dict(plan_searches=1, plan_memo_hits=2, refused=3,
                                                           split_stock_deferrals=3, admitted=0, refusal_seq=3),
              delta=delta, schedule_wall_ns=walls)
        # Abort while held: the real EngineCore abort path.
        b = stack.request('t2-fresh-abort', W2['fresh'])
        out, _ = stack.step()
        check('second-request-held', b.request_id not in out.num_scheduled_tokens
              and a.request_id not in out.num_scheduled_tokens)
        stack.abort(b)
        check('abort-while-held', b.request_id not in S.requests and b not in list(S.waiting)
              and b not in list(getattr(S, 'skipped_waiting', ()))
              and b.is_finished() and stack.free_blocks() == free0 and not stack.leases.held_blocks(),
              status=str(b.status))
        # prefill_end's own engine reads, in the session's order: the overlap
        # evidence (a refused request journals no native work, so the driver
        # refuses the empty window), then the queried status record.
        end_ns = time.monotonic_ns()
        try:
            driver.submit_control({'action': 'overlap_evidence', 'start_ns': start_ns, 'end_ns': end_ns})
        except stack.ed.DriverError as exc:
            evidence_error = str(exc)
        else:
            evidence_error = None
        status = driver.submit_control({'action': 'status'})
        refusal = _session().native_admission_refusal(begin.get('native_admission'),
                                                      status.get('native_admission'))
        check('prefill-end-typed-refusal', evidence_error == 'overlap-journal-empty-window'
              and type(refusal) is dict and refusal['reason'] == 'native-admission-refused:no-staged-plan'
              and refusal['deltas']['refused'] == 4 and refusal['deltas']['admitted'] == 0
              # A held on 4 steps, B on 1 (A's memo answers native admission there)
              and refusal['deltas']['split_stock_deferrals'] == 5 and refusal['deltas']['plan_searches'] == 1,
              evidence_error=evidence_error, refusal=refusal)
        # Disarm + FULL: stock owns the request again.
        deferred = typed(controller)['split_stock_deferrals']
        stack.disarm()
        out, _ = stack.step()
        check('stock-after-disarm', out.num_scheduled_tokens.get(a.request_id) == BUDGET
              and a.request_id in [r.req_id for r in out.scheduled_new_reqs]
              and typed(controller)['split_stock_deferrals'] == deferred,
              scheduled=dict(out.num_scheduled_tokens))
        stack.abort(a)
        check('stock-request-retired', not S.requests and stack.free_blocks() == free0)
        # Positive control: a plannable W2 length is admitted natively while armed.
        stack.arm()
        refused, admitted = controller.refused, typed(controller)['admitted']
        c = stack.request('t2-append', W2['append'])
        out, _ = stack.step()
        parked = controller.parked
        check('positive-control-admitted', parked is not None and parked.transaction.request_uid == c.request_id
              and c.request_id not in out.num_scheduled_tokens and controller.refused == refused
              and typed(controller)['admitted'] == admitted + 1,
              chunk=None if parked is None else [parked.transaction.chunk_start, parked.transaction.chunk_end])
        back = controller.cancel(parked.transaction, never_submitted=True)
        S.waiting.prepend_request(back)
        stack.abort(c)
        stack.disarm()
        check('clean', not S.requests and stack.free_blocks() == free0 and not stack.leases.held_blocks()
              and controller.parked is None and driver.admission_intent() is None)
        record['verdict'] = 'PASS'
    except _Stop as stop:
        record.update(verdict='FAIL', first_failure=str(stop))
    finally:
        _retire_all(stack)
    return record


class _Stop(Exception):
    pass


def _session():
    from vllm.v1.flash_e3 import native_overlap_session
    return native_overlap_session


def _retire_all(stack):
    """Leave nothing behind after a FAIL: retire parked work, abort requests, disarm."""
    try:
        parked = stack.controller.parked
        if parked is not None:
            back = stack.controller.cancel(parked.transaction, never_submitted=True)
            stack.scheduler.waiting.prepend_request(back)
        for request in list(stack.scheduler.requests.values()):
            stack.abort(request)
        if stack.driver.admission_intent() is not None:
            stack.disarm()
    except Exception as exc:  # noqa: BLE001 - recorded by the caller's verdict
        print('native37 probe cleanup:', type(exc).__name__, exc, file=sys.stderr)


def mounted():
    from vllm.v1.core.sched import scheduler as core_scheduler
    from vllm.v1.flash_e3 import scheduler_hooks, engine_driver, native_overlap_session
    rows = {}
    for name, module in (('core/scheduler.py', core_scheduler), ('e3/scheduler_hooks.py', scheduler_hooks),
                         ('e3/engine_driver.py', engine_driver),
                         ('e3/native_overlap_session.py', native_overlap_session)):
        path = Path(inspect.getfile(module))
        rows[name] = dict(origin=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--label', required=True)
    a = p.parse_args()
    record = dict(schema='native37-admission-probe/1', label=a.label, verdict='INCONCLUSIVE', rc=2,
                  gpu_execution=False, box_evidence=False)
    try:
        import torch
        record['mounted'] = mounted()
        stack = Stack()
        record['geometry'] = stack.geometry
        record['cache_configuration'] = stack.cache_record
        record['t1'] = t1(stack)
        record['t2'] = t2(stack)
        assert not torch.cuda.is_initialized()
        record['cuda_initialized'] = False
        failed = [name for name in ('t1', 't2') if record[name]['verdict'] != 'PASS']
        record.update(verdict='FAIL' if failed else 'PASS', rc=1 if failed else 0, failed=failed)
    except Exception as exc:  # noqa: BLE001 - classified, never retried
        record.update(verdict='FAIL', rc=1, failure_class='HARNESS-OR-ENGINE-UNCLASSIFIED',
                      error_type=type(exc).__name__, error=str(exc)[:2000],
                      traceback=traceback.format_exc()[-6000:])
    record['probe_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with a.out.open('x') as f:
        json.dump(record, f, indent=2, sort_keys=True)
        f.write('\n')
    summary = {k: record.get(k) for k in ('label', 'verdict', 'rc', 'failed', 'error_type', 'error')}
    for name in ('t1', 't2'):
        if name in record:
            view = record[name]
            summary[name] = {k: view.get(k) for k in ('verdict', 'cases', 'failure_count', 'refused', 'planned',
                                                      'stock_scheduled_on_split', 'untyped_refusals',
                                                      'rule_mismatches', 'first_failure')}
    print(json.dumps(summary, sort_keys=True))
    raise SystemExit(record['rc'])


if __name__ == '__main__':
    main()
