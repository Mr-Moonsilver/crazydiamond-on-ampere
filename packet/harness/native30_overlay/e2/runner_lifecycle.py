"""E2 serial runner seam, CPU-only until supplied the event-backed DMA adapter.

This is not a deployable overlay. Source hooks/startup checks must install it
around the pinned runner in the stated order. No scheduler references, PP
sample-ring state or CUDA graph policy are changed here. Never log request IDs.
"""
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from functools import wraps
import json

from .block_mirror import Frame
from .retention_contract import ContractError, Pending

# q16f (25 Oct 2026, q16e dense.restart FAIL): the pinned vLLM runs its kernel warmup through the ORDINARY execute path
# (gpu_worker.compile_or_warm_up_model -> gpu/warmup.py warmup_kernels/_warmup_kernels and run_mixed_prefill_decode_warmup
# -> worker.execute_model(SchedulerOutput.make_empty() ...)), not with dummy_run=True. Those outputs never pass through
# Scheduler.schedule(), so they carry no flash_e2_frame and the instrumented worker refused them by contract (ContractError
# 'instrumented worker requires a scheduler frame' during EngineCore initialization). The pass-through below is bounded
# three ways: the output has no frame, EVERY request id it names uses vLLM's warmup prefixes, and no real frame has been
# accepted yet (the window closes for good at the first real RPC). Hooks are inert inside the window; the count is
# reported in every FLASH_E2_STATE record. A frameless output that fails any bound is still a ContractError.
WARMUP_ID_PREFIXES = ('_warmup_', '_v2_mixed_warmup')


def warmup_request_ids(output):
    """All request ids an execute_model SchedulerOutput names (new, cached, scheduled, finished); never logged."""
    ids = []
    for r in getattr(output, 'scheduled_new_reqs', None) or ():
        ids.append(getattr(r, 'req_id', None))
    cached = getattr(output, 'scheduled_cached_reqs', None)
    ids.extend(getattr(cached, 'req_ids', None) or ())
    ids.extend((getattr(output, 'num_scheduled_tokens', None) or {}).keys())
    ids.extend(getattr(output, 'finished_req_ids', None) or ())
    return tuple(ids)


@dataclass(frozen=True, repr=False)
class BatchRow:
    request_id: str
    slot: int
    start: int
    end: int
    length: int


class RunnerLifecycle:
    """Bind one ordinary execute_model RPC to immutable CPU request metadata.

    Call order: rpc -> existing finish/add/update + actual_cow -> updated ->
    existing prepare_inputs -> batch -> zero or more staged calls -> rpc exit.
    Dummy/profile/capture execute calls bypass rpc and never use staged().
    vLLM's kernel-warmup execute calls (frameless, warmup request ids only, before
    the first real frame) pass through warmup() with every hook inert.
    dma_enabled is sampled ONCE at RPC entry; a switch cannot divide its layers.
    """
    def __init__(self, mirror, dma, dma_enabled, emit):
        if dma.mirror is not mirror:
            raise ContractError('runner and DMA mirrors differ')
        self.mirror, self.dma = mirror, dma
        self.dma_enabled, self.emit = dma_enabled, emit
        self.frame = None
        self.updated_ok = False
        self.batch_rows = None
        self.actual_copy_counts = (0, 0)
        self.copy_seen = False
        self.enabled = False
        self.off_calls = {layer: 0 for layer in mirror.layer_groups}
        self.in_staged = False
        self.in_warmup = False
        self.warmup_rpcs = 0
        self.real_frames = 0

    def is_vllm_warmup(self, output):
        """Frameless output naming only vLLM warmup request ids, before any real frame was accepted."""
        if self.real_frames or getattr(output, 'flash_e2_frame', None) is not None:
            return False
        ids = warmup_request_ids(output)
        return bool(ids) and all(isinstance(i, str) and i.startswith(WARMUP_ID_PREFIXES) for i in ids)

    @contextmanager
    def warmup(self, output):
        """One pinned-vLLM warmup execute call: no frame, no mirror step, no emission; hooks are inert."""
        if self.frame is not None or self.in_warmup or self.mirror.active or self.mirror.failed:
            raise Pending('runner RPC is active or requires recovery')
        if not self.is_vllm_warmup(output):
            raise ContractError('warmup pass-through needs vLLM warmup request ids and no real frame yet')
        self.in_warmup = True
        try:
            yield
        except BaseException:
            self.mirror.failed = True
            raise
        finally:
            self.in_warmup = False
            self.warmup_rpcs += 1

    @contextmanager
    def rpc(self, output):
        if self.frame is not None or self.in_warmup or self.mirror.active or self.mirror.failed:
            raise Pending('runner RPC is active or requires recovery')
        frame = getattr(output, 'flash_e2_frame', None)
        if not isinstance(frame, Frame):
            raise ContractError('instrumented worker requires a scheduler frame')
        # No source, request or output payload is emitted.
        self.frame = frame
        self.real_frames += 1
        self.updated_ok = False
        self.batch_rows = None
        self.actual_copy_counts = (0, 0)
        self.copy_seen = False
        try:
            choice = self.dma_enabled()
            if type(choice) is not bool:
                raise ContractError('DMA switch must be boolean')
            self.enabled = choice
            yield
            if not self.updated_ok:
                raise ContractError('runner returned before CPU table validation')
            if frame.rows and self.batch_rows is None:
                raise ContractError('scheduled RPC returned without an input batch')
            if self.in_staged or self.dma.current is not None:
                raise Pending('runner returned with an active attention invocation')
            self.mirror.leave()
            self.emit(self.mirror.cow_line())
            if frame.retired or not frame.rows:
                # Pending producer markers are conservative stream provenance,
                # not live DMA leases; report them separately, never erase them
                # merely because the scheduler became idle.
                self.emit('FLASH_E2_STATE ' + json.dumps(dict(
                    self.mirror.counters(), dma=self.dma.stats,
                    routes_by_layer=self.dma.routes_by_layer, warmup_rpcs=self.warmup_rpcs), sort_keys=True))
        except BaseException:
            self.mirror.failed = True
            raise
        finally:
            if not self.mirror.failed:
                self.frame = None
                self.batch_rows = None
                self.updated_ok = False
                self.enabled = False

    def actual_cow(self, pairs, local_views):
        """Observe the EXISTING runner's filtered list, before its copy/verify.

        Its early return (no copies, no watchers) leaves the default zero. This
        hook must also run when QSA_COW_VERIFY takes its separate copy branch.
        """
        if self.in_warmup:
            return
        if self.frame is None or self.updated_ok or self.copy_seen:
            raise ContractError('CoW hook outside worker update or duplicated')
        if (type(pairs) is not int or pairs < 0 or type(local_views) is not int or
                local_views != len(self.mirror.layer_groups)):
            raise ContractError('invalid actual CoW count')
        self.actual_copy_counts = (pairs, pairs * local_views)
        self.copy_seen = True

    def updated(self, slots, cpu_tables):
        if self.in_warmup:
            return
        if self.frame is None or self.updated_ok:
            raise ContractError('CPU table hook outside RPC or duplicated')
        self.mirror.enter(self.frame, slots, cpu_tables)
        for note in self.mirror.take_divergence_notes():
            self.emit('FLASH_E2_TABLES ' + json.dumps(note, sort_keys=True))
        if self.actual_copy_counts != self.mirror.frame_copies:
            self.mirror.failed = True
            raise ContractError('actual runner CoW list differs from scheduler mirror')
        self.updated_ok = True

    def batch(self, input_batch):
        """Copy metadata from the real InputBatch, after real input preparation.

        CPU upper bounds are accepted only when they equal the immutable frame
        for every row. Speculative/adaptive/context-parallel paths are outside
        this first serial candidate. GPU lengths/table equality are an additional
        runtime exactness gate, not something this CPU comparison can establish.
        """
        if self.in_warmup:
            return
        if not self.updated_ok or self.batch_rows is not None:
            raise ContractError('input batch hook outside RPC or duplicated')
        n = input_batch.num_reqs
        if type(n) is not int or n != len(self.frame.rows):
            raise ContractError('input batch request count differs from frame')
        ids = tuple(input_batch.req_ids)
        arrays = (input_batch.idx_mapping_np, input_batch.num_scheduled_tokens,
                  input_batch.num_computed_tokens_np, input_batch.seq_lens_cpu_upper_bound)
        if len(ids) != n or len(set(ids)) != n or any(len(x) < n for x in arrays):
            raise ContractError('input batch arrays disagree')
        seq = input_batch.seq_lens_cpu_upper_bound
        if str(seq.device) != 'cpu':
            raise ContractError('sequence metadata must be CPU resident')
        offsets = input_batch.query_start_loc_np
        if len(offsets) < n + 1 or int(offsets[0]) != 0:
            raise ContractError('input batch offsets invalid')
        expected = {r.request_id: r for r in self.frame.rows}
        if set(ids) != set(expected):
            raise ContractError('input batch requests differ from frame')
        rows = []
        for i, rid in enumerate(ids):
            row = expected[rid]
            start, end = int(offsets[i]), int(offsets[i + 1])
            slot = int(input_batch.idx_mapping_np[i])
            key = self.mirror.requests[rid]
            length = int(seq[i])
            if (key.uid != 'slot' + str(slot) or
                    int(input_batch.num_computed_tokens_np[i]) != row.computed or
                    int(input_batch.num_scheduled_tokens[i]) != row.scheduled or
                    end - start != row.scheduled or start < 0 or
                    length != row.computed + row.scheduled):
                raise ContractError('input batch row differs from scheduler/worker mapping')
            rows.append(BatchRow(rid, slot, start, end, length))
        self.batch_rows = tuple(rows)

    def staged(self, layer, best, start, end, length, args, original, candidate, *, gpu_metadata=None):
        """Route the existing staged call, with its exact positional arguments.

        args = query, UVA-KV, indices, GPU table row, rows, max_page, EXISTING
        arena, use_prefill_config, output. Both callables use these unchanged.
        Candidate additionally accepts e2=Session; no retained= shortcut.
        """
        if self.in_warmup:
            # Unreachable through the STAGED hook (it requires a frame); kept so a warmup call can never take the arena.
            original(*args)
            return
        if (not self.updated_ok or self.batch_rows is None or self.in_staged or
                layer not in self.mirror.layer_groups or type(best) is not int or
                not 0 <= best < len(self.batch_rows) or len(args) != 9):
            raise ContractError('staged call outside matching worker batch')
        row = self.batch_rows[best]
        if (start, end, length) != (row.start, row.end, row.length):
            raise ContractError('staged row/length disagrees with input batch')
        if args[5] != (length - 1) // self.mirror.pool.page_tokens:
            raise ContractError('staged logical page frontier differs from mirror')
        if hasattr(self, 'uva_facts') and self.dma.backend.facts(args[1]) != self.uva_facts[layer]:
            raise ContractError('staged UVA binding changed after startup')
        self.in_staged = True
        try:
            if not self.enabled:
                self.off_calls[layer] += 1
                original(*args)
            else:
                oracle = getattr(self, 'exactness', None)
                context = oracle.layer(self, layer, best, row, args, gpu_metadata) if oracle else nullcontext()
                with context:
                    with self.dma.session(row.request_id, layer, args[6], args[3], length) as session:
                        candidate(*args, e2=session)
                    if oracle:
                        oracle.output(self, original, session.route_name)
                if oracle:
                    oracle.completed()
        except BaseException:
            self.mirror.failed = True
            raise
        finally:
            if not self.mirror.failed:
                self.in_staged = False

    def shutdown(self, wait):
        if self.frame is not None or self.in_staged or self.dma.current is not None:
            raise Pending('shutdown with an active serial RPC')
        state = self.mirror.shutdown(wait)
        self.emit('FLASH_E2_DRAIN ' + json.dumps(state, sort_keys=True))
        return state


def execute_with_e2(method):
    """Decorator matching the pinned execute_model signature, including dummy.

    Called outside torch.inference_mode's wrapper. Off/absent and dummy calls
    invoke the original method exactly once; no PP or graph method is called here.
    """
    @wraps(method)
    def wrapped(self, scheduler_output, intermediate_tensors=None, dummy_run=False,
                skip_attn_for_dummy_run=False, is_profile=False, context_len=0,
                valid_dummy_state_slots=False):
        args = (self, scheduler_output, intermediate_tensors, dummy_run,
                skip_attn_for_dummy_run, is_profile, context_len, valid_dummy_state_slots)
        lifecycle = getattr(self, '_flash_e2_lifecycle', None)
        if lifecycle is None or dummy_run:
            return method(*args)
        if is_profile or skip_attn_for_dummy_run or context_len or valid_dummy_state_slots:
            raise ContractError('profiling/capture flags on a real E2 RPC')
        if lifecycle.is_vllm_warmup(scheduler_output):
            with lifecycle.warmup(scheduler_output):
                return method(*args)
        with lifecycle.rpc(scheduler_output):
            return method(*args)
    return wrapped
