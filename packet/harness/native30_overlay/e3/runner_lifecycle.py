"""E2 serial runner seam, CPU-only until supplied the event-backed DMA adapter.

This is not a deployable overlay. Source hooks/startup checks must install it
around the pinned runner in the stated order. No scheduler references, PP
sample-ring state or CUDA graph policy are changed here. Never log request IDs.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
import json

from block_mirror import Frame
from retention_contract import ContractError, Pending


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

    @contextmanager
    def rpc(self, output):
        if self.frame is not None or self.mirror.active or self.mirror.failed:
            raise Pending('runner RPC is active or requires recovery')
        frame = getattr(output, 'flash_e2_frame', None)
        if not isinstance(frame, Frame):
            raise ContractError('instrumented worker requires a scheduler frame')
        # No source, request or output payload is emitted.
        self.frame = frame
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
        if self.frame is None or self.updated_ok or self.copy_seen:
            raise ContractError('CoW hook outside worker update or duplicated')
        if (type(pairs) is not int or pairs < 0 or type(local_views) is not int or
                local_views != len(self.mirror.layer_groups)):
            raise ContractError('invalid actual CoW count')
        self.actual_copy_counts = (pairs, pairs * local_views)
        self.copy_seen = True

    def updated(self, slots, cpu_tables):
        if self.frame is None or self.updated_ok:
            raise ContractError('CPU table hook outside RPC or duplicated')
        self.mirror.enter(self.frame, slots, cpu_tables)
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

    def staged(self, layer, best, start, end, length, args, original, candidate):
        """Route the existing staged call, with its exact positional arguments.

        args = query, UVA-KV, indices, GPU table row, rows, max_page, EXISTING
        arena, use_prefill_config, output. Both callables use these unchanged.
        Candidate additionally accepts e2=Session; no retained= shortcut.
        """
        if (not self.updated_ok or self.batch_rows is None or self.in_staged or
                layer not in self.mirror.layer_groups or type(best) is not int or
                not 0 <= best < len(self.batch_rows) or len(args) != 9):
            raise ContractError('staged call outside matching worker batch')
        row = self.batch_rows[best]
        if (start, end, length) != (row.start, row.end, row.length):
            raise ContractError('staged row/length disagrees with input batch')
        if args[5] != (length - 1) // self.mirror.pool.page_tokens:
            raise ContractError('staged logical page frontier differs from mirror')
        self.in_staged = True
        try:
            if not self.enabled:
                self.off_calls[layer] += 1
                original(*args)
            else:
                with self.dma.session(row.request_id, layer, args[6], args[3], length) as session:
                    candidate(*args, e2=session)
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
        with lifecycle.rpc(scheduler_output):
            return method(*args)
    return wrapped
