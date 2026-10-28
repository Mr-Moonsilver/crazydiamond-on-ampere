"""Source-backed tests for the native admission-metadata provider.

These tests construct the real :class:`admission_metadata.NativeAdmissionMetadata`
and the real :class:`scheduler_hooks.PrefillAdmissionController` over a
source-faithful fixture of the pinned allocator. The fixture reproduces the
pinned per-group queue semantics, CoW redirect, Mamba "align" hash relocation
and stock copy-retention release; only the block pool (id allocation, refcount
accounting), the scheduler's deferred-free hook and the wire/leases boundaries
are faked.

The pinned ``KVCacheBlock`` class is executed from the reviewed reference
source bytes, so endpoint identity checks exercise the real dataclass. Missing
source fixtures are not a passing test: a digest or text mismatch fails here.

No GPU, no vLLM process, no engine. CPU only.
"""
from dataclasses import dataclass, field
from pathlib import Path
import ast
import hashlib
import sys
import types
import unittest
from typing import NewType
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'patches/s5_engine'))

import admission_metadata as am
import async_handoff as ah
import engine_driver as edrv
import executor_hooks as eh
import native_backend as nb
import native_resources as nr
import scheduler_hooks as sh
import s5_startup as su
import wire_v1 as w
from e3_types import (
    CompletionKind,
    CompletionProof,
    DependencyRole,
    E3ContractError,
    E3Pending,
    LaneId,
    LaneRole,
    OperationTicket,
)

REF = ROOT / 'reference/vllm-src'
UID = 'req-admission-metadata'

BlockHash = NewType('BlockHash', bytes)
BlockHashWithGroupId = NewType('BlockHashWithGroupId', bytes)


# ---------------------------------------------------------------------------
# Pinned source fixtures
# ---------------------------------------------------------------------------

def _read(relative):
    return (REF / relative).read_text()


def _class_source(relative, name):
    """Return the exact reviewed class source bytes (including decorators)."""
    text = _read(relative)
    tree = ast.parse(text)
    lines = text.splitlines()
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            first = min([node.lineno] + [item.lineno for item in node.decorator_list])
            return '\n'.join(lines[first - 1:node.end_lineno]), text
    raise AssertionError(f'{name} is missing from {relative}')


def _pinned_block_class():
    source, _ = _class_source('v1/core/kv_cache_utils.py', 'KVCacheBlock')
    namespace = {
        'dataclass': dataclass,
        'field': field,
        'BlockHashWithGroupId': BlockHashWithGroupId,
        'BlockHash': BlockHash,
    }
    exec(compile(source, 'pinned/kv_cache_utils.py', 'exec'), namespace)
    klass = namespace['KVCacheBlock']
    # The executed bytes, not a reimplementation, must be what we exercise.
    assert 'class KVCacheBlock' in source
    return klass


PinnedBlock = _pinned_block_class()


def cdiv(a, b):
    return (a + b - 1) // b


class FixturePool:
    """External allocator boundary: physical ids, refcounts, hash moves."""

    def __init__(self, *, first_id=1):
        self.next_id = first_id
        self.null = PinnedBlock(0, is_null=True)
        # Index-addressed, exactly like the pinned pool: ``pool.blocks[id]``
        # is the authoritative object for that physical page.
        self.blocks = [self.null]
        self.created = []
        self.freed = []
        self.hash_moves = []

    def precreate(self, count=1, *, ref=0):
        blocks = []
        for _ in range(count):
            block = PinnedBlock(self.next_id)
            self.next_id += 1
            block.ref_cnt = ref
            self.created.append(block)
            blocks.append(block)
            if block.block_id != len(self.blocks):
                raise AssertionError('fixture blocks are not index-addressed')
            self.blocks.append(block)
        return blocks

    def get_new_blocks(self, count):
        if type(count) is not int or count < 0:
            raise AssertionError('get_new_blocks needs a nonnegative int')
        blocks = self.precreate(count, ref=1)
        return blocks

    def touch(self, blocks):
        for block in blocks:
            if block.is_null:
                raise AssertionError('stock never touches the null block as a hit')
            block.ref_cnt += 1

    def free_blocks(self, blocks):
        for block in list(blocks):
            if block.is_null:
                raise AssertionError('stock never frees the null block')
            block.ref_cnt -= 1
            if block.ref_cnt < 0:
                raise AssertionError(f'reference underflow for block {block.block_id}')
            self.freed.append(block.block_id)

    def move_block_hashes(self, source, destination):
        if source.is_null or destination.is_null:
            raise AssertionError('hash relocation must use real blocks')
        if destination.block_hash is not None:
            raise AssertionError('destination already carries a cache entry')
        self.hash_moves.append((source.block_id, destination.block_id,
                                source.block_hash_num_tokens))
        destination.set_block_hash(source.block_hash, source.block_hash_num_tokens)
        source.reset_hash()

    def block(self, block_id):
        for block in self.created:
            if block.block_id == block_id:
                return block
        raise AssertionError(f'unknown fixture block {block_id}')


class FixtureBlocks:
    """The pinned ``KVCacheBlocks`` surface this manager consumes."""

    def __init__(self, per_group):
        self.blocks = tuple(tuple(row) for row in per_group)

    def get_block_ids(self, group_id=None):
        groups = self.blocks if group_id is None else (self.blocks[group_id],)
        return tuple([block.block_id for block in row] for row in groups)

    def __len__(self):
        return sum(len(row) for row in self.blocks)


class FixtureGroup:
    """Source-faithful per-group queue and allocation behaviour."""

    def __init__(self, pool, *, group_id=0, block_size=4, records=None,
                 fail_after_append=False):
        self.kv_cache_group_id = group_id
        self.block_size = block_size
        self.block_pool = pool
        self.pool = pool
        self.new_block_ids = []
        self._pending_cow_copies = []
        self._pending_boundary_state_offloads = []
        self.req_to_blocks = {}
        self.num_cached_block = {}
        self._partial_hit_reqs = {}
        self._allocated_block_reqs = set()
        self.last_state_block_idx = {}
        self.producer_tails = {}
        self.removed = []
        self.cache_calls = []
        self.fail_after_append = fail_after_append
        self._record_new_block_ids = (self.kind == 'attention' if records is None
                                      else bool(records))

    kind = 'attention'

    @property
    def records_new_block_ids(self):
        return self._record_new_block_ids

    # Pinned base hook: only windowed managers skip tokens.
    def get_num_skipped_tokens(self, num_computed_tokens):
        return 0

    def add_local_computed_blocks(self, request_id, blocks, num_local_computed_tokens,
                                  num_external_computed_tokens=0):
        if num_external_computed_tokens:
            raise AssertionError('the fixture does not model external loads')
        row = self.req_to_blocks.setdefault(request_id, [])
        if row:
            raise AssertionError('prefix hits are only added to a fresh row')
        blocks = list(blocks)
        skipped_tokens = self.get_num_skipped_tokens(num_local_computed_tokens)
        skipped_blocks = skipped_tokens // self.block_size
        if skipped_blocks:
            blocks = blocks[skipped_blocks:]
        self.pool.touch(blocks)
        row.extend([self.pool.null] * skipped_blocks)
        row.extend(blocks)
        self.num_cached_block[request_id] = len(row)
        if blocks and num_local_computed_tokens % self.block_size != 0:
            # Pinned _has_partial_local_hit + add_local_computed_blocks.
            block_idx = num_local_computed_tokens // self.block_size
            self._partial_hit_reqs[request_id] = (block_idx, blocks[-1])
            self.num_cached_block[request_id] = block_idx

    def remove_skipped_blocks(self, request_id, processed_computed_tokens):
        skipped = self.get_num_skipped_tokens(processed_computed_tokens)
        if skipped <= 0:
            return
        row = self.req_to_blocks.get(request_id)
        if row is None:
            return
        count = min(skipped // self.block_size, len(row))
        for index in range(count):
            block = row[index]
            if block.is_null:
                continue
            row[index] = self.pool.null
            self.pool.free_blocks([block])
            self.removed.append(block.block_id)

    def _apply_cow(self, request_id, block_idx, source, cow):
        row = self.req_to_blocks[request_id]
        if row[block_idx] is not source:
            raise AssertionError('CoW source is not the row entry')
        if source.is_null or source.ref_cnt <= 0:
            raise AssertionError('CoW source must be a live cached block')
        row[block_idx] = cow
        self._pending_cow_copies.append((source, cow))
        cow.ref_cnt += 1

    def allocate_new_blocks(self, request_id, num_tokens, num_tokens_main_model):
        row = self.req_to_blocks.setdefault(request_id, [])
        cow_blocks = []
        partial = self._partial_hit_reqs.pop(request_id, None)
        if partial is not None:
            block_idx, source = partial
            cow = self.pool.get_new_blocks(1)[0]
            self._apply_cow(request_id, block_idx, source, cow)
            self.new_block_ids.append(cow.block_id)
            cow_blocks.append(cow)
        required = cdiv(num_tokens, self.block_size)
        count = required - len(row)
        if count > 0:
            new_blocks = self.pool.get_new_blocks(count)
            row.extend(new_blocks)
            if self._record_new_block_ids:
                self.new_block_ids.extend(block.block_id for block in new_blocks)
            cow_blocks.extend(new_blocks)
        if self.fail_after_append:
            raise RuntimeError('fixture allocation failure after mutation')
        return cow_blocks

    def take_new_block_ids(self):
        ids = self.new_block_ids
        self.new_block_ids = []
        return ids

    def take_pending_cow_copies(self):
        pending = self._pending_cow_copies
        self._pending_cow_copies = []
        return pending

    def take_pending_boundary_state_offloads(self):
        pending = self._pending_boundary_state_offloads
        self._pending_boundary_state_offloads = []
        return pending

    def cache_blocks(self, request, num_tokens):
        self.cache_calls.append((request.request_id, num_tokens))
        row = self.req_to_blocks.get(request.request_id)
        if row is None:
            return
        frontier = self.num_cached_block.get(request.request_id, 0)
        full = num_tokens // self.block_size
        for index in range(frontier, min(full, len(row))):
            block = row[index]
            if block.is_null:
                continue
            if block.block_hash is None:
                block.set_block_hash(BlockHashWithGroupId(bytes([index, self.kv_cache_group_id])),
                                     (index + 1) * self.block_size)
            self.num_cached_block[request.request_id] = index + 1


class FixtureEmptyGroup(FixtureGroup):
    """A KV group with no blocks for this request (e.g. an unused spec)."""

    kind = 'empty'
    _record_new_block_ids = False

    def add_local_computed_blocks(self, request_id, blocks, *args, **kwargs):
        if blocks:
            raise AssertionError('the empty group cannot have prefix hits')
        self.req_to_blocks.setdefault(request_id, [])
        self.num_cached_block[request_id] = 0

    def allocate_new_blocks(self, request_id, num_tokens, num_tokens_main_model):
        self.req_to_blocks.setdefault(request_id, [])
        return []


class FixtureMambaGroup(FixtureGroup):
    """Mamba 'align' allocation: running-request hash relocation and offers."""

    kind = 'mamba'
    _record_new_block_ids = False

    def get_num_skipped_tokens(self, num_computed_tokens):
        return num_computed_tokens - 1

    def allocate_new_blocks(self, request_id, num_tokens, num_tokens_main_model):
        row = self.req_to_blocks.setdefault(request_id, [])
        blocks_allocated = request_id in self._allocated_block_reqs
        required = cdiv(num_tokens_main_model, self.block_size)
        partial = self._partial_hit_reqs.pop(request_id, None)
        if required <= len(row) and partial is None:
            self._allocated_block_reqs.add(request_id)
            return []
        previous = len(row)
        if previous > 0:
            self.last_state_block_idx[request_id] = previous - 1
        skipped = required - 1
        if previous < skipped:
            row.extend([self.pool.null] * (skipped - previous))
        extra = 0 if partial is None else 1
        new_blocks = self.pool.get_new_blocks(max(required - len(row), 0) + extra)
        returned = []
        if partial is not None:
            block_idx, source = partial
            cow = new_blocks[0]
            new_blocks = new_blocks[1:]
            if blocks_allocated:
                self.pool.move_block_hashes(source, cow)
                self._pending_cow_copies.append((source, cow))
                source.ref_cnt += 1
                tail = self.producer_tails.pop(request_id, None)
                if tail is not None:
                    marker, boundary_tokens = tail
                    if marker is not source:
                        raise AssertionError('producer tail marker differs')
                    self._pending_boundary_state_offloads.append(
                        (request_id, self.kv_cache_group_id, cow, boundary_tokens))
            else:
                self._apply_cow(request_id, block_idx, source, cow)
                returned.append(cow)
        row.extend(new_blocks)
        self._allocated_block_reqs.add(request_id)
        returned.extend(new_blocks)
        if self.fail_after_append:
            raise RuntimeError('fixture allocation failure after mutation')
        return returned


class FixtureHiSparseGroup(FixtureGroup):
    """Host-tier group shape: the provider must refuse this before mutation."""

    def take_host_cow_copies(self):
        return []


class FixtureManager:
    """The pinned ``KVCacheManager`` surface the controller calls."""

    def __init__(self, pool, groups, *, hit_groups=None, hit_tokens=0,
                 fail_before_mutation=False, return_none_after_mutation=False):
        self.block_pool = pool
        self.groups = tuple(groups)
        self.coordinator = type('Coordinator', (), {
            'single_type_managers': self.groups})()
        self.hit_groups = hit_groups
        self.hit_tokens = hit_tokens
        self.fail_before_mutation = fail_before_mutation
        self.return_none_after_mutation = return_none_after_mutation
        self.alloc_calls = []
        self.cache_calls = []
        self.lookup_calls = 0

    def get_computed_blocks(self, request):
        self.lookup_calls += 1
        if self.hit_groups is None:
            return None, 0, 0
        return (FixtureBlocks(self.hit_groups), self.hit_tokens, 0)

    def allocate_slots(self, request, num_new_tokens, num_new_computed_tokens=0,
                       new_computed_blocks=None, num_lookahead_tokens=0,
                       delay_cache_blocks=False, full_sequence_must_fit=False,
                       reserved_blocks=0, has_scheduled_reqs=True):
        self.alloc_calls.append(dict(
            request=request, num_new=num_new_tokens,
            num_new_computed=num_new_computed_tokens,
            new_computed_blocks=new_computed_blocks,
            delay_cache_blocks=delay_cache_blocks,
            num_lookahead=num_lookahead_tokens,
            full_sequence_must_fit=full_sequence_must_fit,
            reserved_blocks=reserved_blocks,
            has_scheduled_reqs=has_scheduled_reqs))
        if self.fail_before_mutation:
            return None
        request_id = request.request_id
        local = request.num_computed_tokens + num_new_computed_tokens
        blocks_allocated = request.num_computed_tokens > 0
        if new_computed_blocks is not None and not blocks_allocated:
            for index, group in enumerate(self.groups):
                group.add_local_computed_blocks(
                    request_id, new_computed_blocks.blocks[index], local, 0)
        processed = max(0, local - getattr(request, 'num_in_flight_tokens', 0))
        for group in self.groups:
            group.remove_skipped_blocks(request_id, processed)
        total = local + num_new_tokens
        per_group = [group.allocate_new_blocks(request_id, total, total)
                     for group in self.groups]
        if self.return_none_after_mutation:
            return None
        return FixtureBlocks(per_group)

    def get_block_ids(self, request_id):
        return tuple([block.block_id for block in self.groups[index].req_to_blocks[request_id]]
                     for index in range(len(self.groups)))

    def cache_blocks(self, request, num_tokens):
        self.cache_calls.append((request.request_id, num_tokens))
        for group in self.groups:
            group.cache_blocks(request, num_tokens)


class Request:
    def __init__(self, uid=UID, tokens=12, computed=0):
        self.request_id = uid
        self.all_token_ids = list(range(tokens))
        self.prompt_token_ids = list(range(tokens))
        self.num_tokens = tokens
        self.num_prompt_tokens = tokens
        self.num_computed_tokens = computed
        self.num_in_flight_tokens = 0
        self.num_output_tokens = 0
        self.status = 'waiting'


class Queue:
    def __init__(self, requests=()):
        self.items = list(requests)

    def peek_request(self):
        return self.items[0]

    def remove_request(self, request):
        self.items.remove(request)

    def prepend_request(self, request):
        self.items.insert(0, request)

    def __len__(self):
        return len(self.items)

    def __bool__(self):
        return bool(self.items)


def text_config(*, ple=(2,), ngram=3, eos=248044):
    return type('TextConfig', (), {
        'ple_layer_ids': ple, 'ngram_size': ngram, 'eos_token_id': eos})()


class FixtureScheduler:
    """Scheduler boundary: deferred-free route, queues and declared settings."""

    def __init__(self, manager, requests=(), *, text=None, needs_zeroing=True):
        self.kv_cache_manager = manager
        self.waiting = Queue(requests)
        self.running = []
        self.max_num_running_reqs = 8
        self.max_num_scheduled_tokens = 1024
        self.scheduler_reserve_full_isl = False
        self.num_waiting_for_streaming_input = 0
        self._pause_state = None
        self.connector = None
        self.needs_kv_cache_zeroing = needs_zeroing
        self._skip_zero_block_ids = set()
        self.sched_step_seq = 10
        self.processed_step_seq = 10
        self.defer_block_free = False
        self.deferred_frees = []
        self.retained_frees = []
        self._flash_s5_admission = None
        self._flash_s5_admission_metadata = None
        self.vllm_config = type('Config', (), {
            'model_config': type('ModelConfig', (), {
                'hf_text_config': text if text is not None else text_config()})(),
            'speculative_config': None})()
        self.pool = manager.block_pool

    def _select_waiting_queue_for_scheduling(self):
        return self.waiting or None

    def _is_blocked_waiting_status(self, status):
        return False

    def _free_cow_retained_blocks(self, blocks, fence):
        self.retained_frees.append((fence, tuple(blocks)))
        if not self.defer_block_free or fence <= self.processed_step_seq:
            self.kv_cache_manager.block_pool.free_blocks(blocks)
            return
        # Pinned route: the deferred drain consults the step fence later.
        self.deferred_frees.append((fence, tuple(blocks)))


def make_controller(scheduler, *, chunk=4, ranks=(0, 1), max_history=64,
                    leases=None, frontier=None):
    if getattr(scheduler, '_flash_s5_admission_metadata', None) is None:
        am.install_admission_metadata(scheduler)
    controller = sh.PrefillAdmissionController(
        scheduler, leases=leases or sh.SchedulerLeases(),
        frontier=frontier or ah.CommittedFrontier(), engine_epoch=7,
        chunk_tokens=chunk, ranks=ranks, max_history=max_history)
    return controller.attach()


def make_ticket(transaction, rank, op_seq, role):
    return OperationTicket(transaction, rank,
                           LaneId(rank, LaneRole.PREFILL, 5, 0), op_seq, role)


def make_proof(transaction, rank, op_seq, role, kind, completion_seq=1):
    return CompletionProof(make_ticket(transaction, rank, op_seq, role), kind,
                           completion_seq)


def make_report(transaction, rank, completed, proofs, *, pending=(), epoch=1):
    return w.WorkerCompletionReport(
        transaction, rank, tuple(sorted(completed)), tuple(sorted(pending)),
        max([p.completion_seq for p in proofs] + [0]), tuple(proofs),
        w.CompletionCursor('commit', 0, 0), epoch)


def reports_for(transaction, ranks=(0, 1), epoch=1):
    first = ranks[0]
    reports = []
    for rank in ranks:
        if rank == first:
            proofs = (
                make_proof(transaction, rank, 0, DependencyRole.PRODUCER,
                           CompletionKind.GPU_EVENT),
                make_proof(transaction, rank, 1, DependencyRole.COMMIT,
                           CompletionKind.STATE_WRITE))
            completed = (0, 1)
        else:
            proofs = (make_proof(transaction, rank, 0, DependencyRole.COMMIT,
                                 CompletionKind.STATE_WRITE),)
            completed = (0,)
        reports.append(make_report(transaction, rank, completed, proofs, epoch=epoch))
    return reports


def finish(controller, record, *, commit=True):
    """Drive the real lifecycle to commit or the full-barrier cancellation."""
    transaction = record.transaction
    controller.bind(make_ticket(transaction, 0, 0, DependencyRole.PRODUCER))
    for report in reports_for(transaction, controller.ranks, record.state_epoch):
        controller.note_report(report)
        controller.note_worker_release(w.ReleaseAck(
            transaction, report.worker_rank, record.state_epoch,
            report.completed_sequence, released=True,
            request_closed=record.terminal))
    proof = make_proof(transaction, 0, 0, DependencyRole.PRODUCER,
                       CompletionKind.GPU_EVENT)
    if commit:
        return controller.commit(transaction, proof)
    return controller.cancel(transaction, proof=proof)


def free_request_rows(manager, request_id):
    """Engine free: drop the request's own references, once."""
    for group in manager.groups:
        row = group.req_to_blocks.pop(request_id, [])
        for block in row:
            if not block.is_null:
                group.pool.free_blocks([block])
        group.num_cached_block.pop(request_id, None)


# ---------------------------------------------------------------------------
# 1. Source facts
# ---------------------------------------------------------------------------

class PinnedSourceTests(unittest.TestCase):
    def test_provider_pins_match_the_reviewed_bytes(self):
        self.assertTrue(am.verify_sources(REF))
        self.assertEqual(
            hashlib.sha256((REF / am.SINGLE_TYPE_PATH).read_bytes()).hexdigest(),
            am.SINGLE_TYPE_SHA256)

    def test_attention_cow_and_queue_semantics_are_the_pinned_ones(self):
        source, _ = _class_source('v1/core/single_type_kv_cache_manager.py',
                                  'SingleTypeKVCacheManager')
        self.assertIn('self._pending_cow_copies.append((source_block, cow_block))',
                      source)
        self.assertIn('cow_block.ref_cnt += 1', source)
        self.assertIn('self.new_block_ids.append(cow_block.block_id)', source)
        self.assertIn('def take_new_block_ids', source)
        self.assertIn('self.new_block_ids = []', source)
        self.assertIn('self._pending_cow_copies = []', source)

    def test_mamba_align_moves_the_hash_before_the_copy_finishes(self):
        source, _ = _class_source('v1/core/single_type_kv_cache_manager.py',
                                  'MambaManager')
        self.assertIn('self.block_pool.move_block_hashes(source_block, cow_block)',
                      source)
        self.assertIn('self._pending_boundary_state_offloads.append(', source)
        self.assertIn('self.kv_cache_group_id,', source)
        self.assertIn('boundary_tokens,', source)

    def test_stock_scheduler_drains_queues_every_step(self):
        source = _read(am.SCHEDULER_PATH)
        self.assertIn('new_block_ids_to_zero=self._get_new_block_ids_to_zero()',
                      source)
        self.assertIn('boundary_state_offloads = '
                      'self.kv_cache_manager.take_boundary_state_offloads()', source)
        self.assertIn('self._free_cow_retained_blocks(cow_retained_blocks, '
                      'self.sched_step_seq + 1)', source)
        self.assertIn('if self._skip_zero_block_ids:', source)
        self.assertIn('skip.clear()', source)

    def test_pinned_block_class_is_executed_not_reimplemented(self):
        block = PinnedBlock(3)
        self.assertEqual(block.block_id, 3)
        self.assertEqual(block.ref_cnt, 0)
        self.assertFalse(block.is_null)
        block.set_block_hash(BlockHashWithGroupId(b'h'), 4)
        self.assertEqual(block.block_hash, b'h')
        block.reset_hash()
        self.assertIsNone(block.block_hash)


# ---------------------------------------------------------------------------
# 2. Scoped capture around one allocation
# ---------------------------------------------------------------------------

class ScopedCaptureTests(unittest.TestCase):
    def _scheduler(self, *, groups=None, request=None, needs_zeroing=True,
                   **manager_kwargs):
        pool = FixturePool()
        groups = groups if groups is not None else [FixtureGroup(pool)]
        manager = FixtureManager(pool, groups, **manager_kwargs)
        request = request or Request()
        scheduler = FixtureScheduler(manager, [request], needs_zeroing=needs_zeroing)
        return pool, manager, request, scheduler

    def test_unrelated_pending_work_keeps_identity_and_stock_order(self):
        pool, manager, request, scheduler = self._scheduler()
        group = manager.groups[0]
        foreign_new = 900
        foreign_src, foreign_dst = pool.precreate(2, ref=1)
        group.new_block_ids.extend([foreign_new])
        group._pending_cow_copies.append((foreign_src, foreign_dst))
        group._pending_boundary_state_offloads.append(
            ('other-request', 0, foreign_src, 7))
        before_new = group.new_block_ids
        before_cow = group._pending_cow_copies
        before_boundary = group._pending_boundary_state_offloads

        controller = make_controller(scheduler)
        record = controller.admit(scheduler)
        capture = controller.parked.capture

        # Exactly this allocation's queue suffix was taken.
        allocated = group.req_to_blocks[UID][0].block_id
        self.assertEqual(capture.zero_ids, (allocated,))
        self.assertEqual(record.new_block_ids_to_zero, (allocated,))
        self.assertEqual(capture.cow_pairs, ())
        # Unrelated work is still present, in stock order and identity.
        self.assertIs(group.new_block_ids, before_new)
        self.assertEqual(group.new_block_ids, [foreign_new])
        self.assertIs(group._pending_cow_copies, before_cow)
        self.assertEqual(len(group._pending_cow_copies), 1)
        self.assertIs(group._pending_cow_copies[0][0], foreign_src)
        self.assertIs(group._pending_cow_copies[0][1], foreign_dst)
        self.assertEqual(group._pending_boundary_state_offloads,
                         [('other-request', 0, foreign_src, 7)])
        # The stock drains still see exactly the unrelated prefix.
        self.assertEqual(group.take_new_block_ids(), [foreign_new])
        self.assertEqual(group.take_pending_cow_copies(),
                         [(foreign_src, foreign_dst)])

    def test_prefix_mutation_during_allocation_is_refused(self):
        pool, manager, request, scheduler = self._scheduler()
        group = manager.groups[0]
        group.new_block_ids.append(900)
        scheduler._flash_s5_admission_metadata = am.install_admission_metadata(scheduler)
        provider = scheduler._flash_s5_admission_metadata
        provider.begin(scheduler, request)
        # A concurrent stock drain/replace is not attributable.
        group.new_block_ids = []
        with self.assertRaises(E3ContractError):
            provider.capture(request, object(), 0, 4)

    def test_zeroing_disabled_drains_and_discards_with_evidence(self):
        pool, manager, request, scheduler = self._scheduler(needs_zeroing=False)
        controller = make_controller(scheduler)
        record = controller.admit(scheduler)
        capture = controller.parked.capture
        self.assertEqual(record.new_block_ids_to_zero, ())
        self.assertEqual(capture.evidence['discarded_zero_ids'], 1)
        self.assertEqual(capture.evidence['zeroing_reason'],
                         'needs_kv_cache_zeroing is false')
        # Stock still drains the queue even when it discards the ids.
        self.assertEqual(manager.groups[0].new_block_ids, [])

    def test_skip_zero_ids_filters_only_this_allocation(self):
        pool, manager, request, scheduler = self._scheduler()
        group = manager.groups[0]
        group.new_block_ids.append(900)
        scheduler._skip_zero_block_ids.update({900, 1})
        controller = make_controller(scheduler)
        record = controller.admit(scheduler)
        self.assertEqual(record.new_block_ids_to_zero, ())
        self.assertEqual(controller.parked.capture.evidence['skipped_zero_ids'], (1,))
        # The unrelated skip entry is not consumed by this admission.
        self.assertEqual(scheduler._skip_zero_block_ids, {900})
        self.assertEqual(group.new_block_ids, [900])

    def test_allocation_failure_keeps_the_orphan_journal(self):
        pool, manager, request, scheduler = self._scheduler()
        manager.groups[0].fail_after_append = True
        controller = make_controller(scheduler)
        with self.assertRaises(RuntimeError):
            controller.admit(scheduler)
        capture = controller.parked.capture
        self.assertTrue(controller.parked.failed)
        self.assertTrue(capture.orphan)
        self.assertEqual(capture.zero_ids, (1,))
        # Nothing was consumed: the queues still hold the side effect.
        self.assertEqual(manager.groups[0].new_block_ids, [1])
        self.assertEqual(len(manager.groups[0].new_block_ids), 1)
        self.assertEqual(controller.leases.held_blocks(), ())
        self.assertFalse(scheduler.waiting)
        # Queue ownership is exclusive: the journal must not also decrement.
        with self.assertRaises(E3ContractError):
            capture.release_retained(scheduler)
        self.assertEqual(manager.groups[0].pool.block(1).ref_cnt, 1)
        # No retry can consume another transaction's work.
        self.assertIsNone(controller.admit(scheduler))

    def test_none_after_recording_park_is_quarantined(self):
        pool, manager, request, scheduler = self._scheduler()
        manager.return_none_after_mutation = True
        controller = make_controller(scheduler)
        with self.assertRaises(E3ContractError):
            controller.admit(scheduler)
        capture = controller.parked.capture
        self.assertTrue(capture.orphan)
        self.assertEqual(capture.zero_ids, (1,))
        self.assertEqual(manager.groups[0].new_block_ids, [1])
        self.assertFalse(scheduler.waiting)

    def test_clean_allocation_failure_stays_ordinary_stock(self):
        pool, manager, request, scheduler = self._scheduler()
        manager.fail_before_mutation = True
        controller = make_controller(scheduler)
        self.assertIsNone(controller.admit(scheduler))
        self.assertEqual(controller.alloc_failures, 1)
        self.assertIsNone(controller.parked)
        self.assertTrue(scheduler.waiting)

    def test_abort_is_none_when_nothing_was_appended(self):
        pool, manager, request, scheduler = self._scheduler()
        provider = am.install_admission_metadata(scheduler)
        provider.begin(scheduler, request)
        self.assertIsNone(provider.abort())
        self.assertEqual(provider.aborts, 1)

    def test_abort_without_a_scope_is_total_and_journals_the_error(self):
        pool, manager, request, scheduler = self._scheduler()
        provider = am.install_admission_metadata(scheduler)
        # A provider-side journal failure must never strand the request: the
        # controller still receives a (empty, orphan) record and quarantines.
        capture = provider.abort()
        self.assertTrue(capture.orphan)
        self.assertIn('not begun', capture.error)
        self.assertEqual(capture.details, {
            'new_block_ids_to_zero': (), 'cow_pairs': (),
            'ngram_context': (), 'unpublished_blocks': ()})

    def test_stale_request_incarnation_is_refused(self):
        pool, manager, request, scheduler = self._scheduler()
        provider = am.install_admission_metadata(scheduler)
        provider.begin(scheduler, request)
        with self.assertRaises(E3ContractError):
            provider.capture(Request(uid=request.request_id), object(), 0, 4)
        # The failed take consumed the scope; a new allocation must begin again.
        with self.assertRaises(E3ContractError):
            provider.capture(request, object(), 0, 4)

    def test_connector_boundary_offer_is_refused_with_the_schema_reason(self):
        pool = FixturePool()
        mamba = FixtureMambaGroup(pool, group_id=0)
        manager = FixtureManager(pool, [mamba])
        request = Request(tokens=12, computed=6)
        source, first = pool.precreate(2, ref=1)
        mamba.req_to_blocks[UID] = [first, source]
        mamba.num_cached_block[UID] = 1
        mamba._allocated_block_reqs.add(UID)
        mamba._partial_hit_reqs[UID] = (1, source)
        mamba.producer_tails[UID] = (source, 6)
        source.set_block_hash(BlockHashWithGroupId(b'h'), 6)
        scheduler = FixtureScheduler(manager, [request])
        scheduler.connector = object()
        provider = am.install_admission_metadata(scheduler)
        provider.begin(scheduler, request)
        blocks = manager.allocate_slots(request, 4)
        with self.assertRaises(E3ContractError) as caught:
            provider.capture(request, blocks, 6, 10)
        self.assertIn('four-field', str(caught.exception))
        self.assertIn('boundary_tokens', str(caught.exception))
        # Validation happened before the permanent drain.
        self.assertEqual(len(mamba._pending_cow_copies), 1)
        self.assertEqual(len(mamba._pending_boundary_state_offloads), 1)

    def test_duplicate_cow_destination_is_refused(self):
        pool = FixturePool()
        group = FixtureGroup(pool)
        manager = FixtureManager(pool, [group])
        request = Request()
        scheduler = FixtureScheduler(manager, [request])
        src, dst = pool.precreate(2, ref=1)
        provider = am.install_admission_metadata(scheduler)
        provider.begin(scheduler, request)
        group._pending_cow_copies.extend([(src, dst), (src, dst)])
        with self.assertRaises(E3ContractError):
            provider.capture(request, FixtureBlocks([[src]]), 0, 4)
        self.assertEqual(len(group._pending_cow_copies), 2)

    def test_null_cow_endpoint_is_refused(self):
        pool = FixturePool()
        group = FixtureGroup(pool)
        manager = FixtureManager(pool, [group])
        request = Request()
        scheduler = FixtureScheduler(manager, [request])
        src, = pool.precreate(1, ref=1)
        provider = am.install_admission_metadata(scheduler)
        provider.begin(scheduler, request)
        group._pending_cow_copies.append((src, pool.null))
        with self.assertRaises(E3ContractError):
            provider.capture(request, FixtureBlocks([[src]]), 0, 4)

    def test_install_is_refused_twice(self):
        pool, manager, request, scheduler = self._scheduler()
        am.install_admission_metadata(scheduler)
        with self.assertRaises(E3ContractError):
            am.install_admission_metadata(scheduler)

    def test_hisparse_host_groups_are_refused_before_allocation(self):
        pool = FixturePool()
        group = FixtureHiSparseGroup(pool)
        manager = FixtureManager(pool, [group])
        scheduler = FixtureScheduler(manager, [Request()])
        with self.assertRaises(E3ContractError) as caught:
            am.install_admission_metadata(scheduler)
        self.assertIn('HiSparse', str(caught.exception))
        self.assertEqual(manager.alloc_calls, [])


# ---------------------------------------------------------------------------
# 3. CoW ownership and retirement
# ---------------------------------------------------------------------------

class CowOwnershipTests(unittest.TestCase):
    def setUp(self):
        self.pool = FixturePool()
        self.group = FixtureGroup(self.pool)
        self.manager = FixtureManager(self.pool, [self.group])
        self.request = Request(tokens=12)
        self.scheduler = FixtureScheduler(self.manager, [self.request])
        self.controller = make_controller(self.scheduler)

    def test_cold_chunk_has_no_cow_and_retires_nothing(self):
        record = self.controller.admit(self.scheduler)
        capture = self.controller.parked.capture
        self.assertEqual(record.new_block_ids_to_zero, (1,))
        self.assertEqual(record.cow_pairs, ())
        self.assertEqual(capture.unpublished, (1,))
        finish(self.controller, record)
        self.assertEqual(self.controller.retained_releases, 0)
        self.assertEqual(self.pool.freed, [])
        # Row references are still owned by the request.
        self.assertEqual(self.group.req_to_blocks[UID][0].ref_cnt, 1)
        free_request_rows(self.manager, UID)
        self.assertEqual(self.group.req_to_blocks.get(UID), None)
        self.assertEqual(self.pool.freed, [1])

    def test_partial_prefix_hit_retains_real_endpoints_and_retires_once(self):
        hit_tail, source = self.pool.precreate(2, ref=0)
        self.manager.hit_groups = ([hit_tail, source],)
        self.manager.hit_tokens = 6
        source.set_block_hash(BlockHashWithGroupId(b'src'), 6)
        record = self.controller.admit(self.scheduler)
        capture = self.controller.parked.capture
        cow_id = record.cow_pairs[0][1]
        cow = self.pool.block(cow_id)
        # (source, destination) in order; the endpoint objects are retained.
        self.assertEqual(record.cow_pairs, ((source.block_id, cow_id),))
        self.assertIs(capture.retained_blocks[0], source)
        self.assertIs(capture.retained_blocks[1], cow)
        self.assertIs(self.group.req_to_blocks[UID][1], cow)
        self.assertIsNot(self.group.req_to_blocks[UID][1], source)
        # Copy semantics: source keeps its hit ref; the cow takes one extra.
        self.assertEqual(source.ref_cnt, 1)
        self.assertEqual(cow.ref_cnt, 2)
        # Zeroing covers the cow and the new block, not the cached prefix hit.
        self.assertEqual(record.new_block_ids_to_zero, (cow_id, cow_id + 1))
        # Publication: the CoW destination replaced the hit inside the row and
        # the source is not a publication obligation.
        self.assertEqual(capture.unpublished, (cow_id, cow_id + 1))
        self.assertNotIn(source.block_id, capture.unpublished)
        # The lease keeps the copy source alive for the async copy.
        self.assertIn(source.block_id, self.controller.leases.held_blocks())

        finish(self.controller, record)
        self.assertEqual(capture.released, True)
        self.assertEqual(source.ref_cnt, 0)   # hit ref returned once
        self.assertEqual(cow.ref_cnt, 1)      # row keeps its reference
        self.assertEqual(self.controller.retained_releases, 1)
        # The stock retention call went through the deferred-free route.
        self.assertEqual(self.scheduler.retained_frees,
                         [(self.scheduler.sched_step_seq,
                           (source, cow))])
        with self.assertRaises(E3ContractError):
            capture.release_retained(self.scheduler)
        free_request_rows(self.manager, UID)
        self.assertEqual(source.ref_cnt, 0)
        self.assertEqual(cow.ref_cnt, 0)

    def test_subsequent_chunks_add_no_cow_and_publish_only_new_pages(self):
        record = self.controller.admit(self.scheduler)
        finish(self.controller, record)
        self.assertEqual(record.new_block_ids_to_zero, (1,))

        record = self.controller.admit(self.scheduler)
        finish(self.controller, record)
        self.assertEqual(record.transaction.chunk_start, 4)
        self.assertEqual(record.transaction.chunk_end, 8)
        self.assertEqual(record.new_block_ids_to_zero, (2,))
        self.assertEqual(record.cow_pairs, ())
        self.assertEqual(self.controller.parked, None)

        record = self.controller.admit(self.scheduler)
        finish(self.controller, record)
        self.assertEqual(record.transaction.chunk_start, 8)
        self.assertEqual(record.transaction.chunk_end, 11)
        self.assertEqual(record.new_block_ids_to_zero, (3,))
        # Terminal chunk drops the identity owner: no stale generation.
        self.assertEqual(self.controller._request_identities, {})
        free_request_rows(self.manager, UID)
        self.assertEqual(self.pool.freed, [1, 2, 3])

    def test_non_aligned_chunks_do_not_re_own_the_previous_partial_tail(self):
        scheduler = FixtureScheduler(self.manager, [self.request])
        controller = make_controller(scheduler, chunk=3)
        observed = []
        for start, end in ((0, 3), (3, 6), (6, 9), (9, 11)):
            record = controller.admit(scheduler)
            capture = controller.parked.capture
            self.assertEqual(record.transaction.chunk_start, start)
            self.assertEqual(record.transaction.chunk_end, end)
            observed.append(capture.unpublished)
            finish(controller, record)
        # Chunks 2 and 3 own the pages they allocate; chunk 4 extends a block
        # already published by chunk 3 and must not reserve it again.
        self.assertEqual(observed, [(1,), (2,), (3,), ()])
        free_request_rows(self.manager, UID)
        self.assertEqual(self.pool.freed, [1, 2, 3])

    def test_warm_full_block_hit_does_not_copy_or_lease_the_null_block(self):
        hit, = self.pool.precreate(1, ref=0)
        hit.set_block_hash(BlockHashWithGroupId(b'full'), 4)
        self.manager.hit_groups = ([hit],)
        self.manager.hit_tokens = 4
        record = self.controller.admit(self.scheduler)
        capture = self.controller.parked.capture
        self.assertEqual(record.transaction.chunk_start, 4)
        self.assertEqual(record.cow_pairs, ())
        self.assertEqual(capture.retained_blocks, ())
        self.assertNotIn(0, record.flat_blocks())
        self.assertNotIn(0, record.new_block_ids_to_zero)
        self.assertNotIn(0, capture.unpublished)
        finish(self.controller, record)
        self.assertEqual(hit.ref_cnt, 1)   # still owned by the row
        free_request_rows(self.manager, UID)
        self.assertEqual(hit.ref_cnt, 0)

    def test_null_rows_and_empty_groups_are_preserved_and_never_leased(self):
        pool = FixturePool()
        attention = FixtureGroup(pool, group_id=0)
        empty = FixtureEmptyGroup(pool, group_id=1)
        mamba = FixtureMambaGroup(pool, group_id=2)
        manager = FixtureManager(pool, [attention, empty, mamba])
        request = Request(tokens=12, computed=6)
        first, source = pool.precreate(2, ref=1)
        mamba.req_to_blocks[UID] = [first, source]
        mamba.num_cached_block[UID] = 1
        mamba._allocated_block_reqs.add(UID)
        mamba._partial_hit_reqs[UID] = (1, source)
        source.set_block_hash(BlockHashWithGroupId(b'h'), 6)
        scheduler = FixtureScheduler(manager, [request])
        controller = make_controller(scheduler)
        record = controller.admit(scheduler)
        capture = controller.parked.capture
        self.assertEqual(len(record.block_tables), 3)
        self.assertEqual(record.block_tables[1], ())
        # Mamba head state freed by remove_skipped_blocks became a null.
        self.assertEqual(record.block_tables[2][0], 0)
        self.assertNotIn(0, record.new_block_ids_to_zero)
        self.assertNotIn(0, capture.unpublished)
        self.assertNotIn(0, [b for pair in record.cow_pairs for b in pair])
        self.assertNotIn(0, controller.leases.held_blocks())
        self.assertIn(first.block_id, pool.freed)
        finish(controller, record)
        free_request_rows(manager, UID)

    def test_never_opened_cancellation_retires_the_retention_once(self):
        hit_tail, source = self.pool.precreate(2, ref=0)
        self.manager.hit_groups = ([hit_tail, source],)
        self.manager.hit_tokens = 6
        record = self.controller.admit(self.scheduler)
        capture = self.controller.parked.capture
        returned = self.controller.cancel(record.transaction, never_submitted=True)
        self.assertIs(returned, self.request)
        self.assertEqual(capture.released, True)
        self.assertEqual(self.controller.retained_releases, 1)
        self.assertEqual(source.ref_cnt, 0)
        self.assertEqual(self.controller.leases.held_blocks(), ())
        self.assertEqual(self.controller._request_identities, {})
        with self.assertRaises(E3ContractError):
            capture.release_retained(self.scheduler)

    def test_open_before_first_submit_retains_ownership(self):
        hit_tail, source = self.pool.precreate(2, ref=0)
        self.manager.hit_groups = ([hit_tail, source],)
        self.manager.hit_tokens = 6
        record = self.controller.admit(self.scheduler)
        capture = self.controller.parked.capture
        transaction = record.transaction
        self.controller.bind(make_ticket(transaction, 0, 0,
                                         DependencyRole.PRODUCER))
        # An opened worker is not a local cancel: no proof means no release.
        with self.assertRaises(E3ContractError):
            self.controller.cancel(transaction, never_submitted=True)
        proof = make_proof(transaction, 0, 0, DependencyRole.PRODUCER,
                           CompletionKind.GPU_EVENT)
        with self.assertRaises(E3Pending):
            self.controller.cancel(transaction, proof=proof)
        self.assertFalse(capture.released)
        self.assertEqual(source.ref_cnt, 1)
        held = self.controller.leases.held_blocks()
        self.assertIn(source.block_id, held)
        self.assertIn(record.cow_pairs[0][1], held)
        # The full worker-release barrier is the only exit.
        for report in reports_for(transaction, self.controller.ranks,
                                  record.state_epoch):
            self.controller.note_report(report)
            self.controller.note_worker_release(w.ReleaseAck(
                transaction, report.worker_rank, record.state_epoch,
                report.completed_sequence, released=True,
                request_closed=record.terminal))
        self.controller.cancel(transaction, proof=proof)
        self.assertTrue(capture.released)
        self.assertEqual(self.controller.retained_releases, 1)


# ---------------------------------------------------------------------------
# 4. Mamba align, boundary work and allocation failure
# ---------------------------------------------------------------------------

class MambaBoundaryTests(unittest.TestCase):
    def _running_setup(self):
        pool = FixturePool()
        group = FixtureMambaGroup(pool, group_id=0)
        manager = FixtureManager(pool, [group])
        request = Request(tokens=12, computed=6)
        first, source = pool.precreate(2, ref=1)
        group.req_to_blocks[UID] = [first, source]
        group.num_cached_block[UID] = 1
        group._allocated_block_reqs.add(UID)
        group._partial_hit_reqs[UID] = (1, source)
        group.producer_tails[UID] = (source, 6)
        source.set_block_hash(BlockHashWithGroupId(b'h'), 6)
        scheduler = FixtureScheduler(manager, [request])
        return pool, group, manager, request, scheduler

    def test_running_align_relocates_the_hash_and_retains_the_copy(self):
        pool, group, manager, request, scheduler = self._running_setup()
        controller = make_controller(scheduler)
        record = controller.admit(scheduler)
        capture = controller.parked.capture
        source_id, cow_id = record.cow_pairs[0]
        source = pool.block(source_id)
        cow = pool.block(cow_id)
        self.assertEqual(pool.hash_moves, [(source_id, cow_id, 6)])
        self.assertIsNone(source.block_hash)
        self.assertEqual(cow.block_hash, b'h')
        self.assertEqual(cow.block_hash_num_tokens, 6)
        # Destination leaves the returned row, but it is a publication duty.
        self.assertNotIn(cow.block_id, record.block_tables[0])
        self.assertEqual(record.cow_pairs, ((source.block_id, cow.block_id),))
        self.assertIn(cow.block_id, capture.unpublished)
        # The row still holds the (hash-migrated) source, but publication moved
        # to the destination: hash state is not ownership evidence.
        self.assertNotIn(source.block_id, capture.unpublished)
        self.assertEqual(capture.unpublished,
                         (record.block_tables[0][-1], cow.block_id))
        self.assertEqual(record.new_block_ids_to_zero, ())
        # Boundary offer is audited, not invented, and left for the stock drain.
        self.assertEqual(capture.evidence['boundary_offloads'],
                         ((0, cow.block_id, 6),))
        self.assertEqual(group._pending_boundary_state_offloads,
                         [(UID, 0, cow, 6)])
        self.assertEqual(source.ref_cnt, 2)
        self.assertEqual(cow.ref_cnt, 1)
        finish(controller, record)
        self.assertEqual(source.ref_cnt, 1)
        self.assertEqual(cow.ref_cnt, 0)
        self.assertEqual(controller.retained_releases, 1)
        # The stock end-of-step drain consumes the offer exactly once.
        drained = group.take_pending_boundary_state_offloads()
        self.assertEqual(drained, [(UID, 0, cow, 6)])
        self.assertEqual(group.take_pending_boundary_state_offloads(), [])
        free_request_rows(manager, UID)

    def test_running_align_allocation_failure_retains_every_obligation(self):
        pool = FixturePool()
        group = FixtureMambaGroup(pool, group_id=0)
        group.fail_after_append = True
        manager = FixtureManager(pool, [group])
        request = Request(tokens=12, computed=6)
        first, source = pool.precreate(2, ref=1)
        group.req_to_blocks[UID] = [first, source]
        group.num_cached_block[UID] = 1
        group._allocated_block_reqs.add(UID)
        group._partial_hit_reqs[UID] = (1, source)
        group.producer_tails[UID] = (source, 6)
        source.set_block_hash(BlockHashWithGroupId(b'h'), 6)
        scheduler = FixtureScheduler(manager, [request])
        controller = make_controller(scheduler)
        with self.assertRaises(RuntimeError):
            controller.admit(scheduler)
        capture = controller.parked.capture
        self.assertTrue(capture.orphan)
        source_id, cow_id = capture.cow_pairs[0]
        self.assertEqual(source.ref_cnt, 2)
        self.assertEqual(pool.block(cow_id).ref_cnt, 1)
        self.assertEqual(len(capture.retained_blocks), 2)
        # Both queue entries survive for the engine/abort path.
        self.assertEqual(len(group._pending_cow_copies), 1)
        self.assertEqual(len(group._pending_boundary_state_offloads), 1)
        self.assertEqual(len(capture.unpublished), 3)   # source, state block, cow
        self.assertEqual(controller.leases.held_blocks(), ())
        self.assertFalse(scheduler.waiting)

    def test_not_running_mamba_hit_uses_the_local_cow_path(self):
        pool = FixturePool()
        group = FixtureMambaGroup(pool, group_id=0)
        manager = FixtureManager(pool, [group])
        request = Request(tokens=12)
        tail, source = pool.precreate(2, ref=0)
        manager.hit_groups = ([tail, source],)
        manager.hit_tokens = 6
        source.set_block_hash(BlockHashWithGroupId(b'h'), 6)
        scheduler = FixtureScheduler(manager, [request])
        controller = make_controller(scheduler)
        record = controller.admit(scheduler)
        capture = controller.parked.capture
        cow_id = record.cow_pairs[0][1]
        self.assertIs(capture.retained_blocks[0], source)
        self.assertIs(group.req_to_blocks[UID][1], pool.created[cow_id - 1])
        self.assertEqual(capture.evidence['boundary_offloads'], ())
        self.assertEqual(group._pending_boundary_state_offloads, [])
        finish(controller, record)
        self.assertEqual(controller.retained_releases, 1)
        free_request_rows(manager, UID)


# ---------------------------------------------------------------------------
# 5. PLE ngram context
# ---------------------------------------------------------------------------

class PleContextTests(unittest.TestCase):
    def _admit(self, *, tokens=12, hit=None, hit_tokens=0, ple=(2,),
               ngram=3, eos=99, chunk=4):
        pool = FixturePool()
        group = FixtureGroup(pool)
        manager = FixtureManager(pool, [group], hit_groups=hit,
                                 hit_tokens=hit_tokens)
        request = Request(tokens=tokens)
        scheduler = FixtureScheduler(
            manager, [request],
            text=text_config(ple=ple, ngram=ngram, eos=eos))
        controller = make_controller(scheduler, chunk=chunk)
        record = controller.admit(scheduler)
        return controller, record, scheduler

    def test_cold_start_is_left_padded_with_the_configured_eos(self):
        controller, record, _ = self._admit(ple=(2,), ngram=3, eos=99)
        self.assertEqual(record.ngram_context, (99, 99))
        self.assertEqual(controller.parked.capture.evidence['ngram_padding'], 2)

    def test_warm_start_uses_tokens_immediately_before_the_chunk(self):
        pool = FixturePool()
        group = FixtureGroup(pool)
        hit_tail, = pool.precreate(1, ref=0)
        manager = FixtureManager(pool, [group], hit_groups=([hit_tail],),
                                 hit_tokens=4)
        request = Request(tokens=12)
        scheduler = FixtureScheduler(manager, [request])
        controller = make_controller(scheduler)
        record = controller.admit(scheduler)
        self.assertEqual(record.transaction.chunk_start, 4)
        self.assertEqual(record.ngram_context, (2, 3))
        self.assertEqual(controller.parked.capture.evidence['ngram_padding'], 0)

    def test_each_chunk_uses_its_own_preceding_window(self):
        pool = FixturePool()
        group = FixtureGroup(pool)
        manager = FixtureManager(pool, [group])
        request = Request(tokens=12)
        scheduler = FixtureScheduler(manager, [request])
        controller = make_controller(scheduler)
        record = controller.admit(scheduler)
        self.assertEqual(record.transaction.chunk_start, 0)
        finish(controller, record)
        record = controller.admit(scheduler)
        self.assertEqual(record.transaction.chunk_start, 4)
        self.assertEqual(record.ngram_context, (2, 3))
        finish(controller, record)
        record = controller.admit(scheduler)
        self.assertEqual(record.transaction.chunk_start, 8)
        self.assertEqual(record.ngram_context, (6, 7))

    def test_configured_ngram_size_controls_the_length(self):
        controller, record, _ = self._admit(ngram=4, eos=7)
        self.assertEqual(record.ngram_context, (7, 7, 7))

    def test_declared_empty_ple_path_is_the_only_empty_context(self):
        controller, record, _ = self._admit(ple=())
        self.assertEqual(record.ngram_context, ())
        self.assertEqual(controller.parked.capture.evidence['ngram_context_length'], 0)

    def test_missing_ple_declaration_is_refused(self):
        pool = FixturePool()
        manager = FixtureManager(pool, [FixtureGroup(pool)])
        config = type('TextConfig', (), {'ngram_size': 3, 'eos_token_id': 7})()
        scheduler = FixtureScheduler(manager, [Request()], text=config)
        with self.assertRaises(E3ContractError):
            am.install_admission_metadata(scheduler)


# ---------------------------------------------------------------------------
# 6. Post-allocation failure boundaries
# ---------------------------------------------------------------------------

class PostAllocationFailureTests(unittest.TestCase):
    def _fixture(self):
        pool = FixturePool()
        group = FixtureGroup(pool)
        manager = FixtureManager(pool, [group])
        request = Request(tokens=12)
        scheduler = FixtureScheduler(manager, [request])
        controller = make_controller(scheduler)
        return pool, group, manager, request, scheduler, controller

    def test_metadata_validation_failure_leaves_queues_untouched(self):
        pool, group, manager, request, scheduler, controller = self._fixture()
        provider = scheduler._flash_s5_admission_metadata
        original = provider.capture

        def broken(request_arg, blocks, start, end):
            capture = original(request_arg, blocks, start, end)
            # Simulate a post-capture contract violation without losing the
            # journal: zero ids outside the returned rows.
            capture.zero_ids = (999,)
            return capture

        provider.capture = broken
        with self.assertRaises(E3ContractError):
            controller.admit(scheduler)
        self.assertTrue(controller.parked.failed)
        capture = controller.parked.capture
        self.assertIsNotNone(capture)
        self.assertFalse(capture.released)
        # The provider had already taken the suffix; nothing was restored, so
        # the engine still owns every side effect it recorded.
        self.assertEqual(group.new_block_ids, [])
        self.assertFalse(scheduler.waiting)
        self.assertEqual(controller.leases.held_blocks(), ())

    def _inject_after_allocation(self, manager, group, pairs):
        """Allocation appends extra (invalid) copy pairs: a capture refusal."""
        original = manager.allocate_slots

        def allocate(*args, **kwargs):
            blocks = original(*args, **kwargs)
            group._pending_cow_copies.extend(pairs)
            return blocks

        manager.allocate_slots = allocate

    def test_capture_refusal_through_the_controller_journals_the_suffix(self):
        pool, group, manager, request, scheduler, controller = self._fixture()
        src, dst = pool.precreate(2, ref=1)
        self._inject_after_allocation(manager, group, [(src, dst), (src, dst)])
        with self.assertRaises(E3ContractError) as caught:
            controller.admit(scheduler)
        self.assertIn('destination', str(caught.exception))
        parked = controller.parked
        self.assertTrue(parked.failed)
        self.assertFalse(scheduler.waiting)
        capture = parked.capture
        # Not a silent drop: the refused pairs and zero work are journaled.
        self.assertIsNotNone(capture)
        self.assertTrue(capture.orphan)
        self.assertIn('share a destination', capture.error)
        self.assertEqual(capture.zero_ids, tuple(group.new_block_ids))
        self.assertTrue(capture.zero_ids)
        self.assertEqual(capture.cow_pairs, ((src.block_id, dst.block_id),) * 2)
        self.assertEqual(capture.retained_blocks, (src, dst, src, dst))
        # The queue suffix was never truncated: the engine still owns it.
        self.assertEqual(len(group._pending_cow_copies), 2)
        with self.assertRaises(E3ContractError):
            capture.release_retained(scheduler)
        self.assertEqual((src.ref_cnt, dst.ref_cnt), (1, 1))
        self.assertIsNone(controller.admit(scheduler))
        self.assertIsNone(scheduler._flash_s5_admission_metadata._scope)

    def test_capture_refusal_on_a_null_endpoint_keeps_the_raw_pair(self):
        pool, group, manager, request, scheduler, controller = self._fixture()
        src, = pool.precreate(1, ref=1)
        self._inject_after_allocation(manager, group, [(src, pool.null)])
        with self.assertRaises(E3ContractError):
            controller.admit(scheduler)
        capture = controller.parked.capture
        self.assertTrue(capture.orphan)
        self.assertIn('null block', capture.error)
        self.assertEqual(capture.cow_pairs, ((src.block_id, 0),))
        self.assertIs(capture.retained_blocks[1], pool.null)
        self.assertEqual(len(group._pending_cow_copies), 1)

    def test_frontier_reuse_failure_retains_the_journal(self):
        pool, group, manager, request, scheduler, controller = self._fixture()
        # Another transaction already published the id this allocation reuses.
        blocker = type('T', (), {})()
        controller.frontier._published = {blocker: (1,)}
        with self.assertRaises(ah.HandoffError):
            controller.admit(scheduler)
        self.assertTrue(controller.parked.failed)
        capture = controller.parked.capture
        self.assertFalse(capture.released)
        self.assertEqual(controller.retained_releases, 0)
        self.assertEqual(controller.leases.held_blocks(), (1,))

    def test_incomplete_proofs_do_not_release_retention(self):
        pool, group, manager, request, scheduler, controller = self._fixture()
        record = controller.admit(scheduler)
        transaction = record.transaction
        controller.bind(make_ticket(transaction, 0, 0, DependencyRole.PRODUCER))
        proof = make_proof(transaction, 0, 0, DependencyRole.PRODUCER,
                           CompletionKind.GPU_EVENT)
        with self.assertRaises(E3Pending):
            controller.commit(transaction, proof)
        self.assertFalse(controller.parked.capture.released)
        self.assertEqual(controller.retained_releases, 0)

    def test_repeated_uid_reuse_gets_a_new_generation_and_bounded_history(self):
        pool, group, manager, request, scheduler, controller = self._fixture()
        controller.max_history = 1
        seen = []
        for _ in range(3):
            # A new request object with the same uid is a new incarnation.
            request = Request(uid=UID, tokens=12)
            scheduler.waiting = Queue([request])
            controller.parked = None
            request.num_computed_tokens = 0
            scheduler.running = []
            record = controller.admit(scheduler)
            seen.append(record.transaction.request_generation)
            self.assertEqual(record.transaction.request_uid, UID)
            finish(controller, record)
            free_request_rows(manager, UID)
        self.assertEqual(len(set(seen)), 3)
        self.assertEqual(len(controller._committed), 1)
        # No capture or endpoint object is retained by the bounded history.
        record = next(iter(controller._committed.values()))
        self.assertNotIn('capture', record)
        self.assertEqual(controller.parked, None)


# ---------------------------------------------------------------------------
# 7. Composed first-chunk consumer
# ---------------------------------------------------------------------------

class FirstChunkConsumer:
    """Stand-in for the bridge's first-chunk path.

    It accepts only the immutable wire record: zero ids and CoW ids are handed
    to device-side operations; ngram context is handed to the PLE path. It
    deliberately has no register step: the real bridge must register the
    operation on the worker. A valid provider cannot supply that.
    """

    def __init__(self, record):
        self.record = record
        self.zero_ops = []
        self.copy_ops = []
        self.ple_windows = []
        self.registered = False

    def consume(self, *, register_path):
        for block_id in self.record.new_block_ids_to_zero:
            self.zero_ops.append(('zero', block_id))
        for source, destination in self.record.cow_pairs:
            self.copy_ops.append(('copy', source, destination))
        self.ple_windows.append(tuple(self.record.ngram_context))
        if not register_path:
            raise RuntimeError('worker register path is not implemented')
        self.registered = True
        return self


class StartupSeamTests(unittest.TestCase):
    """The provider is constructed by enabled engine startup, not by tests."""

    def test_enabled_startup_installs_the_provider_before_the_driver(self):
        pool = FixturePool()
        manager = FixtureManager(pool, [FixtureGroup(pool)])
        scheduler = FixtureScheduler(manager, [Request()])
        executor = types.SimpleNamespace(response_mqs={0: object(), 1: object()})
        engine = types.SimpleNamespace(model_executor=executor, scheduler=scheduler)
        leases = sh.SchedulerLeases()
        frontier = ah.CommittedFrontier()
        observed = {}

        class FakeComposition(ah.Composition):
            def __init__(self):
                self.scheduler_leases = leases
                self.frontier = frontier

            def attach(self, **kwargs):
                observed['attached'] = kwargs
                return self

        composition = FakeComposition()

        class FakeDriver:
            def __init__(self, *args, **kwargs):
                target = kwargs.get('scheduler')
                observed['provider_at_driver_construction'] = getattr(
                    target, '_flash_s5_admission_metadata', None)
                self.installed = False

            def install(self):
                self.installed = True
                return self

        with mock.patch.object(su, 'require_native_ready', lambda: None), \
                mock.patch.object(su, '_native_execution_shape',
                                  lambda *a, **k: (2, 1, ((0, 0), (1, 1)))), \
                mock.patch.object(su, 'build_composition',
                                  lambda *a, **k: composition), \
                mock.patch.object(eh, 'install', lambda *a, **k: object()), \
                mock.patch.object(nb, 'RemoteCompletionBackend',
                                  lambda config, **kwargs: object()), \
                mock.patch.object(nr, 'response_success', lambda: object()), \
                mock.patch.object(edrv, 'NativeSegmentDriver', FakeDriver):
            result = su.install_engine(engine, env={'FLASH_S5_OVERLAP': '1'})

        self.assertIs(result, composition)
        self.assertIsInstance(scheduler._flash_s5_admission_metadata,
                              am.NativeAdmissionMetadata)
        self.assertIs(scheduler._flash_s5_admission_metadata._scheduler, scheduler)
        self.assertIsNotNone(scheduler._flash_s5_admission)
        # The driver is constructed only after the provider is installed.
        self.assertIs(observed['provider_at_driver_construction'],
                      scheduler._flash_s5_admission_metadata)
        self.assertTrue(engine._flash_s5_driver.installed)
        self.assertEqual(observed['attached'], {
            'engine': engine, 'scheduler': scheduler, 'executor': executor})


class ComposedConsumerTests(unittest.TestCase):
    def test_valid_metadata_does_not_conceal_the_missing_register_path(self):
        pool = FixturePool()
        group = FixtureGroup(pool)
        tail, source = pool.precreate(2, ref=0)
        manager = FixtureManager(pool, [group], hit_groups=([tail, source],),
                                 hit_tokens=6)
        source.set_block_hash(BlockHashWithGroupId(b'src'), 6)
        request = Request(tokens=12)
        scheduler = FixtureScheduler(manager, [request])
        controller = make_controller(scheduler)
        record = controller.admit(scheduler)
        self.assertEqual(len(record.new_block_ids_to_zero), 2)
        self.assertEqual(len(record.cow_pairs), 1)
        self.assertEqual(len(record.ngram_context), 2)

        consumer = FirstChunkConsumer(record)
        with self.assertRaises(RuntimeError):
            consumer.consume(register_path=False)
        self.assertEqual(consumer.zero_ops,
                         [('zero', record.new_block_ids_to_zero[0]),
                          ('zero', record.new_block_ids_to_zero[1])])
        self.assertEqual(consumer.copy_ops,
                         [('copy', record.cow_pairs[0][0],
                           record.cow_pairs[0][1])])
        self.assertEqual(consumer.ple_windows, [record.ngram_context])

        consumer.consume(register_path=True)
        self.assertTrue(consumer.registered)
        # The consumer reads the record only; ownership stays in the journal.
        capture = controller.parked.capture
        self.assertEqual(len(capture.retained_blocks), 2)
        self.assertFalse(capture.released)
        finish(controller, record)
        self.assertTrue(capture.released)
        free_request_rows(manager, UID)


if __name__ == '__main__':
    unittest.main()
