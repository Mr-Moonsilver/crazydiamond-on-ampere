"""Serial E2 metadata binding, CPU-only and deliberately independent of vLLM.

The engine hook supplies allocation events, exact worker table writes and a
completed producer fence. This module does NOT pin scheduler block references.
Safety requires ordered worker RPCs, no external KV writers, and every arena
consumer/DMA drained before the RPC returns. E3 cannot inherit that assumption.
Never log Frame/RequestRow: their request IDs are internal routing metadata.
"""
from dataclasses import dataclass
import json

from .arena_contract import Arena, Owner
from .retention_contract import ContractError, Pending, Pool, RequestKey


def natural(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ContractError('invalid ' + name)
    return value


def freeze_tables(tables):
    return tuple(tuple(natural(b, 'physical block') for b in row) for row in tables)


def table_delta(group, worker, scheduler):
    """Metadata only (q16f, 25 Oct): lengths, differing positions, scheduler
    NULL entries among them and the first differing index. No block IDs."""
    n = min(len(worker), len(scheduler))
    differ = [i for i in range(n) if worker[i] != scheduler[i]]
    return dict(group=group, worker=len(worker), scheduler=len(scheduler),
                differ=len(differ) + abs(len(worker) - len(scheduler)),
                scheduler_null=sum(1 for i in differ if scheduler[i] == 0),
                first=differ[0] if differ else n)


@dataclass(frozen=True, repr=False)
class RequestRow:
    request_id: str
    tables: tuple
    computed: int
    scheduled: int


@dataclass(frozen=True, repr=False)
class Frame:
    epoch: int
    sequence: int
    allocations: tuple  # (global physical block ID, allocation incarnation)
    rows: tuple
    restarted: tuple
    retired: tuple
    copies: tuple       # (source physical ID, destination physical ID)
    copy_destinations: tuple  # per-group IDs selected by runner's CoW filter


class AllocationJournal:
    """Observe get_new_blocks, NOT zero-block lists or prefix-cache touches.

    Sized by the fixed pool, not by request lifetime. Resetting prefix hashes
    does not reset generations; only a new worker/pool creates a new epoch.
    """
    def __init__(self, epoch, num_blocks):
        self.epoch = natural(epoch, 'pool epoch', 1)
        self.num_blocks = natural(num_blocks, 'pool size', 2)
        self.generations = [0] * num_blocks
        self.pending = {}
        self.sequence = 0

    def allocated(self, block_ids):
        ids = tuple(block_ids)
        if (len(set(ids)) != len(ids) or
                any(type(b) is not int or not 0 < b < self.num_blocks for b in ids)):
            raise ContractError('invalid allocated blocks')
        for block in ids:
            self.generations[block] += 1
            self.pending[block] = self.generations[block]

    def frame(self, output, get_block_ids):
        """Call after all scheduler table/CoW edits, BEFORE dispatch.

        get_block_ids is the real kv_cache_manager method. The worker rebuilds
        its table independently from the actual append_block_ids calls.
        Frozen copies prevent later schedules from mutating this RPC's view.
        """
        cached = output.scheduled_cached_reqs
        if not (len(cached.req_ids) == len(cached.num_computed_tokens) == len(cached.new_block_ids)):
            raise ContractError('cached request arrays disagree')
        starts = dict(zip(cached.req_ids, cached.num_computed_tokens))
        starts.update((r.req_id, r.num_computed_tokens) for r in output.scheduled_new_reqs)
        rows = []
        for rid, count in output.num_scheduled_tokens.items():
            tables = freeze_tables(get_block_ids(rid))
            if any(b >= self.num_blocks or (b and not self.generations[b]) for row in tables for b in row):
                raise ContractError('table refers to unobserved allocation')
            rows.append(RequestRow(rid, tables, natural(starts[rid], 'computed tokens'),
                                   natural(count, 'scheduled tokens', 1)))
        # Match _qsa_copy_host_pages exactly: new + appended + replacement
        # destinations, not all active table IDs and not new_block_ids_to_zero.
        changes = [r.block_ids for r in output.scheduled_new_reqs]
        changes += [b for b in cached.new_block_ids if b is not None]
        changes += list((output.block_table_updates or {}).values())
        groups = len(rows[0].tables) if rows else (len(changes[0]) if changes else 0)
        if any(len(b) != groups for b in changes) or any(len(r.tables) != groups for r in rows):
            raise ContractError('cache-group count changed')
        destinations = [set() for _ in range(groups)]
        for change in changes:
            for g, row in enumerate(freeze_tables(change)):
                destinations[g].update(row)
        copies = tuple((c.src_block_id, c.dst_block_id) for c in (output.kv_cache_block_copies or ()))
        if any(type(b) is not int or not 0 < b < self.num_blocks or not self.generations[b]
               for pair in copies for b in pair):
            raise ContractError('CoW endpoint lacks allocation provenance')
        self.sequence += 1
        result = Frame(self.epoch, self.sequence, tuple(sorted(self.pending.items())), tuple(rows),
                       tuple(r.req_id for r in output.scheduled_new_reqs),
                       tuple(sorted(set(output.finished_req_ids) | set(output.preempted_req_ids or ()))),
                       copies, tuple(tuple(sorted(x)) for x in destinations))
        self.pending.clear()
        return result


class CpuTables:
    """Mirror real BlockTables.append_block_ids, including full replacements.

    Attach after the real method succeeds. An empty append is not a replace.
    The E2 candidate permits only QSA kernel block size == scheduler block size.
    Expansion is reproduced here so mismatched configuration cannot hide.
    """
    def __init__(self, blocks_per_kv_block, capacity):
        self.bpk = tuple(natural(x, 'block expansion', 1) for x in blocks_per_kv_block)
        self.capacity = natural(capacity, 'row capacity', 1)
        self.rows = {}

    def append(self, slot, blocks, overwrite):
        natural(slot, 'worker slot')
        if type(overwrite) is not bool or len(blocks) != len(self.bpk):
            raise ContractError('invalid block table update')
        old = self.rows.get(slot, tuple(() for _ in self.bpk))
        new = []
        for g, row in enumerate(freeze_tables(blocks)):
            expanded = tuple(b * self.bpk[g] + k for b in row for k in range(self.bpk[g]))
            result = expanded if overwrite else old[g] + expanded
            if len(result) > self.capacity:
                raise ContractError('CPU block table exceeds worker capacity')
            new.append(result)
        self.rows[slot] = tuple(new)


class WorkerMirror:
    """Serial RPC source versions and one-arena ownership, no new KV storage.

    Page.valid/version describe ENQUEUED stream effects. Page.writer blocks DMA
    until producer_ready observes an actual fence covering those effects. This
    also covers graph-replayed decode stores, which have no Python callback.
    Request retirement drops ownership, not cached bytes or allocation stamps.
    """
    def __init__(self, epoch, num_blocks, layer_groups, page_tokens=4048, arena_pages=23):
        self.pool = Pool(epoch, page_tokens)
        self.layer_groups = dict(layer_groups)
        self.arena = Arena(self.pool, tuple(self.layer_groups), arena_pages)
        self.num_blocks = natural(num_blocks, 'pool size', 2)
        self.incarnations = [0] * num_blocks
        self.requests = {}  # private req ID -> bounded slot RequestKey
        self.slot_generations = {}
        self.sequence = 0
        self.active = False
        self.failed = False
        self.current = {}
        self.cow_pairs = 0
        self.cow_views = 0
        self.frame_copies = (0, 0)
        self.ready_layers = set()
        # E2 reads only its QSA source groups. Other cache groups are owned by
        # the stock runner: the pinned Mamba align-mode manager retires older
        # state blocks to NULL in the scheduler's table only (the worker table
        # is append-only by design), so those groups legitimately diverge.
        self.source_groups = frozenset(self.layer_groups.values())
        self.table_divergences = 0
        self.divergent_groups = set()
        self.divergence_notes = []

    def _quiescent(self):
        if self.failed or self.arena.state == 'quarantined':
            raise Pending('worker requires recovery')
        if self.arena.state != 'idle' or self.pool.held_count():
            raise Pending('previous RPC still has DMA/consumer leases')

    def _page(self, layer, physical):
        if type(physical) is not int or not 0 < physical < self.num_blocks or not self.incarnations[physical]:
            raise ContractError('NULL or unknown source block')
        slot = (layer, physical)
        if slot not in self.pool.pages:
            self.pool.incarnations[slot] = self.incarnations[physical] - 1
            self.pool.allocate(layer, physical)
        page = self.pool.pages[slot]
        if page.key.incarnation != self.incarnations[physical]:
            raise ContractError('allocation incarnation disagrees')
        return page

    def _retire(self, rid):
        key = self.requests.pop(rid, None)
        if key is not None:
            self.arena.cancel(key)
            self.arena.retire(key)

    def enter(self, frame, slots, cpu_tables):
        """After finish/add/update_requests; before graph/eager forward.

        slots is the runner's actual req_id_to_index. Re-entry after any failed
        validation is forbidden: do not recover with stale or partial metadata.
        """
        self._quiescent()
        if self.active:
            raise Pending('RPC has not returned')
        try:
            self._enter(frame, slots, cpu_tables)
        except BaseException:
            self.failed = True
            raise

    def _enter(self, frame, slots, cpu_tables):
        if frame.epoch != self.pool.epoch or frame.sequence != self.sequence + 1:
            raise ContractError('missing, duplicate or foreign RPC frame')
        if any(g >= len(cpu_tables.bpk) or g < 0 or cpu_tables.bpk[g] != 1
               for g in self.layer_groups.values()):
            raise ContractError('E2 requires unexpanded local QSA groups')
        if len(set(slots.values())) != len(slots):
            raise ContractError('duplicate worker slot')
        for rid in tuple(self.requests):
            if rid not in slots or rid in frame.restarted or rid in frame.retired:
                self._retire(rid)
        seen = set()
        for physical, generation in frame.allocations:
            if (type(physical) is not int or not 0 < physical < self.num_blocks or physical in seen or
                    type(generation) is not int or generation <= self.incarnations[physical]):
                raise ContractError('invalid allocation generation delta')
            seen.add(physical)
            for layer in self.layer_groups:
                old = self.pool.pages.get((layer, physical))
                if old is not None:
                    # Prior stream writes may still be in flight, but no DMA
                    # can own them: this worker's previous RPC has returned.
                    # Keep no old-incarnation validity for a future source.
                    if old.holders or old.readers:
                        raise Pending('allocator reused an active DMA source')
                    del self.pool.pages[(layer, physical)]
            self.incarnations[physical] = generation
        self.current = {}
        for row in frame.rows:
            if row.request_id in self.current or row.request_id not in slots:
                raise ContractError('duplicate or missing scheduled request')
            slot = natural(slots[row.request_id], 'worker slot')
            mine = cpu_tables.rows.get(slot)
            if mine is None or len(mine) != len(row.tables):
                raise ContractError('worker CPU table differs from scheduler snapshot: %s vs %d groups'
                                    % ('no' if mine is None else len(mine), len(row.tables)))
            for g in sorted(self.source_groups):
                if g >= len(mine) or mine[g] != row.tables[g]:
                    raise ContractError('worker CPU table differs from scheduler snapshot in QSA group '
                                        + json.dumps(table_delta(g, mine[g] if g < len(mine) else (),
                                                                 row.tables[g] if g < len(row.tables) else ()),
                                                     sort_keys=True))
            deltas = [table_delta(g, mine[g], row.tables[g]) for g in range(len(mine))
                      if g not in self.source_groups and mine[g] != row.tables[g]]
            if deltas:
                self.table_divergences += 1
                new = [d for d in deltas if d['group'] not in self.divergent_groups]
                if new:
                    self.divergent_groups.update(d['group'] for d in new)
                    self.divergence_notes.append(dict(
                        sequence=frame.sequence, slot=slot, groups=sorted(self.divergent_groups),
                        new=new[:8], total_groups=len(deltas)))
            key = self.requests.get(row.request_id)
            if key is not None and key.uid != 'slot' + str(slot):
                raise ContractError('request moved without retirement')
            if key is None:
                generation = self.slot_generations.get(slot, 0) + 1
                self.slot_generations[slot] = generation
                key = RequestKey(self.pool.epoch, 'slot' + str(slot), generation)
                self.arena.register(key)
                self.requests[row.request_id] = key
            natural(row.computed, 'computed tokens')
            natural(row.scheduled, 'scheduled tokens', 1)
            end = row.computed + row.scheduled
            for layer, group in self.layer_groups.items():
                n = (end + self.pool.page_tokens - 1) // self.pool.page_tokens
                if len(row.tables[group]) < n or len(set(row.tables[group][:n])) != n:
                    raise ContractError('missing or aliased QSA logical pages')
                for physical in row.tables[group][:n]:
                    self._page(layer, physical)
            self.current[row.request_id] = row

        # Same filtering as the pinned runner, one physical copy per selected
        # pair across all local QSA views. Metadata for each view stays separate.
        selected = set(b for g in set(self.layer_groups.values())
                       for b in (frame.copy_destinations[g] if frame.copy_destinations else ()))
        copies = [pair for pair in frame.copies if pair[1] in selected]
        if len(set(copies)) != len(copies):
            raise ContractError('duplicate CoW pair')
        for src, dst in copies:
            if src == dst:
                raise ContractError('self-copy cannot establish private ownership')
            for layer in self.layer_groups:
                source = self._page(layer, src)
                dest = self._page(layer, dst)
                dest.valid = source.valid
                dest.version += 1
                dest.writer = (frame.sequence, layer)
        self.frame_copies = (len(copies), len(copies) * len(self.layer_groups))
        self.cow_pairs += self.frame_copies[0]
        self.cow_views += self.frame_copies[1]
        for row in frame.rows:
            start, end = row.computed, row.computed + row.scheduled
            for layer, group in self.layer_groups.items():
                for logical in range((end + self.pool.page_tokens - 1) // self.pool.page_tokens):
                    page = self._page(layer, row.tables[group][logical])
                    offset = logical * self.pool.page_tokens
                    prior = max(0, min(self.pool.page_tokens, start - offset))
                    if page.valid < prior:
                        raise ContractError('computed prefix has no observed producer or CoW')
                    if offset + self.pool.page_tokens > start:
                        page.valid = max(page.valid, min(self.pool.page_tokens, end - offset))
                        page.version += 1
                        page.writer = (frame.sequence, layer)
        self.sequence = frame.sequence
        self.active = True
        self.ready_layers.clear()

    def producer_ready(self, layer, sequence, wait):
        """wait() must observe a SAME-worker execution-stream event recorded
        after this layer's real KV store (and earlier CoW/graph writes).
        A boolean, scheduler ack or completion from another stream is not a fence.
        CUDA event construction and validation belong to the step-3 adapter.
        """
        if self.failed or not self.active or sequence != self.sequence or layer not in self.layer_groups:
            raise ContractError('foreign producer fence')
        if self.arena.state != 'idle':
            raise Pending('cannot publish while arena is consumed')
        try:
            wait()
        except BaseException:
            self.failed = True
            raise
        for (name, _), page in self.pool.pages.items():
            if name == layer and page.writer is not None:
                if page.writer[0] > sequence:
                    raise ContractError('fence does not cover pending writer')
                page.writer = None
        self.ready_layers.add(layer)

    def acquire(self, rid, layer):
        if self.failed or not self.active or layer not in self.ready_layers:
            raise Pending('real producer completion not established')
        row = self.current[rid]
        end = row.computed + row.scheduled
        n = (end + self.pool.page_tokens - 1) // self.pool.page_tokens
        keys = tuple(self._page(layer, b).key for b in row.tables[self.layer_groups[layer]][:n])
        self.arena.acquire(Owner(self.requests[rid], self.sequence, layer), keys, end)
        return self.arena

    def leave(self):
        self._quiescent()
        if not self.active:
            raise ContractError('no RPC to finish')
        self.active = False
        self.current.clear()
        self.ready_layers.clear()

    def shutdown(self, wait):
        self._quiescent()
        try:
            wait()  # real worker device/stream quiescence before cache destruction
        except BaseException:
            self.failed = True
            raise
        for rid in tuple(self.requests):
            self._retire(rid)
        for page in self.pool.pages.values():
            page.writer = None
        self.active = False
        self.current.clear()
        self.ready_layers.clear()
        return self.counters()

    def cow_line(self):
        return ('FLASH_E2_COW seq=%d pairs=%d views=%d cumulative_pairs=%d cumulative_views=%d'
                % (self.sequence, *self.frame_copies, self.cow_pairs, self.cow_views))

    def take_divergence_notes(self):
        """Once per newly divergent non-source group: bounded, no block IDs."""
        notes, self.divergence_notes = self.divergence_notes, []
        return notes

    def counters(self):
        return dict(self.arena.counters(), rpc_sequence=self.sequence, rpc_active=self.active,
                    pending_producer_pages=sum(p.writer is not None for p in self.pool.pages.values()),
                    metadata_pages=len(self.pool.pages), failed=self.failed,
                    cow_pairs=self.cow_pairs, cow_views=self.cow_views,
                    table_divergences=self.table_divergences,
                    divergent_groups=sorted(self.divergent_groups))
