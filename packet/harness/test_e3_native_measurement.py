"""CPU-only tests for bounded source-derived native long-window measurement.

Every accepted stream here is produced by the **actual current engine source**:
``NativeSegmentDriver._record`` while the real driver drives four real
``LaneDispatcher``/``NativeWorkerBridge`` workers through the reviewed
driver/bridge fixtures (``test_e3_native_storage_replay.make_native_stack`` for
the 4-layer shape and the same fixture parameterised to the declared 48-layer
shape for the long window).  Entries carry the driver's own stage-local segment
vocabulary, per-stage ``op_seq``s, typed cumulative reports/proofs, PP receipts,
commit/release rows, pre-commit ``SourceHold.describe`` and committed frontier.

Refusal tests inject their fault by removing or mutating those real entries
(one missing segment, one missing proof, a wrong route/epoch/identity, a
reordered ingress, a cancelled chunk); no test hand-writes a substitute
"source" stream.  No device, engine, GPU, network, CUDA or box access is used.
"""
import copy
import time
import sys
import types
import unittest
from collections import deque
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'patches/s5_engine'))
sys.path.insert(0, str(HERE))

import native_measurement as nm  # noqa: E402
import test_e3_r4_worker_bridge as bridge_suite  # noqa: E402
import test_e3_native_storage_replay as replay_suite  # noqa: E402
from test_e3_native_storage_replay import make_native_stack  # noqa: E402

UID = bridge_suite.UID


# --- real long-window shape -------------------------------------------------

class ChunkBlocks(bridge_suite.Manager):
    """Fixture allocator at the declared 1024-token chunk granularity.

    The reviewed admission fixture allocates one fake block per scheduled
    token; its lease table then grows quadratically in block count for a
    200k-token request (``release_after_workers`` walks the held lease once per
    block).  Block granularity is not part of the measurement source shape --
    segments, reports, proofs, commits and holds are unchanged -- so one block
    per declared chunk keeps the driver/bridge work identical while the
    long-window run stays bounded.
    """

    last_alloc = ()

    def allocate_slots(self, request, num_new, **kwargs):
        count = max(1, (num_new + nm.NATIVE_CHUNK_TOKENS - 1)
                    // nm.NATIVE_CHUNK_TOKENS)
        self.alloc_calls.append((request, num_new, kwargs))
        if self.fail:
            return None
        ids = self.block_pool.allocate(count)
        row = self.rows.get(request.request_id, ())
        row += ids
        self.next_block += count
        self.rows[request.request_id] = row
        self.last_alloc = ids
        return object()


def make_long_stack(*, tokens, layers=nm.MODEL_LAYER_COUNT, chunk=1024,
                    manager_cls=None):
    """The reviewed 4-layer Stack fixture, parameterised to the real shape."""
    half = layers // 2
    stack = types.SimpleNamespace()
    manager = (manager_cls or bridge_suite.Manager)()
    stack.scheduler = bridge_suite.Scheduler(
        manager, [bridge_suite.Request(uid=UID, tokens=tokens, generation=1)])
    if manager_cls is not None:
        # The reviewed fixture derives "unpublished pages" by slicing the last
        # (end - start) block-row entries, which is only correct at one block
        # per token.  At declared chunk granularity the same field is exactly
        # the ids this allocation call created.
        stack.scheduler._flash_s5_admission_metadata = (
            lambda request, allocated, start, end: dict(
                new_block_ids_to_zero=(), cow_pairs=(), ngram_context=(),
                unpublished_blocks=getattr(manager, 'last_alloc', ())))
    stack.controller = bridge_suite.make_controller(
        stack.scheduler, chunk=chunk, ranks=(0, 1, 2, 3))
    stack.record = stack.controller.admit(stack.scheduler)
    stack.sender = bridge_suite.RoutingSender()
    stack.transport = bridge_suite.DirectTransport(stack.sender)
    stack.executor = bridge_suite.FakeExecutor(stack.sender, stack.transport)
    stack.driver = bridge_suite.ed.NativeSegmentDriver(
        stack.controller, executor=stack.executor, scheduler=stack.scheduler,
        engine=types.SimpleNamespace(), layer_count=layers, layer_split=half,
        stage_of=((0, 0), (1, 0), (2, 1), (3, 1)),
        reply_timeout_s=5.0, stop_budget=64, stop_tick_budget=8).install()
    stack.workers = {}

    def add_worker(rank, stage):
        lane_layers = (list(range(half)) if stage == 0
                       else list(range(half, layers)))
        bus = bridge_suite.EventBus()
        torch = bridge_suite.bus_torch(bus)
        state = bridge_suite.WorkerState(rank, engine_epoch=0,
                                         engine_bound=False)
        state.retain_decode_storage(context_id=100 + rank)
        split, controller, router, route_backend = bridge_suite.route_objects(
            rank, state, bus, torch)
        sender = stage == 0
        if sender:
            receiver = stack.workers.get(2 if rank == 0 else 3)
            receiver_lane = None
            if receiver is not None:
                receiver_lane = getattr(
                    receiver.server.runtime.state.prefill, 'handle', None)
            channel = bridge_suite.FakePPChannel(
                bus, is_sender=True, receiver_lane=receiver_lane)
        else:
            channel = bridge_suite.FakePPChannel(bus, is_sender=False)
        lane = bridge_suite.FakeLane(layers=lane_layers)
        lane.is_sender = sender
        native = types.SimpleNamespace(
            split=split,
            comms=types.SimpleNamespace(pp=channel),
            copy_stream=bridge_suite.FakeStream(100 + rank),
            controller=controller,
            router=router,
            lane_forward=lambda inputs, forward_context=None,
            attention_driver=None: lane)
        server = bridge_suite.BridgeServer(state)
        factory = bridge_suite.FakeFactory(rank)
        dispatcher = bridge_suite.LaneDispatcher(
            'runner-%d' % rank, native, factory, server=server,
            driver_factory=lambda operation, module, context:
            bridge_suite.FakeDriver(),
            torch_module=torch, clock=lambda: 1000 + rank)
        bridge = bridge_suite.NativeWorkerBridge(
            dispatcher, server=server,
            source_binder=bridge_suite.fake_source_binder)
        worker = bridge_suite.FakeWorker(rank, stage, bridge, dispatcher)
        worker.bus = bus
        worker.torch = torch
        worker.state = state
        worker.split = split
        worker.controller = controller
        worker.router = router
        worker.route_backend = route_backend
        worker.lane = lane
        worker.channel = channel
        worker.server = server
        worker.factory = factory
        bridge.attach(worker)
        stack.workers[rank] = worker

    for rank, stage in ((0, 0), (1, 0), (2, 1), (3, 1)):
        add_worker(rank, stage)
    stack.transport.network = types.SimpleNamespace(workers=stack.workers)
    stack.worker = stack.workers.__getitem__
    for rank in (2, 3):
        stack.worker(rank).channel.ready = True

    def drive(until, limit=4000):
        for _ in range(limit):
            if until():
                return
            stack.driver.advance()
        raise AssertionError(
            'driver did not reach the expected state; phase=%s index=%s'
            % (stack.driver.phase, stack.driver.index))

    def drive_to_finalize():
        drive(lambda: stack.driver.phase == 'finalize')

    def complete_everything():
        for worker in stack.workers.values():
            worker.bus.complete_all(True)

    stack.drive = drive
    stack.drive_to_finalize = drive_to_finalize
    stack.complete_everything = complete_everything
    stack.finish_chunk = lambda: (
        drive_to_finalize(), complete_everything(),
        drive(lambda: stack.driver.tx is None))
    return stack


# --- capture and drive the real source --------------------------------------

def capture_record(stack):
    """Collect the exact entries ``NativeSegmentDriver._record`` writes."""
    entries = []
    original = stack.driver._record

    def recording(entry_kind, **fields):
        entry = original(entry_kind, **fields)
        entries.append(entry)
        return entry

    stack.driver._record = recording
    return entries


def run_journal(stack, *, extra_requests=(), pauses=()):
    """Drive complete chunks and return the real journal entries, in order.

    The finished request row is dropped from the fixture's waiting queue after
    its terminal chunk: the stock scheduler owns that one-token handback row
    and the admission hook correctly refuses it, so a queued *next* request
    can only be reached once the fixture removes it.  This changes no engine
    event and no measurement fact.
    """
    entries = capture_record(stack)
    stack.drive(lambda: stack.driver._split_ready)
    # The fixture admits its first request at construction; only extra
    # requests wait here.
    stack.scheduler.waiting.items = []
    for request in extra_requests:
        stack.scheduler.waiting.items.append(request)
    for rank in pauses:
        stack.worker(rank).lane.pause_counts[0] = 2
    chunks = 0
    for _ in range(400):
        if stack.driver.tx is None and not stack.controller.active:
            stack.scheduler.waiting.items = [
                item for item in stack.scheduler.waiting.items
                if item.num_computed_tokens < item.num_tokens - 1]
            if not stack.scheduler.waiting.items:
                break
            if stack.controller.admit(stack.scheduler) is None:
                break
        stack.drive_to_finalize()
        stack.complete_everything()
        stack.drive(lambda: stack.driver.tx is None)
        if stack.driver.error is not None:
            raise AssertionError(stack.driver.error)
        chunks += 1
    else:
        raise AssertionError('request did not finish within the bound')
    return entries


def census_for(stack, **overrides):
    kwargs = dict(
        engine_epoch=stack.driver.engine_epoch,
        ranks=tuple(stack.driver.ranks),
        stage_ranks={stage: tuple(members)
                     for stage, members in stack.driver.stage_ranks.items()},
        pp_senders=tuple(stack.driver.stage_ranks[0]),
        expected_segments_per_chunk=nm.SEGMENTS_PER_CHUNK)
    kwargs.update(overrides)
    return nm.NativeMeasurementCensus(**kwargs)


def window_of(entries):
    stamps = [entry['observed_ns'] for entry in entries]
    return min(stamps) - 1, max(stamps) + 1


def classify(stack, entries, **overrides):
    census = census_for(stack, **overrides)
    for entry in entries:
        census.ingest(entry)
    start_ns, end_ns = window_of(entries)
    return census, census.evidence(start_ns, end_ns, now_ns=end_ns + 1)


def reasons(evidence):
    return set(evidence['reasons'])


def transaction_entries(entries, txn):
    return [entry for entry in entries if entry.get('transaction') == txn]


def kind_entries(entries, kind, *, txn=None, rank=None):
    return [entry for entry in entries
            if entry.get('kind') == kind
            and (txn is None or entry.get('transaction') == txn)
            and (rank is None or entry.get('worker_rank') == rank)]


def spans(prompt_end, start=0, size=1024):
    """Admitted chunk spans: the final prompt token is left for decode."""
    rows = []
    cursor = start
    last = prompt_end - 1
    while cursor < last:
        end = min(cursor + size, last)
        rows.append((cursor, end))
        cursor = end
    return rows


def repeat_reports(entries, txn, repeats):
    """Copy real typed completion reports as later cumulative polls.

    The driver records one proof-carrying completion report per rank and chunk;
    the census must retain that first queried observation even when the same
    report is polled again.  Copies keep the real report shape unchanged and
    are observed before the chunk commit.
    """
    events = []
    for entry in entries:
        events.append(entry)
        if entry.get('kind') == 'report' and entry.get('transaction') == txn \
                and entry['proofs']:
            for index in range(1, repeats + 1):
                duplicate = copy.deepcopy(entry)
                duplicate['observed_ns'] = entry['observed_ns'] + index
                events.append(duplicate)
    return events


def first_open(entries, txn=1):
    return next(entry for entry in entries
                if entry.get('kind') == 'chunk_open'
                and entry.get('transaction') == txn)


_LONG_RUN = {}


def long_run():
    """The declared 200k fresh + 200k-cached 4096 append, once per suite."""
    if not _LONG_RUN:
        # Same native route objects the reviewed Stack fixture uses, applied
        # to the 48-layer parameterisation of that fixture.
        with mock.patch.object(bridge_suite, 'route_objects',
                               replay_suite.native_route_objects):
            stack = make_long_stack(tokens=200000, manager_cls=ChunkBlocks)
        append = bridge_suite.Request(uid='uid-append', tokens=204096,
                                      computed=200000, generation=1)
        entries = run_journal(stack, extra_requests=(append,))
        _LONG_RUN['run'] = (stack, entries)
    return _LONG_RUN['run']


class NativeMeasurementSourceShapeTests(unittest.TestCase):

    # -- helpers -----------------------------------------------------------
    def standard(self, **kwargs):
        stack = make_native_stack()
        return stack, run_journal(stack, **kwargs)

    def assert_refused(self, evidence, *expected):
        self.assertEqual(evidence['status'], 'INCONCLUSIVE')
        self.assertFalse(evidence['construction_valid'])
        missing = set(expected) - reasons(evidence)
        self.assertFalse(missing, 'missing %r in %r'
                         % (sorted(missing), sorted(reasons(evidence))))

    def assert_complete(self, evidence):
        self.assertEqual(evidence['status'], 'COMPLETE',
                         evidence['reasons'])
        self.assertTrue(evidence['construction_valid'])
        self.assertEqual(evidence['reasons'], [])

    # -- long real shape ---------------------------------------------------
    def test_full_declared_fresh_prompt_and_suffix_append(self):
        """200k fresh (196 chunks) + 200k-cached 4096 append from the driver."""
        stack, entries = long_run()
        census, evidence = classify(stack, entries)
        self.assert_complete(evidence)
        fresh = [row for row in evidence['segments']
                 if row['request'] == UID]
        append = [row for row in evidence['segments']
                  if row['request'] == 'uid-append']
        self.assertEqual(len(fresh), len(spans(200000)))
        self.assertEqual(len(append), len(spans(204096, start=200000)))
        self.assertEqual({row['prefill'] for row in fresh}, {'fresh'})
        self.assertEqual({row['prefill'] for row in append}, {'append'})
        self.assertTrue(all(row['accepted'] for row in evidence['segments']))
        self.assertGreater(len(entries), 1024)
        groups = {row['request']: row for row in evidence['groups']}
        fresh_group = groups[UID]
        append_group = groups['uid-append']
        self.assertTrue(fresh_group['complete'])
        self.assertTrue(append_group['complete'])
        self.assertEqual(fresh_group['family'], 'fresh')
        self.assertEqual(append_group['family'], 'append')
        self.assertEqual(fresh_group['initial_frontier'], 0)
        self.assertEqual(append_group['initial_frontier'], 200000)
        self.assertEqual(fresh_group['chunks'], len(spans(200000)))
        self.assertEqual(append_group['chunks'],
                         len(spans(204096, start=200000)))
        self.assertEqual(fresh_group['prompt_tokens'], 200000)
        self.assertEqual(append_group['prompt_tokens'], 204096)
        self.assertEqual(fresh_group['useful_tokens'], 199999)
        self.assertEqual(append_group['useful_tokens'], 4095)
        self.assertEqual(fresh_group['chunk_start'], 0)
        self.assertEqual(fresh_group['chunk_end'], 199999)
        self.assertEqual(append_group['chunk_start'], 200000)
        self.assertEqual(append_group['chunk_end'], 204095)
        self.assertEqual(fresh_group['start_ns'],
                         min(row['start_monotonic_ns'] for row in fresh))
        self.assertEqual(fresh_group['end_ns'],
                         max(row['final_queried_state_write_ns']
                             for row in fresh))
        self.assertEqual(len(evidence['completions']),
                         (len(fresh) + len(append)) * len(stack.driver.ranks))
        self.assertEqual(len(evidence['pp_consumer_receipts']),
                         (len(fresh) + len(append))
                         * len(stack.driver.stage_ranks[0]))
        # Every accepted segment satisfies the downstream metrics contract and
        # carries the real descriptor shape of the 48-layer plan: 29 stage-local
        # operations per stage.  That is the observed vocabulary of this run,
        # never a completeness count (the paused test keeps more).
        stage_count = len(stack.driver.stage_ranks)
        for row in evidence['segments']:
            self.assertIs(row['accepted'], True)
            self.assertEqual(row['transport'], 'native-split')
            self.assertIs(row['state_write'], True)
            self.assertIs(type(row['start_monotonic_ns']), int)
            self.assertIs(type(row['end_monotonic_ns']), int)
            self.assertGreater(row['end_monotonic_ns'],
                               row['start_monotonic_ns'])
            self.assertIs(type(row['chunk_tokens']), int)
            self.assertEqual(row['segment_identity'], '(stage, op_seq)')
            self.assertEqual(len(row['stage_segment_counts']), stage_count)
            self.assertEqual(
                set(row['stage_segment_counts']),
                {str(stage) for stage in stack.driver.stage_ranks})
            self.assertEqual(
                sum(row['stage_segment_counts'].values()),
                (10 + nm.MODEL_LAYER_COUNT))
            self.assertEqual(row['plan_segments_per_chunk'],
                             nm.SEGMENTS_PER_CHUNK)
        for row in evidence['completions']:
            self.assertIs(row['queried'], True)
            writes = [proof['completed_ns'] for proof in row['proofs']
                      if proof['kind'] == 'state_write']
            self.assertEqual(len(writes), 1)
            self.assertEqual(row['state_write_ns'], max(writes))
        receipt_bytes = evidence['completeness']['serialized_receipt_bytes']
        retained_bytes = census.stats()['host_bytes']['bytes']
        self.assertIs(type(receipt_bytes), int)
        self.assertGreater(receipt_bytes, 0)
        self.assertIs(type(retained_bytes), int)
        self.assertLess(retained_bytes, 64 * 1024 * 1024,
                        'retained=%d receipt=%d' % (retained_bytes,
                                                    receipt_bytes))
        self.assertLess(receipt_bytes, 64 * 1024 * 1024)
        stats = census.stats()
        capacity = stats['capacity']
        self.assertEqual(capacity['max_chunks_per_request'],
                         nm.MAX_CHUNKS_PER_REQUEST)
        self.assertEqual(capacity['max_chunks_total'], nm.MAX_CHUNKS_TOTAL)
        self.assertEqual(capacity['segment_identity'], '(stage, op_seq)')
        self.assertEqual(capacity['plan_segments_per_chunk'],
                         nm.SEGMENTS_PER_CHUNK)
        self.assertIs(type(stats['projected_host_bytes_at_capacity']['bytes']),
                      int)
        self.assertEqual(stats['overflow']['count'], 0)
        self.assertEqual(stats['faults']['count'], 0)
        print('scale: chunks=%d segments=%d completions=%d retained=%dB '
              'receipt=%dB projected_at_capacity=%sB raw_journal_dropped=%d '
              'capacity=%s/%s'
              % (len(evidence['segments']),
                 sum(row['segment_count'] for row in evidence['segments']),
                 len(evidence['completions']), retained_bytes, receipt_bytes,
                 stats['projected_host_bytes_at_capacity']['bytes'],
                 stack.driver._journal_dropped,
                 capacity['max_requests'], capacity['max_chunks_total']))

    # -- cumulative reports ------------------------------------------------
    def test_repeated_cumulative_reports_keep_first_completion_observation(self):
        stack, entries = self.standard()
        events = repeat_reports(entries, 1, 40)
        census, evidence = classify(stack, events)
        self.assert_complete(evidence)
        self.assertEqual(len(evidence['completions']),
                         len(evidence['segments']) * len(stack.driver.ranks))
        repeats = 0
        for row in evidence['completions']:
            rank = row['worker_rank']
            txn = row['transaction']
            proofs = [entry for entry in kind_entries(
                events, 'report', txn=txn, rank=rank) if entry['proofs']]
            self.assertTrue(proofs)
            first = min(entry['observed_ns'] for entry in proofs)
            self.assertEqual(row['observed_ns'], first,
                             'first queried proof observation, not a later '
                             'cumulative duplicate')
            for proof in row['proofs']:
                same = [entry['observed_ns'] for entry in proofs
                        if any(item['kind'] == proof['kind']
                               and item['op_seq'] == proof['op_seq']
                               and item['completion_seq']
                               == proof['completion_seq']
                               for item in entry['proofs'])]
                self.assertGreaterEqual(len(same), 1)
                self.assertEqual(proof['completed_ns'], min(same))
            if txn == 1:
                self.assertEqual(row['proof_repeats'],
                                 40 * len(row['proofs']))
                repeats += row['proof_repeats']
            else:
                self.assertEqual(row['proof_repeats'], 0)
        self.assertGreater(repeats, 0)
        self.assertGreaterEqual(
            census.stats()['ingested']['duplicate_observations'], repeats)

    def test_paused_pending_samples_are_not_completions(self):
        stack, entries = self.standard(pauses=(0, 1))
        census, evidence = classify(stack, entries)
        self.assert_complete(evidence)
        paused_rows = [row for row in evidence['segments']
                       if row['transaction'] == 1]
        self.assertEqual(len(paused_rows), 1)
        row = paused_rows[0]
        # The paused QSA continuation is re-issued under new stage-local ids:
        # stage 0 keeps its normal seven operations and adds two more layer
        # re-issues; the plan length is not a count.
        self.assertEqual(row['stage_segment_counts'],
                         {'0': 9, '1': 7})
        self.assertEqual(row['segment_count'], 16)
        completions = [item for item in evidence['completions']
                       if item['transaction'] == 1]
        self.assertEqual(len(completions), len(stack.driver.ranks))
        for item in completions:
            pending = [entry for entry in kind_entries(
                entries, 'report', txn=1, rank=item['worker_rank'])
                if entry['pending_op_seqs']]
            self.assertEqual(item['pending_samples'], len(pending))
            self.assertGreater(item['pending_samples'], 0)
            for entry in pending:
                self.assertFalse(entry['proofs'])
                self.assertFalse(entry['state_written'])
            self.assertFalse(item['pending_after_completion'])
            writes = [proof['completed_ns'] for proof in item['proofs']
                      if proof['kind'] == 'state_write']
            self.assertEqual(len(writes), 1)
            self.assertEqual(item['state_write_ns'], max(writes))
            if item['stage'] == 0:
                self.assertEqual(item['completed_ops'],
                                 list(range(9)),
                                 'paused layer re-issues are real accepted ops')
            else:
                self.assertEqual(item['completed_ops'], list(range(7)))
        self.assertEqual(evidence['prefill']['completed_prefills'], 3)
        self.assertEqual(evidence['prefill']['useful_completed_tokens'], 11)
        # Pending samples create no proof of their own.
        self.assertGreater(census.stats()['ingested']['first_seen_proofs'], 0)

    # -- PP stages ---------------------------------------------------------
    def test_two_pp_stages_each_logical_proof_retained_once(self):
        stack, entries = self.standard()
        events = repeat_reports(entries, 1, 5)
        _census, evidence = classify(stack, events)
        self.assert_complete(evidence)
        pairs = [(entry['stage'], entry['op_seq'])
                 for entry in entries if entry['kind'] == 'segment']
        self.assertLess(len(set(op for _stage, op in pairs)), len(pairs),
                        'the two stages legitimately share op_seq values')
        self.assertEqual({stage for stage, _op in pairs},
                         set(stack.driver.stage_ranks))
        for txn in (1, 2, 3):
            stage_ops = {}
            for entry in kind_entries(entries, 'segment', txn=txn):
                stage_ops.setdefault(entry['stage'], set()).add(
                    entry['op_seq'])
            self.assertEqual(set(stage_ops), set(stack.driver.stage_ranks))
            # Same op id on both stages names two different descriptors.
            self.assertEqual(stage_ops[0], stage_ops[1])
        self.assertEqual(len(evidence['pp_consumer_receipts']),
                         len(evidence['segments']) * 2)
        self.assertEqual({row['sender_rank']
                          for row in evidence['pp_consumer_receipts']},
                         set(stack.driver.stage_ranks[0]))
        stage_completions = {}
        for row in evidence['completions']:
            stage_completions.setdefault(row['transaction'], set()).add(
                row['stage'])
            writes = [proof for proof in row['proofs']
                      if proof['kind'] == 'state_write']
            consumers = [proof for proof in row['proofs']
                         if proof['kind'] == 'pp_consumer']
            self.assertEqual(len(writes), 1)
            if row['stage'] == 0:
                self.assertEqual(consumers, [])
            else:
                self.assertEqual(len(consumers), 1)
            if row['transaction'] == 1:
                self.assertEqual(row['proof_repeats'],
                                 5 * len(row['proofs']))
            else:
                self.assertEqual(row['proof_repeats'], 0)
        for stages in stage_completions.values():
            self.assertEqual(stages, set(stack.driver.stage_ranks))
        # TP peers in a stage must agree on the actual completed op set.
        for txn in (1, 2, 3):
            by_stage = {}
            for row in evidence['completions']:
                if row['transaction'] != txn:
                    continue
                by_stage.setdefault(row['stage'], set()).add(
                    tuple(row['completed_ops']))
            for stage, sets in by_stage.items():
                self.assertEqual(len(sets), 1,
                                 'stage %d peers disagree: %r' % (stage, sets))

    def test_missing_pp_proof_refuses(self):
        stack, entries = self.standard()
        dropped = [entry for entry in entries
                   if not (entry['kind'] == 'pp_receipt'
                           and entry['transaction'] == 1
                           and entry['sender_rank'] == 0)]
        self.assertEqual(len(dropped), len(entries) - 1)
        _census, evidence = classify(stack, dropped)
        self.assert_refused(evidence, 'chunk-1:missing-pp-consumer-proof:0')

    # -- request family ----------------------------------------------------
    def test_fresh_chunk_two_plus_stays_fresh_and_distinct_append_is_append(self):
        stack = make_native_stack()
        append = bridge_suite.Request(uid='uid-append', tokens=19,
                                      computed=12, generation=1)
        entries = run_journal(stack, extra_requests=(append,))
        _census, evidence = classify(stack, entries)
        self.assert_complete(evidence)
        fresh = [row for row in evidence['segments'] if row['request'] == UID]
        added = [row for row in evidence['segments']
                 if row['request'] == 'uid-append']
        self.assertEqual(len(fresh), 3)
        self.assertEqual(len(added), 2)
        self.assertEqual([row['prefill'] for row in fresh], ['fresh'] * 3)
        self.assertEqual([row['prefill'] for row in added], ['append'] * 2)
        # Chunk two onward of the SAME fresh request has committed_end > 0.
        self.assertGreater(fresh[1]['committed_end'], 0)
        self.assertGreater(fresh[2]['committed_end'], 0)
        self.assertEqual([row['initial_frontier'] for row in fresh], [0, 0, 0])
        self.assertEqual([row['initial_frontier'] for row in added], [12, 12])
        groups = {row['request']: row for row in evidence['groups']}
        self.assertEqual(groups[UID]['family'], 'fresh')
        self.assertEqual(groups['uid-append']['family'], 'append')
        self.assertEqual(groups['uid-append']['initial_frontier'], 12)

    def test_uid_reuse_with_other_generation_is_a_distinct_request(self):
        stack = make_native_stack()
        reuse = bridge_suite.Request(uid=UID, tokens=6, computed=0,
                                     generation=2)
        entries = run_journal(stack, extra_requests=(reuse,))
        _census, evidence = classify(stack, entries)
        self.assert_complete(evidence)
        groups = evidence['groups']
        self.assertEqual(len(groups), 2)
        self.assertEqual({row['request'] for row in groups}, {UID})
        self.assertEqual(len({row['request_generation'] for row in groups}), 2)
        self.assertEqual(len({row['request_generation']
                              for row in evidence['completions']}), 2)
        self.assertEqual(len({row['request_generation']
                              for row in evidence['segments']}), 2)

    # -- ordering, gaps, duplicate conflicts -------------------------------
    def test_reordered_chunks_do_not_pass(self):
        stack, entries = self.standard()
        events = (transaction_entries(entries, 1)
                  + transaction_entries(entries, 3)
                  + transaction_entries(entries, 2))
        _census, evidence = classify(stack, events)
        self.assert_refused(evidence,
                            'request-%s#1:chunk-reordered' % UID)
        self.assertEqual(len(evidence['segments']), 3)

    def test_chunk_frontier_gap_does_not_pass(self):
        stack, entries = self.standard()
        dropped = (transaction_entries(entries, 1)
                   + transaction_entries(entries, 3))
        _census, evidence = classify(stack, dropped)
        self.assert_refused(evidence,
                            'request-%s#1:chunk-frontier-gap' % UID)

    def test_duplicate_open_same_fields_is_a_repeated_observation(self):
        stack, entries = self.standard()
        events = copy.deepcopy(entries)
        source = first_open(events)
        duplicate = copy.deepcopy(source)
        duplicate['observed_ns'] = max(
            entry['observed_ns'] for entry in events) + 2
        events.insert(events.index(source) + 1, duplicate)
        census, evidence = classify(stack, events)
        self.assert_complete(evidence)
        self.assertGreaterEqual(
            census.stats()['ingested']['duplicate_observations'], 1)

    def test_conflicting_duplicate_open_refuses(self):
        stack, entries = self.standard()
        events = copy.deepcopy(entries)
        source = first_open(events)
        duplicate = copy.deepcopy(source)
        duplicate['observed_ns'] = max(
            entry['observed_ns'] for entry in events) + 2
        duplicate['state_epoch'] = duplicate['state_epoch'] + 1
        events.insert(events.index(source) + 1, duplicate)
        _census, evidence = classify(stack, events)
        self.assert_refused(evidence, 'chunk-1:duplicate-chunk-open-conflict')

    def test_duplicate_open_generation_conflict_refuses(self):
        stack, entries = self.standard()
        events = copy.deepcopy(entries)
        source = first_open(events)
        duplicate = copy.deepcopy(source)
        duplicate['observed_ns'] = max(
            entry['observed_ns'] for entry in events) + 2
        duplicate['request_generation'] = duplicate['request_generation'] + 1
        events.insert(events.index(source) + 1, duplicate)
        _census, evidence = classify(stack, events)
        self.assert_refused(evidence, 'chunk-1:duplicate-chunk-open-conflict')

    def test_terminal_coverage_required(self):
        stack, entries = self.standard()
        truncated = (transaction_entries(entries, 1)
                     + transaction_entries(entries, 2))
        _census, evidence = classify(stack, truncated)
        self.assert_refused(
            evidence, 'request-%s#1:request-terminal-not-covered' % UID)

    # -- identity / route / epoch / proof refusals -------------------------
    def test_missing_rank_state_write_refuses(self):
        stack, entries = self.standard()
        events = copy.deepcopy(entries)
        for entry in events:
            if entry['kind'] != 'report' or entry['worker_rank'] != 3:
                continue
            entry['proofs'] = [proof for proof in entry['proofs']
                               if proof['kind'] != 'state_write']
            entry['state_written'] = False
        _census, evidence = classify(stack, events)
        self.assert_refused(evidence, 'chunk-1:missing-state-write:3')

    def test_missing_rank_report_is_a_partial_reply_refusal(self):
        stack, entries = self.standard()
        events = [entry for entry in entries
                  if not (entry['kind'] == 'report'
                          and entry.get('transaction') == 1
                          and entry.get('worker_rank') == 2)]
        _census, evidence = classify(stack, events)
        self.assert_refused(evidence, 'chunk-1:missing-rank-report:2')

    def test_wrong_route_refuses(self):
        stack, entries = self.standard()
        events = copy.deepcopy(entries)
        for entry in events:
            if entry.get('transaction') == 1:
                entry['route'] = 'full'
        _census, evidence = classify(stack, events)
        self.assert_refused(evidence, 'chunk-1:chunk-route-not-split')

    def test_wrong_stage_and_unknown_kind_refuse(self):
        stack, entries = self.standard()
        events = copy.deepcopy(entries)
        stage0 = next(entry for entry in events
                      if entry['kind'] == 'segment'
                      and entry['transaction'] == 1
                      and entry['stage'] == 0)
        # A real stage-0 segment relabelled stage 1 keeps its stage-0 target
        # ranks: the declared stage and its rank set no longer agree.
        wrong_stage = copy.deepcopy(stage0)
        wrong_stage['observed_ns'] = stage0['observed_ns'] + 1
        wrong_stage['stage'] = 1
        events.insert(events.index(stage0) + 1, wrong_stage)
        unknown_kind = copy.deepcopy(stage0)
        unknown_kind['observed_ns'] = stage0['observed_ns'] + 2
        unknown_kind['segment_kind'] = 'kernel-count-magic'
        events.insert(events.index(stage0) + 2, unknown_kind)
        _census, evidence = classify(stack, events)
        self.assert_refused(
            evidence,
            'chunk-1:segment-targets-differ-from-stage-ranks',
            'chunk-1:segment-kind-unproved')

    def test_wrong_engine_epoch_is_global_refusal(self):
        stack, entries = self.standard()
        events = copy.deepcopy(entries)
        first_report = next(entry for entry in events
                            if entry['kind'] == 'report'
                            and entry['transaction'] == 1)
        first_report['engine_epoch'] = stack.driver.engine_epoch + 1
        _census, evidence = classify(stack, events)
        self.assert_refused(evidence, 'engine-epoch-differs')

    def test_commit_identity_conflict_refuses(self):
        stack, entries = self.standard()
        events = copy.deepcopy(entries)
        for entry in events:
            if entry['kind'] == 'chunk_commit' and entry['transaction'] == 1:
                entry['request'] = 'uid-other'
        _census, evidence = classify(stack, events)
        self.assert_refused(evidence, 'chunk-1:commit-request-conflict')

    def test_premature_source_release_refuses(self):
        stack, entries = self.standard()
        events = copy.deepcopy(entries)
        # The driver records hold.describe() *before* commit: released is
        # false in the real entry and the census must accept it as observed.
        hold = next(entry for entry in events
                    if entry['kind'] == 'source_hold'
                    and entry['transaction'] == 1)
        self.assertIs(hold['description']['released'], False)
        commit_ns = [entry['observed_ns'] for entry in events
                     if entry['kind'] == 'chunk_commit'
                     and entry['transaction'] == 1][0]
        for entry in events:
            if entry['kind'] == 'source_hold' and entry['transaction'] == 1:
                entry['observed_ns'] = commit_ns - 1
        _census, evidence = classify(stack, events)
        self.assert_refused(evidence, 'chunk-1:source-released-before-commit')

    def test_wrong_source_identity_refuses(self):
        stack, entries = self.standard()
        events = copy.deepcopy(entries)
        hold = next(entry for entry in events
                    if entry['kind'] == 'source_hold'
                    and entry['transaction'] == 1)
        hold['description']['transaction_seq'] = hold['transaction'] + 1
        hold['description']['request_uid'] = 'uid-other'
        hold['description']['bound'] = False
        hold['description']['ranks'] = len(stack.driver.ranks) - 1
        _census, evidence = classify(stack, events)
        self.assert_refused(
            evidence,
            'chunk-1:source-hold-transaction-conflict',
            'chunk-1:source-hold-request-conflict',
            'chunk-1:source-hold-not-bound',
            'chunk-1:source-hold-rank-count-differs')

    def test_frontier_pp_seq_conflict_refuses(self):
        stack, entries = self.standard()
        events = copy.deepcopy(entries)
        for entry in events:
            if entry['kind'] == 'committed_frontier' \
                    and entry['transaction'] == 1:
                entry['frontier']['pp_seqs'] = [7]
        _census, evidence = classify(stack, events)
        self.assert_refused(evidence, 'chunk-1:frontier-pp-seq-differs:0',
                            'chunk-1:frontier-pp-seq-differs:1')

    def test_conflicting_repeated_proof_refuses(self):
        stack, entries = self.standard()
        events = copy.deepcopy(entries)
        final = [entry for entry in events
                 if entry['kind'] == 'report'
                 and entry['transaction'] == 1
                 and entry['worker_rank'] == 0
                 and entry['state_written']][-1]
        conflicting = copy.deepcopy(final)
        conflicting['observed_ns'] = max(
            entry['observed_ns'] for entry in events) + 2
        for proof in conflicting['proofs']:
            if proof['kind'] == 'state_write':
                proof['completion_seq'] = proof['completion_seq'] + 100
        events.append(conflicting)
        _census, evidence = classify(stack, events)
        self.assert_refused(evidence, 'chunk-1:report-proof-conflict')

    # -- cancellation, capacity, incompleteness ----------------------------
    def test_cancellation_is_visible_and_never_completed(self):
        stack = make_native_stack()
        entries = capture_record(stack)
        stack.drive(lambda: stack.driver._split_ready)
        stack.drive_to_finalize()
        stack.complete_everything()
        stack.advance()
        self.assertEqual(stack.executor.last_method(), 'flash_s5_poll')
        stack.driver.stop('cancel')
        cancels = kind_entries(entries, 'cancel')
        self.assertEqual(len(cancels), 1)
        self.assertEqual(cancels[0]['closed_ranks'], [0, 1, 2, 3])
        self.assertFalse(cancels[0]['never_submitted'])
        _census, evidence = classify(stack, entries)
        self.assert_refused(evidence, 'chunk-1:chunk-cancelled',
                            'chunk-1:missing-chunk-commit')
        self.assertEqual(len(evidence['cancellations']), 1)
        self.assertEqual(evidence['cancellations'][0]['closed_ranks'],
                         [0, 1, 2, 3])
        self.assertEqual(evidence['prefill']['completed_prefills'], 0)
        self.assertEqual(evidence['prefill']['useful_completed_tokens'], 0)

    def test_capacity_overflow_is_explicit_inconclusive(self):
        stack, entries = self.standard()
        census, evidence = classify(stack, entries, max_chunks_total=1)
        self.assert_refused(evidence, 'measurement-capacity-overflow')
        stats = census.stats()
        self.assertGreaterEqual(stats['overflow']['count'], 1)
        self.assertIn('chunk-capacity-total', stats['overflow']['reasons'])

    def test_one_missing_ingress_event_prevents_completeness(self):
        stack, entries = self.standard(pauses=(0, 1))
        # One expanded stage-local segment (a paused layer re-issue) removed:
        # the rank's own cumulative completed set still names it.
        expanded = [entry for entry in entries
                    if entry['kind'] == 'segment'
                    and entry['transaction'] == 1
                    and entry['stage'] == 0
                    and entry['op_seq'] == 3]
        self.assertEqual(len(expanded), 1)
        dropped = [entry for entry in entries if entry is not expanded[0]]
        _census, evidence = classify(stack, dropped)
        self.assert_refused(evidence,
                            'chunk-1:segment-set-differs-from-report:0')
        # One rank proof removed from every cumulative report of that rank:
        # the GPU_EVENT join no longer covers the stage's accepted ids.
        events = copy.deepcopy(entries)
        victim = None
        for entry in events:
            if entry['kind'] == 'report' and entry['transaction'] == 1 \
                    and entry['worker_rank'] == 0:
                for proof in entry['proofs']:
                    if proof['kind'] == 'gpu_event' and victim is None:
                        victim = proof['op_seq']
        self.assertIsNotNone(victim)
        for entry in events:
            if entry['kind'] == 'report' and entry['transaction'] == 1 \
                    and entry['worker_rank'] == 0:
                entry['proofs'] = [
                    proof for proof in entry['proofs']
                    if not (proof['kind'] == 'gpu_event'
                            and proof['op_seq'] == victim)]
        _census, evidence = classify(stack, events)
        self.assert_refused(evidence,
                            'chunk-1:gpu-event-set-differs-from-report:0')

    def test_window_excluding_a_chunk_does_not_join_the_gap(self):
        stack, entries = self.standard()
        first = transaction_entries(entries, 1)
        rest = (transaction_entries(entries, 2)
                + transaction_entries(entries, 3))
        census = census_for(stack)
        for entry in entries:
            census.ingest(entry)
        start_ns = max(entry['observed_ns'] for entry in first) + 1
        end_ns = max(entry['observed_ns'] for entry in rest) + 1
        evidence = census.evidence(start_ns, end_ns, now_ns=end_ns + 1)
        self.assert_refused(
            evidence, 'request-%s#1:request-extends-outside-window' % UID)
        self.assertEqual(len(evidence['segments']), 2)

    def test_commit_outside_window_refuses(self):
        stack, entries = self.standard()
        census = census_for(stack)
        for entry in entries:
            census.ingest(entry)
        start_ns = window_of(entries)[0]
        commit_ns = next(entry['observed_ns'] for entry in entries
                         if entry['kind'] == 'chunk_commit'
                         and entry['transaction'] == 1)
        evidence = census.evidence(start_ns, commit_ns - 1,
                                   now_ns=commit_ns)
        self.assert_refused(evidence, 'chunk-1:chunk-commit-after-window')

    # -- the raw ring stays the raw ring ----------------------------------
    def test_raw_ring_loss_unchanged_while_census_is_complete(self):
        stack, entries = long_run()
        ring = deque(maxlen=1024)
        dropped = 0
        for entry in entries:
            if len(ring) == ring.maxlen:
                dropped += 1
            ring.append(entry)
        self.assertGreater(dropped, 0)
        self.assertEqual(ring.maxlen, 1024)
        self.assertEqual(len(ring), 1024)
        _census, evidence = classify(stack, entries)
        self.assert_complete(evidence)
        completeness = evidence['completeness']
        self.assertTrue(completeness['independent_of_raw_journal'])
        self.assertFalse(completeness['raw_journal_consulted'])
        self.assertFalse(completeness['raw_journal_flag_mutated'])
        self.assertTrue(completeness['complete'])
        start_ns, _end_ns = window_of(entries)
        # The raw ring's retained first entry no longer covers the window
        # start, which is exactly what the unchanged journal loss guard says.
        self.assertGreater(ring[0]['observed_ns'], start_ns)
        # The engine's own raw journal is untouched and still bounded; the
        # census never reads, repairs or renames it.
        self.assertEqual(stack.driver._journal.maxlen, 1024)
        self.assertEqual(len(stack.driver._journal), 1024)
        self.assertGreater(stack.driver._journal_dropped, 0)
        # The unchanged engine counter and this independent replay of the raw
        # ring agree on the loss; the census neither reads nor repairs either.
        self.assertEqual(dropped, stack.driver._journal_dropped)
        self.assertEqual(dropped, len(entries) - 1024)

    # -- robustness / configuration ---------------------------------------
    def test_ingest_and_evidence_never_raise(self):
        stack, entries = self.standard()
        census = census_for(stack)
        for entry in (None, 'nonsense', 7, [], {'kind': 'report'},
                      {'kind': 'report', 'observed_ns': -1},
                      {'kind': 'chunk_open', 'observed_ns': 5},
                      {'kind': 'segment', 'observed_ns': 5,
                       'transaction': 'one'},
                      {'kind': 'chunk_open', 'observed_ns': 5,
                       'transaction': 1, 'engine_epoch': 'x'}):
            result = census.ingest(entry)
            self.assertIs(type(result), dict)
            self.assertIs(result.get('accepted'), False)
        for entry in entries:
            self.assertIs(type(census.ingest(entry)), dict)
        for start, end in ((None, None), ('a', 'b'), (0, 0), (10, 5),
                           (-1, 10), (1, 2)):
            evidence = census.evidence(start, end)
            self.assertIs(type(evidence), dict)
            self.assertIn(evidence['status'], ('COMPLETE', 'INCONCLUSIVE'))
            self.assertFalse(evidence['construction_valid'])
        self.assertGreater(census.stats()['faults']['count'], 0)

    def test_invalid_configuration_is_inconclusive_not_raising(self):
        census = nm.NativeMeasurementCensus(engine_epoch=None, ranks=())
        self.assertFalse(census.config_ok)
        result = census.ingest({
            'kind': 'chunk_open', 'observed_ns': 10,
            'clock_basis': nm.CLOCK_BASIS, 'engine_epoch': 0,
            'route': 'split', 'route_generation': 1, 'transaction': 1})
        self.assertEqual(result, {'accepted': False,
                                  'reason': 'measurement-config-invalid'})
        evidence = census.evidence(1, 100, now_ns=101)
        self.assertEqual(evidence['status'], 'INCONCLUSIVE')
        self.assertFalse(evidence['construction_valid'])
        self.assertIn('config-invalid:engine-epoch', evidence['reasons'])
        # The source join needs a declared stage partition; guessing one (or
        # assuming a single stage) is refused rather than invented.
        partitionless = nm.NativeMeasurementCensus(
            engine_epoch=7, ranks=(0, 1), stage_ranks=None)
        self.assertFalse(partitionless.config_ok)
        self.assert_inconclusive_config(partitionless,
                                        'config-stage-partition-missing')
        overlapping = nm.NativeMeasurementCensus(
            engine_epoch=7, ranks=(0, 1, 2, 3),
            stage_ranks={0: (0, 1), 1: (1, 2, 3)})
        self.assert_inconclusive_config(overlapping, 'config-rank-in-two-stages')
        partial = nm.NativeMeasurementCensus(
            engine_epoch=7, ranks=(0, 1, 2),
            stage_ranks={0: (0, 1), 1: (2, 3)})
        self.assert_inconclusive_config(
            partial, 'config-stage-member-outside-ranks')

    def assert_inconclusive_config(self, census, reason):
        evidence = census.evidence(1, 100, now_ns=101)
        self.assertEqual(evidence['status'], 'INCONCLUSIVE')
        self.assertIn(reason, evidence['reasons'])

    def test_bounds_are_stated_as_telemetry_capacities(self):
        census = census_for(make_native_stack())
        stats = census.stats()
        self.assertEqual(stats['capacity']['max_requests'], nm.MAX_REQUESTS)
        self.assertEqual(stats['capacity']['max_chunks_per_request'],
                         nm.MAX_CHUNKS_PER_REQUEST)
        self.assertEqual(stats['capacity']['max_chunks_total'],
                         nm.MAX_REQUESTS * nm.MAX_CHUNKS_PER_REQUEST)
        self.assertEqual(nm.MAX_CHUNKS_PER_REQUEST, 256)
        self.assertEqual(nm.SEGMENTS_PER_CHUNK,
                         10 + nm.MODEL_LAYER_COUNT)
        self.assertEqual(stats['capacity']['segment_identity'],
                         '(stage, op_seq)')
        self.assertEqual(stats['capacity']['plan_segments_per_chunk'],
                         nm.SEGMENTS_PER_CHUNK)
        self.assertIn('descriptive', stats['capacity']['plan_segments_basis'])
        self.assertEqual(set(nm.measurement_event_kinds()), {
            'chunk_open', 'segment', 'report', 'chunk_commit', 'pp_receipt',
            'source_hold', 'committed_frontier', 'cancel'})
        self.assertIn('shape_basis', stats['capacity'])
        self.assertIs(type(stats['host_bytes']['bytes']), int)
        self.assertEqual(stats['projected_host_bytes_at_capacity']['chunks'],
                         nm.MAX_CHUNKS_TOTAL)
        self.assertIn('per_chunk_bytes',
                      stats['projected_host_bytes_at_capacity'])

    def test_module_is_pure_cpu_with_no_engine_imports(self):
        source = (ROOT / 'patches/s5_engine/native_measurement.py').read_text()
        for forbidden in ('import torch', 'import vllm', 'import socket',
                          'import subprocess', 'from vllm', 'cuda',
                          'engine_driver', 'native_overlap'):
            self.assertNotIn(forbidden, source)
        self.assertTrue(hasattr(nm, 'MEASUREMENT_BASIS'))
        self.assertNotIn('journal', nm.MEASUREMENT_BASIS)

    def test_serialized_transport_is_retained_but_never_accepted(self):
        stack, entries = self.standard()
        events = copy.deepcopy(entries)
        for entry in events:
            if entry['kind'] == 'segment' and entry['transaction'] == 1:
                entry['transport'] = 'serialized'
                break
        _census, evidence = classify(stack, events)
        self.assert_refused(evidence,
                            'chunk-1:serialized-or-unproved-transport')
        self.assertIs(evidence['serialized_transport'], True)
        rows = [row for row in evidence['segments']
                if row['transaction'] == 1]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['transport'], 'serialized')
        self.assertFalse(rows[0]['accepted'])

    def test_segment_op_conflict_refuses(self):
        stack, entries = self.standard()
        events = copy.deepcopy(entries)
        layer = next(entry for entry in events
                     if entry['kind'] == 'segment'
                     and entry['transaction'] == 1
                     and entry['stage'] == 0
                     and entry['segment_kind'] == 'layer')
        conflicting = copy.deepcopy(layer)
        conflicting['observed_ns'] = max(
            entry['observed_ns'] for entry in events) + 2
        conflicting['layer_start'] = layer['layer_start'] + 1
        conflicting['layer_end'] = layer['layer_end'] + 1
        events.append(conflicting)
        _census, evidence = classify(stack, events)
        self.assert_refused(evidence, 'chunk-1:segment-op-conflict')

    def test_standalone_cancel_without_open_is_retained(self):
        stack = make_native_stack()
        entries = capture_record(stack)
        stack.driver.stop('abort')
        cancels = kind_entries(entries, 'cancel')
        self.assertEqual(len(cancels), 1)
        self.assertTrue(cancels[0]['never_submitted'])
        census, evidence = classify(stack, cancels)
        self.assertEqual(evidence['status'], 'INCONCLUSIVE')
        self.assertNotIn('cancel-without-open', reasons(evidence))
        self.assertEqual(len(evidence['cancellations']), 1)
        self.assertEqual(evidence['cancellations'][0]['transaction'],
                         cancels[0]['transaction'])
        self.assertEqual(evidence['prefill']['completed_prefills'], 0)
        self.assertGreaterEqual(census.stats()['retained']
                                ['standalone_cancellations'], 1)



class RootWindowIntegrationTests(unittest.TestCase):
    def fixture(self, *, full_tail=False):
        import test_e3_native_overlap as overlap
        stack = make_native_stack()
        core, frontend = overlap.core_for(stack)
        stack.driver.journal_limit = 32
        stack.driver._journal = deque(maxlen=32)
        stack.drive(lambda: stack.driver._split_ready)
        begin = stack.driver._status_record()
        start = time.monotonic_ns()
        entries = run_journal(stack)
        end = time.monotonic_ns()
        if full_tail:
            stack.drive(lambda: not stack.driver.active)
        raw = stack.driver.overlap_evidence(start, end)
        return stack, frontend, begin, start, end, raw, entries

    def test_real_driver_selects_complete_census_not_evicted_tail(self):
        stack, _, begin, start, end, raw, _ = self.fixture()
        original = copy.deepcopy(raw)
        view, binding = nm.select_window_evidence(raw, begin, start, end)
        self.assertTrue(binding['valid'], binding)
        self.assertTrue(raw['measurement']['completeness']['complete'])
        self.assertEqual(len(view['segments']), 3)
        self.assertEqual(len(raw['segments']), 0)
        self.assertFalse(raw['journal']['complete_in_window'])
        self.assertGreater(raw['journal']['dropped'], 0)
        self.assertEqual(raw, original)
        self.assertIsNot(view, raw)

    def test_complete_flag_cannot_replace_bound_source_data(self):
        _, _, begin, start, end, raw, _ = self.fixture()
        for field, value in [('measurement', None), ('engine_epoch', begin['engine_epoch'] + 1)]:
            bad = copy.deepcopy(raw)
            bad[field] = value
            _, result = nm.select_window_evidence(bad, begin, start, end)
            self.assertFalse(result['valid'], result)
        for field, value in [('segments', []), ('completions', []),
                             ('groups', []), ('start_ns', start - 1),
                             ('engine_epoch', begin['engine_epoch'] + 1),
                             ('route_generations', [begin['route_generation'] + 1])]:
            bad = copy.deepcopy(raw)
            bad['measurement'][field] = value
            _, result = nm.select_window_evidence(bad, begin, start, end)
            self.assertFalse(result['valid'], (field, result))

    def test_full_terminal_tail_requires_real_all_rank_transition_acks(self):
        _, _, begin, start, end, raw, _ = self.fixture(full_tail=True)
        self.assertEqual(raw['route'], 'full')
        view, binding = nm.select_window_evidence(raw, begin, start, end)
        self.assertTrue(binding['valid'], binding)
        self.assertEqual(binding['route_basis'], 'queried-all-rank-terminal-full-tail')
        self.assertEqual(len(view['segments']), 3)
        mutations = (
            lambda r: r.update(transition_acks=[]),
            lambda r: r.update(native_owned=True),
            lambda r: r['transition_acks'][0].update(drained_streams=0),
            lambda r: r['transition_acks'][0].update(applied_bank=999),
            lambda r: r['transition_acks'][0].update(applied_route_generation=999),
            lambda r: r['transition_acks'][0].update(observed_ns=start - 1),
            lambda r: r['transition_acks'][0].update(engine_epoch=999),
            lambda r: r['transition_acks'][0].update(applied_storage_epochs=[]),
        )
        for mutate in mutations:
            bad = copy.deepcopy(raw)
            mutate(bad)
            _, result = nm.select_window_evidence(bad, begin, start, end)
            self.assertFalse(result['valid'], result)

    def test_same_uid_other_generation_cannot_supply_the_completion(self):
        import native_overlap_metrics as metrics
        segment = dict(request='reused', request_generation=1,
                       start_monotonic_ns=100, chunk_tokens=3, accepted=True)
        completions = [dict(request='reused', request_generation=2,
                            queried=True, state_write_ns=200)]
        result = metrics._window('fresh-prefill', metrics.FRESH_BASIS,
                                 [segment], completions)
        self.assertFalse(result['valid'], result)
        completions.append(dict(request='reused', request_generation=1,
                                queried=True, state_write_ns=150))
        result = metrics._window('fresh-prefill', metrics.FRESH_BASIS,
                                 [segment], completions)
        self.assertTrue(result['valid'], result)
        self.assertEqual(result['end_ns'], 150)


    def session_fixture(self, *, mutate=None, phase=None):
        import asyncio
        import test_e3_native_overlap as overlap
        import native_overlap_control as control_module
        import native_overlap_session as session_module
        stack = make_native_stack()
        _core, frontend = overlap.core_for(stack)
        stack.driver.journal_limit = 32
        stack.driver._journal = deque(maxlen=32)
        stack.drive(lambda: stack.driver._split_ready)
        control = control_module.NativeControlClient(frontend, timeout_s=5)
        session = session_module.NativeOverlapSession(frontend, control=control)
        asyncio.run(session.execute({'schema': 1, 'action': 'overlap_begin',
                                     'run_id': 'native-measurement-session'}))
        if phase is not None:
            # Isolated negative: declaration alone cannot supply append work.
            session.begin_receipt['admission_intent'] = {'phase': phase}
        run_journal(stack)
        stack.drive(lambda: not stack.driver.active)
        if mutate is not None:
            original = control.overlap_evidence
            async def read(start, end):
                raw = await original(start, end)
                mutate(raw)
                return raw
            control.overlap_evidence = read
        result = asyncio.run(session.execute({'schema': 1, 'action': 'overlap_end',
                                              'run_id': 'native-measurement-session',
                                              'decode_progress': 4}))
        return stack, control, result

    def test_named_utility_session_uses_complete_source_rows_after_raw_loss(self):
        _, control, result = self.session_fixture()
        self.assertTrue(result['enabled'], result['reasons'])
        self.assertEqual(result['route'], 'full')
        self.assertEqual(control.last_snapshot['route'], 'split')
        self.assertEqual(len(result['segments']), 3)
        self.assertEqual(result['raw_journal_evidence']['segments'], [])
        self.assertFalse(result['journal']['complete_in_window'])
        self.assertGreater(result['journal']['dropped'], 0)
        self.assertTrue(result['measurement_binding']['valid'])
        self.assertEqual(result['measurement_binding']['route_basis'],
                         'queried-all-rank-terminal-full-tail')
        self.assertFalse(result['native_ready'])
        self.assertFalse(result['armable'])

    def test_stale_split_client_status_cannot_replace_full_return_proof(self):
        _, control, result = self.session_fixture(
            mutate=lambda raw: raw.update(transition_acks=[]))
        self.assertEqual(control.last_snapshot['route'], 'split')
        self.assertFalse(result['enabled'])
        self.assertIn('measurement-end-route-unproved', result['reasons'])
        self.assertFalse(result['measurement_binding']['valid'])
        self.assertEqual(result['route'], 'full')

    def test_missing_census_does_not_fall_back_to_partial_raw_success(self):
        _, _, result = self.session_fixture(
            mutate=lambda raw: raw.pop('measurement'))
        self.assertFalse(result['enabled'])
        self.assertIn('measurement-census-missing', result['reasons'])
        self.assertFalse(result['journal']['complete_in_window'])
        self.assertEqual(result['raw_journal_evidence']['segments'], [])

    def test_declared_w2_cannot_omit_native_append_work(self):
        _, _, result = self.session_fixture(phase='w2-overlap')
        self.assertFalse(result['enabled'])
        self.assertIn('native-append-window-unproved', result['reasons'])
        self.assertTrue(result['measurement_binding']['valid'])
        self.assertTrue(result['fresh_prefill']['valid'])
        self.assertFalse(result['append_prefill']['valid'])


if __name__ == '__main__':
    unittest.main()
