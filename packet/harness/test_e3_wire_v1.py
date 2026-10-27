"""Wire-v1 schema: frozen records, stage/layer/sequence rules, op_seq discipline.

CPU only. These tests pin the shared schema the engine driver and the worker
lanes both use; they are not native execution evidence.
"""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'patches/s5_engine'))

import wire_v1 as w
from lane_chunk import ChunkAdmission
from e3_types import (
    CompletionKind,
    CompletionProof,
    DependencyRole,
    E3ContractError,
    LaneId,
    LaneRole,
    OperationTicket,
    TransactionId,
)

UID = 'req-not-for-logs'


def txn(*, chunk_start=0, chunk_end=8, txn_seq=1, generation=2):
    return TransactionId(0, UID, generation, chunk_start, chunk_end, txn_seq)


def lane(rank=0, role=LaneRole.PREFILL, epoch=3, context=7):
    return LaneId(rank, role, epoch, context)


def ticket(transaction, rank=0, op_seq=0, role=DependencyRole.PRODUCER):
    return OperationTicket(transaction, rank, lane(rank), op_seq, role)


def proof(transaction, rank=0, op_seq=0, role=DependencyRole.PRODUCER,
          kind=CompletionKind.GPU_EVENT, completion_seq=1):
    return CompletionProof(ticket(transaction, rank, op_seq, role), kind,
                           completion_seq)


def admission(**overrides):
    transaction = txn()
    values = dict(
        transaction=transaction, prompt_end=9, state_epoch=4, committed_end=0,
        block_tables=((1, 2), (3, 4)), slot_mappings=((0, 1, 2, 3), (0, 1, 2, 3)),
        new_block_ids_to_zero=(2, 4), cow_pairs=((1, 5),), input_ids=tuple(range(8)))
    values.update(overrides)
    return w.AdmissionRecord(**values)


def report(**overrides):
    transaction = txn()
    values = dict(
        transaction=transaction, worker_rank=0, completed_op_seqs=(0, 1),
        pending_op_seqs=(2,), completed_sequence=1,
        proofs=(proof(transaction, 0, 0),), cursor=w.CompletionCursor('layer', 3, 2),
        state_epoch=4, host_submit_ns=((0, 10), (1, 20)))
    values.update(overrides)
    return w.WorkerCompletionReport(**values)


class AdmissionRecordTests(unittest.TestCase):
    def test_valid_record_and_derived_facts(self):
        record = admission()
        self.assertEqual(record.chunk_tokens, 8)
        self.assertEqual(record.remaining_tokens, 1)
        self.assertEqual(record.flat_blocks(), (1, 2, 3, 4))
        self.assertEqual(record.align_accepted, 1)
        self.assertEqual(record.align_seed(4), -1)   # no previously committed state
        self.assertEqual(record.align_seed(8), -1)

    def test_repr_hides_uid_tokens_and_tables(self):
        text = repr(admission())
        self.assertNotIn(UID, text)
        self.assertNotIn('1, 2', text)
        self.assertNotIn('input_ids', text)
        self.assertIn('chunk_tokens', text)

    def test_last_prompt_token_is_never_in_the_chunk(self):
        with self.assertRaises(E3ContractError):
            admission(prompt_end=8, committed_end=0, transaction=txn(chunk_end=8))
        record = admission(prompt_end=10)
        self.assertEqual(record.remaining_tokens, 2)

    def test_committed_end_is_the_prior_frontier(self):
        with self.assertRaises(E3ContractError):
            admission(committed_end=9)

    def test_block_rows_are_full_and_non_empty(self):
        for rows in ((), (('1', 2),), ((-1,),)):
            with self.assertRaises(E3ContractError):
                admission(block_tables=rows)

    def test_input_ids_cover_exactly_the_chunk(self):
        with self.assertRaises(E3ContractError):
            admission(input_ids=(1, 2, 3))

    def test_state_epoch_is_validated(self):
        with self.assertRaises(E3ContractError):
            admission(state_epoch=0)
        with self.assertRaises(E3ContractError):
            admission(state_epoch=True)

    def test_cow_pairs(self):
        with self.assertRaises(E3ContractError):
            admission(cow_pairs=((4, 4),))
        with self.assertRaises(E3ContractError):
            admission(cow_pairs=((1, 5), (2, 5)))
        with self.assertRaises(E3ContractError):
            admission(cow_pairs=((1, 5, 6),))

    def test_frozen(self):
        with self.assertRaises(Exception):
            admission().prompt_end = 12

    def test_repeated_tokens_and_null_group_rows_are_preserved(self):
        record = admission(input_ids=(42,) * 8, ngram_context=(7, 7),
                           block_tables=((), (0, 0, 3), (4,)))
        self.assertEqual(record.block_tables, ((), (0, 0, 3), (4,)))
        self.assertEqual(record.flat_blocks(), (0, 0, 3, 4))

    def test_align_seed_uses_previous_frontier_not_future_output(self):
        t = txn(chunk_start=4048, chunk_end=5072)
        record = admission(transaction=t, prompt_end=6000, committed_end=4048,
                           input_ids=(3,) * 1024)
        self.assertEqual(record.align_seed(4048), 0)
        self.assertEqual(record.remaining_tokens, 928)
        with self.assertRaises(E3ContractError):
            admission(transaction=t, prompt_end=6000, committed_end=5072,
                      input_ids=(3,) * 1024)


class SourceLeaseDescriptorTests(unittest.TestCase):
    """The transport half of the engine's source ownership, never a bool."""

    def descriptor(self, **overrides):
        values = dict(transaction=txn(), state_epoch=4,
                      block_ids=(1, 2, 3, 5), consumer_ranks=(0, 1))
        values.update(overrides)
        return w.SourceLeaseDescriptor(**values)

    def test_pages_and_ranks_are_canonical(self):
        with self.assertRaises(E3ContractError):
            self.descriptor(block_ids=(2, 1))
        with self.assertRaises(E3ContractError):
            self.descriptor(block_ids=())
        with self.assertRaises(E3ContractError):
            self.descriptor(block_ids=(0, 1))
        with self.assertRaises(E3ContractError):
            self.descriptor(consumer_ranks=(1, 0))
        with self.assertRaises(E3ContractError):
            self.descriptor(consumer_ranks=())

    def test_frame_must_match_transaction_epoch_pages_and_ranks(self):
        descriptor = self.descriptor()
        self.assertTrue(descriptor.validate_frame(admission(), (0, 1)))
        with self.assertRaises(E3ContractError):
            descriptor.validate_frame(admission(state_epoch=5), (0, 1))
        with self.assertRaises(E3ContractError):
            descriptor.validate_frame(admission(transaction=txn(txn_seq=2)), (0, 1))
        # Page 5 is the retained CoW source: it is absent from the rows and
        # named only by cow_pairs. Dropping that pair must refuse the read.
        with self.assertRaises(E3ContractError):
            descriptor.validate_frame(
                admission(cow_pairs=()), (0, 1))
        with self.assertRaises(E3ContractError):
            descriptor.validate_frame(admission(), (0, 2))
        with self.assertRaises(E3ContractError):
            descriptor.validate_frame(
                admission(block_tables=((1, 2), (3,)), cow_pairs=((1, 4),)),
                (0, 1))
        with self.assertRaises(E3ContractError):
            descriptor.validate_frame(object(), (0, 1))

    def test_worker_lane_frame_is_the_same_proof(self):
        frame = ChunkAdmission(
            transaction=txn(), prompt_end=9, input_ids=tuple(range(8)),
            ngram_context=(), block_tables=((1, 2), (3, 4)),
            cow_pairs=((1, 5),), state_epoch=4)
        self.assertTrue(self.descriptor().validate_frame(frame, (0, 1)))
        stale = ChunkAdmission(
            transaction=txn(), prompt_end=9, input_ids=tuple(range(8)),
            ngram_context=(), block_tables=((1, 2), (3, 4)),
            cow_pairs=((1, 5),), state_epoch=5)
        with self.assertRaises(E3ContractError):
            self.descriptor().validate_frame(stale, (0, 1))


class SegmentDescriptorTests(unittest.TestCase):
    def test_kind_stage_matrix(self):
        transaction = txn()
        self.assertEqual(w.SegmentDescriptor(transaction, w.SegmentKind.PREPARE, 0).kind,
                         w.SegmentKind.PREPARE)
        for bad in ((w.SegmentKind.PP_SEND, 1), (w.SegmentKind.PP_RECEIVE, 0),
                    (w.SegmentKind.PREPARE, 2)):
            kind, stage = bad
            values = dict(seq=0) if kind in (w.SegmentKind.PP_SEND,
                                             w.SegmentKind.PP_RECEIVE) else {}
            with self.assertRaises(E3ContractError):
                w.SegmentDescriptor(transaction, kind, stage, **values)

    def test_layer_segments_are_single_half_open(self):
        transaction = txn()
        segment = w.SegmentDescriptor(transaction, w.SegmentKind.LAYER, 0,
                                      layer_start=7, layer_end=8)
        self.assertEqual(segment.layers, (7,))
        with self.assertRaises(E3ContractError):
            w.SegmentDescriptor(transaction, w.SegmentKind.LAYER, 0,
                                layer_start=7, layer_end=9)
        with self.assertRaises(E3ContractError):
            w.SegmentDescriptor(transaction, w.SegmentKind.LAYER, 0,
                                layer_start=7)
        with self.assertRaises(E3ContractError):
            w.SegmentDescriptor(transaction, w.SegmentKind.PREFIX, 0,
                                layer_start=7, layer_end=8)

    def test_pp_seq_is_explicit_and_kind_only(self):
        transaction = txn()
        send = w.SegmentDescriptor(transaction, w.SegmentKind.PP_SEND, 0, seq=5)
        self.assertTrue(send.is_pp)
        self.assertEqual(send.seq, 5)
        with self.assertRaises(E3ContractError):
            w.SegmentDescriptor(transaction, w.SegmentKind.PP_SEND, 0)
        with self.assertRaises(E3ContractError):
            w.SegmentDescriptor(transaction, w.SegmentKind.PREPARE, 0, seq=0)
        with self.assertRaises(E3ContractError):
            w.SegmentDescriptor(transaction, w.SegmentKind.PP_RECEIVE, 1, seq=True)

    def test_dependencies_must_share_the_transaction(self):
        transaction = txn()
        other = txn(txn_seq=2)
        good = w.SegmentDescriptor(transaction, w.SegmentKind.PREFIX, 0,
                                   depends_on=(ticket(transaction),))
        self.assertTrue(good.satisfied((ticket(transaction),)))
        self.assertFalse(good.satisfied(()))
        with self.assertRaises(E3ContractError):
            w.SegmentDescriptor(transaction, w.SegmentKind.PREFIX, 0,
                                depends_on=(ticket(other),))

    def test_repr_hides_uid(self):
        text = repr(w.SegmentDescriptor(txn(), w.SegmentKind.LAYER, 1,
                                        layer_start=2, layer_end=3))
        self.assertNotIn(UID, text)
        self.assertIn('layer=2', text)


class CompletionReportTests(unittest.TestCase):
    def test_valid_report(self):
        value = report()
        self.assertFalse(value.failed)
        self.assertTrue(value.has_proof(CompletionKind.GPU_EVENT))
        self.assertFalse(value.state_written())
        self.assertNotIn(UID, repr(value))

    def test_state_write_proof_needs_commit_role(self):
        transaction = txn()
        commit = proof(transaction, 1, 1, DependencyRole.COMMIT,
                       CompletionKind.STATE_WRITE)
        value = report(worker_rank=1, completed_op_seqs=(1,), pending_op_seqs=(),
                       completed_sequence=1, proofs=(commit,), host_submit_ns=())
        self.assertTrue(value.state_written())
        producer = proof(transaction, 1, 1, DependencyRole.PRODUCER,
                         CompletionKind.STATE_WRITE)
        not_commit = report(worker_rank=1, completed_op_seqs=(1,),
                            pending_op_seqs=(), completed_sequence=1,
                            proofs=(producer,), host_submit_ns=())
        self.assertFalse(not_commit.state_written())

    def test_completed_and_pending_are_disjoint_and_ordered(self):
        with self.assertRaises(E3ContractError):
            report(pending_op_seqs=(1,))
        with self.assertRaises(E3ContractError):
            report(completed_op_seqs=(1, 0))
        with self.assertRaises(E3ContractError):
            report(completed_op_seqs=(0, 0))

    def test_watermark_is_not_behind_a_completed_op(self):
        with self.assertRaises(E3ContractError):
            report(completed_sequence=0)

    def test_proofs_must_belong_to_the_report(self):
        other = txn(txn_seq=9)
        with self.assertRaises(E3ContractError):
            report(proofs=(proof(other, 0, 0),))
        with self.assertRaises(E3ContractError):
            report(proofs=(proof(txn(), 1, 0),))
        with self.assertRaises(E3ContractError):
            report(proofs=(proof(txn(), 0, 5),))

    def test_host_submit_times_are_completed_and_ordered(self):
        with self.assertRaises(E3ContractError):
            report(host_submit_ns=((0, 10), (1, 20), (3, 30)))
        with self.assertRaises(E3ContractError):
            report(host_submit_ns=((1, 20), (0, 10)))

    def test_failed_code_is_bounded(self):
        self.assertTrue(report(failed_code='worker_failure').failed)
        with self.assertRaises(E3ContractError):
            report(failed_code='rpc_ack')
        with self.assertRaises(E3ContractError):
            report(failed_code=1)

    def test_cursor_phase(self):
        with self.assertRaises(E3ContractError):
            w.CompletionCursor('whole_forward', 0, 0)

    def test_watermark_and_op_sequence_are_independent(self):
        p = proof(txn(), op_seq=900, completion_seq=3)
        r = report(completed_op_seqs=(900,), pending_op_seqs=(901,),
                   completed_sequence=3, proofs=(p,), host_submit_ns=((901, 12),))
        self.assertEqual(r.completed_sequence, 3)

    def test_receive_event_and_consumer_proof_share_one_ticket(self):
        t = ticket(txn(), role=DependencyRole.HANDOFF)
        report(proofs=(CompletionProof(t, CompletionKind.GPU_EVENT, 1),
                       CompletionProof(t, CompletionKind.PP_CONSUMER, 2)),
               completed_sequence=2)


class TransitionTests(unittest.TestCase):
    def command(self, **overrides):
        values = dict(engine_epoch=4, expected_route_generation=2,
                      route_generation=3, target_bank=1,
                      storage_epochs=(5, 6), decode_replay_watermarks=(11, 12))
        values.update(overrides)
        return w.TransitionCommand(**values)

    def ack(self, command, **overrides):
        worker_rank = overrides.get('worker_rank', 0)
        watermark = (command.decode_replay_watermarks[worker_rank]
                     if worker_rank < len(command.decode_replay_watermarks) else 0)
        values = dict(engine_epoch=command.engine_epoch,
                      expected_route_generation=command.expected_route_generation,
                      route_generation=command.route_generation,
                      target_bank=command.target_bank,
                      storage_epochs=command.storage_epochs,
                      applied_route_generation=command.route_generation,
                      applied_bank=command.target_bank,
                      applied_storage_epochs=command.storage_epochs,
                      worker_rank=worker_rank,
                      decode_replay_watermark=watermark)
        values.update(overrides)
        return w.TransitionAck(**values)

    def test_generation_advances_by_one(self):
        with self.assertRaises(E3ContractError):
            self.command(route_generation=4)
        with self.assertRaises(E3ContractError):
            self.command(route_generation=2)

    def test_one_watermark_per_rank(self):
        with self.assertRaises(E3ContractError):
            self.command(decode_replay_watermarks=(1,))
        self.assertEqual(self.command(storage_epochs=(6, 5)).storage_epochs, (6, 5))

    def test_ack_must_echo_the_applied_command(self):
        command = self.command()
        self.assertTrue(self.ack(command).matches(command))
        with self.assertRaises(E3ContractError):
            self.ack(command, applied_bank=0)
        with self.assertRaises(E3ContractError):
            self.ack(command, applied_route_generation=4)
        with self.assertRaises(E3ContractError):
            self.ack(command, applied_storage_epochs=(5, 9))
        stale = self.ack(self.command(engine_epoch=3, expected_route_generation=2))
        self.assertFalse(stale.matches(command))

    def test_ack_reports_its_queried_census_not_an_echo(self):
        command = self.command()
        # The commanded value is a floor: the worker's own queried census may
        # have advanced while the split chunk ran.
        self.assertTrue(self.ack(command, decode_replay_watermark=20).matches(command))
        # A census behind the floor is a stale/storage-reuse answer.
        self.assertFalse(self.ack(command, decode_replay_watermark=10).matches(command))
        # A rank outside the command has no commanded floor to compare.
        self.assertFalse(self.ack(command, worker_rank=2,
                                  decode_replay_watermark=99).matches(command))


class BootstrapTests(unittest.TestCase):
    def command(self, **overrides):
        values = dict(engine_epoch=4, ranks=(0, 1))
        values.update(overrides)
        return w.BootstrapCommand(**values)

    def ack(self, command, **overrides):
        values = dict(engine_epoch=command.engine_epoch, worker_rank=0,
                      incarnation='worker-incarnation-a',
                      lane=lane(0, LaneRole.DECODE, epoch=5, context=9),
                      route_generation=0, decode_replay_watermark=3, bound=True)
        values.update(overrides)
        return w.BootstrapAck(**values)

    def test_command_names_the_engine_owner_and_ordered_ranks(self):
        command = self.command()
        self.assertEqual(command.engine_epoch, 4)
        with self.assertRaises(E3ContractError):
            self.command(engine_epoch=0)
        with self.assertRaises(E3ContractError):
            self.command(ranks=(1, 0))
        with self.assertRaises(E3ContractError):
            self.command(ranks=())
        self.assertNotIn('incarnation', repr(command))

    def test_ack_carries_retained_decode_storage_identity(self):
        command = self.command()
        self.assertTrue(self.ack(command).matches(command))
        self.assertFalse(self.ack(command, worker_rank=1,
                                  lane=lane(1, LaneRole.DECODE)).matches(
            self.command(ranks=(0,))))
        self.assertFalse(self.ack(command, engine_epoch=3).matches(command))
        with self.assertRaises(E3ContractError):
            self.ack(command, lane=lane(0, LaneRole.PREFILL))
        with self.assertRaises(E3ContractError):
            self.ack(command, lane=lane(1, LaneRole.DECODE))
        with self.assertRaises(E3ContractError):
            self.ack(command, bound='yes')
        # A refusal is representable but never a success.
        self.assertFalse(self.ack(command, bound=False).bound)
        self.assertNotIn('worker-incarnation-a', repr(self.ack(command)))


class OpenAndCommandTests(unittest.TestCase):
    def request(self):
        return w.OpenRequest(admission(), (0, 1), 4, 2)

    def test_open_request_identity(self):
        request = self.request()
        self.assertEqual(request.transaction, admission().transaction)
        self.assertNotIn(UID, repr(request))
        with self.assertRaises(E3ContractError):
            w.OpenRequest(admission(), (1, 0), 4, 2)
        with self.assertRaises(E3ContractError):
            w.OpenRequest(admission(), (0,), 5, 2)
        with self.assertRaises(E3ContractError):
            w.OpenRequest(admission(), (0,), 4, 0)

    def test_open_ack_must_match_the_request(self):
        request = self.request()
        ack = w.OpenAck(request.transaction, lane(1), 1, 4, 2, True)
        self.assertTrue(ack.matches(request))
        stale = w.OpenAck(request.transaction, lane(1, epoch=9), 1, 9, 2, True)
        self.assertFalse(stale.matches(request))
        with self.assertRaises(E3ContractError):
            w.OpenAck(request.transaction, lane(1), 2, 4, 2, True)
        with self.assertRaises(E3ContractError):
            w.OpenAck(request.transaction, lane(1), 1, 4, 2, 1)

    def descriptor(self):
        return w.SegmentDescriptor(txn(), w.SegmentKind.LAYER, 0,
                                   layer_start=1, layer_end=2)

    def test_command_is_typed_and_ordered(self):
        command = w.SegmentCommand(self.descriptor(), 3, (0, 1))
        self.assertEqual(command.kind, w.SegmentKind.LAYER)
        self.assertEqual(command.stage, 0)
        self.assertNotIn(UID, repr(command))
        with self.assertRaises(E3ContractError):
            w.SegmentCommand(self.descriptor(), 3, (1, 0))
        with self.assertRaises(E3ContractError):
            w.SegmentCommand(self.descriptor(), True, (0,))
        with self.assertRaises(E3ContractError):
            w.SegmentCommand(self.descriptor(), 3, ())

    def test_acceptance_has_exactly_one_state(self):
        command = w.SegmentCommand(self.descriptor(), 3, (0, 1))
        accepted = w.SegmentAcceptance(command.transaction, 0, command.kind,
                                       command.stage, 3, accepted=True)
        self.assertTrue(accepted.matches(command))
        pending = w.SegmentAcceptance(command.transaction, 0, command.kind,
                                      command.stage, 3, pending=True)
        self.assertFalse(pending.matches(w.SegmentCommand(self.descriptor(), 4, (0, 1))))
        skipped = w.SegmentAcceptance(command.transaction, 5, command.kind,
                                      command.stage, 3, skipped=True)
        self.assertFalse(skipped.matches(command))   # rank 5 is not addressed
        with self.assertRaises(E3ContractError):
            w.SegmentAcceptance(command.transaction, 0, command.kind, 0, 3)
        with self.assertRaises(E3ContractError):
            w.SegmentAcceptance(command.transaction, 0, command.kind, 0, 3,
                                accepted=True, pending=True)

    def test_acceptance_repr_has_no_uid(self):
        command = w.SegmentCommand(self.descriptor(), 3, (0,))
        text = repr(w.SegmentAcceptance(command.transaction, 0, command.kind,
                                        command.stage, 3, accepted=True))
        self.assertNotIn(UID, text)
        self.assertIn('accepted', text)

    def test_acceptance_can_carry_a_host_queried_report(self):
        command = w.SegmentCommand(self.descriptor(), 3, (0, 1))
        attached = report(completed_op_seqs=(0,), pending_op_seqs=(),
                          completed_sequence=1, host_submit_ns=())
        accepted = w.SegmentAcceptance(command.transaction, 0, command.kind,
                                       command.stage, 3, accepted=True,
                                       report=attached)
        self.assertIs(accepted.report, attached)
        self.assertNotIn(UID, repr(accepted))
        with self.assertRaises(E3ContractError):
            w.SegmentAcceptance(command.transaction, 0, command.kind,
                                command.stage, 3, accepted=True, report=object())
        with self.assertRaises(E3ContractError):
            w.SegmentAcceptance(command.transaction, 1, command.kind,
                                command.stage, 3, accepted=True, report=attached)
        other = txn(txn_seq=9)
        with self.assertRaises(E3ContractError):
            w.SegmentAcceptance(other, 0, command.kind, command.stage, 3,
                                accepted=True, report=attached)


class WorkerWireAdapterTests(unittest.TestCase):
    def test_explicit_command_adapter_preserves_pp_sequence(self):
        from lane_dispatch import SegmentCommand as Local
        d = w.SegmentDescriptor(txn(), w.SegmentKind.PP_RECEIVE, 1, seq=37)
        local = Local.from_descriptor(w.SegmentCommand(d, 12, (2, 3)))
        self.assertEqual((local.kind, local.seq, local.op_seq), ('pp_receive', 37, 12))
        self.assertEqual(local.transaction, d.transaction)

    def test_bare_descriptor_cannot_guess_the_operation_sequence(self):
        from lane_dispatch import SegmentCommand as Local
        with self.assertRaises(E3ContractError):
            Local.from_descriptor(w.SegmentDescriptor(txn(), w.SegmentKind.PREPARE, 0))


class OpSeqTrackerTests(unittest.TestCase):
    def setUp(self):
        self.transaction = txn()
        self.tracker = w.OpSeqTracker()
        self.prepare = w.SegmentDescriptor(self.transaction, w.SegmentKind.PREPARE, 0)
        self.prefix = w.SegmentDescriptor(self.transaction, w.SegmentKind.PREFIX, 0)

    def test_proposal_does_not_consume(self):
        self.assertEqual(self.tracker.propose(self.prepare), 0)
        self.assertEqual(self.tracker.last_accepted, -1)
        self.assertEqual(self.tracker.next_op_seq(), 0)

    def test_retry_reuses_the_same_sequence(self):
        first = self.tracker.propose(self.prepare)
        self.assertEqual(self.tracker.propose(self.prepare), first)
        with self.assertRaises(E3ContractError):
            self.tracker.propose(self.prefix)

    def test_acceptance_consumes_and_advances(self):
        self.assertTrue(self.tracker.accepted(0, self.prepare))
        self.assertIsNone(self.tracker.pending)
        self.assertEqual(self.tracker.last_accepted, 0)
        self.assertTrue(self.tracker.accepted(1, self.prefix))
        self.assertEqual(self.tracker.last_accepted, 1)

    def test_identical_replay_is_idempotent(self):
        self.assertTrue(self.tracker.accepted(0, self.prepare))
        self.assertFalse(self.tracker.accepted(0, self.prepare))
        self.assertEqual(self.tracker.replays, 1)
        self.assertEqual(self.tracker.last_accepted, 0)

    def test_replay_of_a_different_command_is_refused(self):
        self.tracker.accepted(0, self.prepare)
        with self.assertRaises(E3ContractError):
            self.tracker.accepted(0, self.prefix)

    def test_sequence_must_strictly_increase(self):
        with self.assertRaises(E3ContractError):
            self.tracker.accepted(1, self.prepare)
        self.tracker.accepted(0, self.prepare)
        with self.assertRaises(E3ContractError):
            self.tracker.accepted(0, self.prefix)

    def test_pending_refusal_consumes_nothing(self):
        op_seq = self.tracker.propose(self.prepare)
        # The worker answered "pending": no acceptance, no sequence consumed.
        self.assertEqual(self.tracker.propose(self.prepare), op_seq)
        self.assertEqual(self.tracker.last_accepted, -1)
        self.assertEqual(self.tracker.accepted(op_seq, self.prepare), True)
        self.assertEqual(self.tracker.last_accepted, op_seq)

    def test_start_sequence(self):
        tracker = w.OpSeqTracker(4)
        self.assertEqual(tracker.propose(self.prepare), 5)


class TerminalReleaseTests(unittest.TestCase):
    """The typed terminal field and the request-row closure it certifies."""

    def request(self, terminal, **overrides):
        values = dict(transaction=txn(), state_epoch=4,
                      completed_sequences=((0, 3), (1, 3)), terminal=terminal)
        values.update(overrides)
        return w.ReleaseRequest(**values)

    def test_terminality_is_derived_from_the_admission_not_a_caller_flag(self):
        self.assertTrue(admission().terminal)
        self.assertFalse(admission(prompt_end=10).terminal)
        self.assertTrue(admission(prompt_end=9, transaction=txn(chunk_end=8)).terminal)

    def test_release_requires_an_explicit_boolean_terminal_field(self):
        self.assertIs(self.request(False).terminal, False)
        self.assertIs(self.request(True).terminal, True)
        with self.assertRaises(E3ContractError):
            self.request(1)
        with self.assertRaises(E3ContractError):
            self.request(None)

    def test_repr_names_the_terminal_disposition(self):
        self.assertIn('terminal=False', repr(self.request(False)))
        self.assertIn('terminal=True', repr(self.request(True)))

    def test_nonterminal_and_terminal_ack_closure_shapes(self):
        transaction = txn()
        nonterminal = w.ReleaseAck(transaction, 0, 4, 3, released=True,
                                   request_closed=False)
        self.assertTrue(nonterminal.released)
        self.assertFalse(nonterminal.closed)
        terminal = w.ReleaseAck(transaction, 0, 4, 3, released=True,
                                request_closed=True)
        self.assertTrue(terminal.closed)
        # The row can never be closed without the storage release.
        with self.assertRaises(E3ContractError):
            w.ReleaseAck(transaction, 0, 4, 3, released=False, request_closed=True)


if __name__ == '__main__':
    unittest.main()
