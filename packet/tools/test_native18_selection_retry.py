"""CPU cached-ready-peer proof after a shared selection submission boundary.

Uses the declared more-convenient CPU tensor/event/model/kernel leaves. This
proves existing idempotence mechanisms, not a completed native18 implementation
or device/box qualification.
"""
import copy
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'harness'),str(ROOT/'patches/s5_engine')]
import test_e3_native_submission_credit as c


class ReadyPeerRetry(c.CreditHarness,unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not c.REFERENCE.is_dir():raise RuntimeError('required source reference unavailable; retry proof NOT_RUN')
        super().setUpClass()

    def scenario(self,fast):
        slow=1-fast
        world=self.world(route='sparse',ranks=(0,1))
        for stack in world.stacks:stack.attention._qsa_stage_min_rows=64
        self.do(world,'prepare',source=world.lease)
        self.do(world,'prefix',paused=False)
        self.do(world,'layer',layer=0,paused=True)
        # Respect event causality: all producer work precedes selection copies.
        world.complete_all()
        boundary,initial=self.do(world,'layer',layer=0,paused=True)
        self.assertEqual({a.cursor.bucket for a in initial.values()},{0})
        world.by_rank[fast].complete_all()
        command,first=world.send('layer',layer=0)
        self.assertTrue(first[fast].accepted);self.assertTrue(first[fast].paused)
        self.assertTrue(first[slow].pending)
        self.assertEqual(first[fast].cursor.bucket,1)
        stack=world.by_rank[fast]
        credit=stack.driver.submission_credit(0)
        self.assertFalse(credit.can_submit)
        self.assertEqual(credit.pending_buckets,1)
        cached=copy.deepcopy(stack.dispatcher.accepted[command.op_seq][1])
        before=self.submission_state(world)
        before_kinds=copy.deepcopy(world.accepted)
        for _ in range(1100):
            acks=world.deliver(command)
            self.assertTrue(acks[fast].accepted)
            self.assertTrue(acks[slow].pending)
            self.assertEqual(acks[fast].op_seq,command.op_seq)
            self.assertEqual(acks[fast].cursor,first[fast].cursor)
        self.assertEqual(self.submission_state(world),before)
        self.assertEqual(world.accepted,before_kinds)
        self.assertEqual(stack.dispatcher.accepted[command.op_seq][1],cached)
        # Credit is still unavailable, yet duplicate identity returns the
        # cached acceptance: it cannot have launched/charged another bucket.
        self.assertFalse(stack.driver.submission_credit(0).can_submit)
        world.by_rank[slow].complete_all()
        final=world.deliver(command)
        self.assertTrue(all(a.accepted and a.paused for a in final.values()))
        self.assertEqual({a.cursor.bucket for a in final.values()},{1})
        for rank in (0,1):
            self.assertEqual(world.by_rank[rank].operation.stats['sparse_buckets'],1)
            self.assertEqual(world.by_rank[rank].operation.stats['selection_copies'],1)
            self.assertEqual(world.accepted[rank],[0,1,2,3,4])
        after=self.submission_state(world)
        for _ in range(32):world.deliver(command)
        self.assertEqual(self.submission_state(world),after)
        return dict(fast_rank=fast,identical_retries=1100,post_join_retries=32,
            op_seq=command.op_seq,descriptor_unchanged=True,cached_ack_unchanged=True,
            credit_unavailable_during_duplicate=True,no_new_submissions=True,
            per_rank_sparse_buckets=[world.by_rank[r].operation.stats['sparse_buckets'] for r in (0,1)],
            per_rank_accepted=world.accepted,source_predicate_methods_patched=False,
            source_only=True,box_evidence=False)

    def test_left_ready_peer_retry(self):self.scenario(0)
    def test_right_ready_peer_retry(self):self.scenario(1)


def capture():
    ReadyPeerRetry.setUpClass();rows=[]
    for rank in (0,1):
        case=ReadyPeerRetry('runTest');case.setUp()
        try:rows.append(case.scenario(rank))
        finally:case.doCleanups()
    return rows

if __name__=='__main__':unittest.main()
