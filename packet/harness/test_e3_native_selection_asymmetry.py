"""Native17 CPU counterexample, NOT a device/box qualification.

Real QSA operation, generated helper, layer driver, lane, dispatcher and worker
bridge over the declared synthetic model/tensor/event/kernel leaves. Driver
reply validation uses an explicitly projected outstanding command, not native
startup. No production runtime method is patched. Faster completion of the
selection-copy event alone changes the accepted continuation boundary.
"""
import copy
import json
from pathlib import Path
import sys
import types
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'harness'),str(ROOT/'patches/s5_engine')]
import async_handoff as ah
import engine_driver as ed
import scheduler_hooks as sh
import test_e3_native_submission_credit as c
import wire_v1 as wire


class SelectionAsymmetry(c.CreditHarness,unittest.TestCase):
    def scenario(self,ready):
        world=self.world(route='sparse',ranks=(0,1))
        for rank in world.ranks:
            world.by_rank[rank].attention._qsa_stage_min_rows=64
        self.do(world,'prepare',source=world.lease)
        self.do(world,'prefix',paused=False)
        self.do(world,'layer',layer=0,paused=True)
        for rank in world.ranks:
            world.by_rank[rank].device.auto_ready=ready[rank]
        trackers=copy.deepcopy(world.trackers)
        command,acks=world.send('layer',layer=0)
        for tracker in trackers.values():self.assertEqual(tracker.propose(command.descriptor),command.op_seq)
        return world,command,acks,trackers

    def driver(self,world,command,trackers):
        # These host-only owners are deliberately not an EngineCore/transport.
        # Neither is invoked on the negative reply-validation path below.
        scheduler=types.SimpleNamespace()
        controller=sh.PrefillAdmissionController(scheduler,leases=sh.SchedulerLeases(),
            frontier=ah.CommittedFrontier(),engine_epoch=world.txn.engine_epoch,
            chunk_tokens=1024,minimum_stage_rows=64,ranks=(0,1,2,3))
        driver=ed.NativeSegmentDriver(controller,executor=object(),scheduler=scheduler,
            layer_count=48,layer_split=24,stage_of=((0,0),(1,0),(2,1),(3,1)))
        driver.tx=world.txn
        driver._command=command
        driver._targets=(0,1)
        driver.trackers=trackers
        driver.lanes={r:world.by_rank[r].lane for r in world.ranks}
        return driver

    def reproduce(self,ready):
        world,command,acks,trackers=self.scenario(ready)
        self.assertTrue(all(a.accepted and a.paused for a in acks.values()))
        self.assertEqual({a.cursor.phase for a in acks.values()},{'layer'})
        self.assertEqual({a.cursor.layer for a in acks.values()},{0})
        self.assertEqual({a.cursor.bucket for a in acks.values()},{0,1})
        driver=self.driver(world,command,trackers)
        skipped=tuple(wire.SegmentAcceptance(world.txn,r,command.kind,command.stage,
            command.op_seq,skipped=True,cursor=wire.CompletionCursor('idle',47,99)) for r in (2,3))
        with self.assertRaisesRegex(ed.DriverError,'TP workers disagree on the layer continuation'):
            driver._apply_segments(tuple(acks.values())+skipped)
        rows=[dict(rank=r,accepted=a.accepted,pending=a.pending,paused=a.paused,
            cursor=dict(phase=a.cursor.phase,layer=a.cursor.layer,bucket=a.cursor.bucket),
            records=world.by_rank[r].driver.records,
            selection_copies=world.by_rank[r].operation.stats['selection_copies'],
            sparse_buckets=world.by_rank[r].operation.stats['sparse_buckets']) for r,a in acks.items()]
        return dict(ready=ready,command_seq=command.op_seq,acceptances=rows,
            other_stage_skipped=[r.worker_rank for r in skipped],
            actual_driver_constructor=True,actual_reply_validation=True,
            runtime_methods_patched=False,engine_startup_constructed=False,
            tensor_event_model_kernel_leaves_synthetic=True,box_evidence=False)

    def test_left_peer_finishes_selection_before_first_readiness_query(self):
        row=self.reproduce((True,False))
        self.assertEqual([r['cursor']['bucket'] for r in row['acceptances']],[1,0])

    def test_right_peer_finishes_selection_before_first_readiness_query(self):
        row=self.reproduce((False,True))
        self.assertEqual([r['cursor']['bucket'] for r in row['acceptances']],[0,1])

    def test_both_selection_queries_pending_have_matching_continuations(self):
        world,command,acks,trackers=self.scenario((False,False))
        self.assertEqual({a.cursor.bucket for a in acks.values()},{0})
        self.assertTrue(all(a.paused and a.accepted for a in acks.values()))

    def test_both_selection_queries_complete_have_matching_continuations(self):
        world,command,acks,trackers=self.scenario((True,True))
        self.assertEqual({a.cursor.bucket for a in acks.values()},{1})
        self.assertTrue(all(a.paused and a.accepted for a in acks.values()))

    def test_other_stage_cannot_be_both_skipped_and_accepted(self):
        world,command,acks,trackers=self.scenario((False,False))
        with self.assertRaisesRegex(ValueError,'exactly one'):
            wire.SegmentAcceptance(world.txn,2,command.kind,command.stage,
                command.op_seq,accepted=True,skipped=True)


def capture():
    SelectionAsymmetry.setUpClass()
    case=SelectionAsymmetry('runTest');case.setUp()
    try:return case.reproduce((True,False))
    finally:case.doCleanups()


if __name__=='__main__':unittest.main()
