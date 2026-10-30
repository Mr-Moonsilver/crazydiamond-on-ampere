"""Counterfactual guards with real immutable engine wire identities."""
from pathlib import Path
from types import SimpleNamespace as NS
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'patches/s5_engine'))
from e3_types import TransactionId
from wire_v1 import SegmentDescriptor,SegmentCommand,SegmentAcceptance,SegmentKind
from native34_control import Arm,ProtocolRefused


def command():
    tx=TransactionId(7,'private-request',0,0,1024,100)
    d=SegmentDescriptor(tx,SegmentKind.LAYER,0,layer_start=3,layer_end=4)
    return SegmentCommand(d,5,(0,1))


def accepted(cmd,pending=False):
    return tuple(SegmentAcceptance(cmd.transaction,r,cmd.kind,cmd.stage,cmd.op_seq,
        accepted=r<2 and not(pending and r==1),pending=pending and r==1,skipped=r>=2)
        for r in range(4))


def proofs(phase,arm):
    return [dict(worker_rank=r,qualification=dict(native34=dict(phase=phase,arm=arm,
        engine_epoch=7,drain_complete=True,prefix_complete=True,
        both_post_prefix_cohorts_complete=True))) for r in range(4)]


class Counterfactual(unittest.TestCase):
    def test_serial_arm_cannot_issue_decode_before_two_real_global_drains(self):
        c=command();a=Arm('S',7);a.admission(NS(transaction=c.transaction))
        self.assertTrue(a.blocks_stock());self.assertTrue(a.blocks_native())
        self.assertEqual(a.needed_checkpoint(),'before-prefix');a.submitted(1,'before-prefix')
        with self.assertRaises(ProtocolRefused):a.completed(1,proofs('before-prefix','S')[:3])
        a.completed(1,proofs('before-prefix','S'))
        self.assertTrue(a.blocks_stock());self.assertFalse(a.blocks_native())
        a.accepted_prefix(c,accepted(c));self.assertTrue(a.blocks_stock());self.assertTrue(a.blocks_native())
        a.submitted(2,'after-prefix');bad=proofs('after-prefix','S');bad[1]['qualification']['native34']['prefix_complete']=False
        with self.assertRaises(ProtocolRefused):a.completed(2,bad)
        a.completed(2,proofs('after-prefix','S'));self.assertFalse(a.blocks_stock());self.assertTrue(a.blocks_native())
        a.ordinary_tick();self.assertFalse(a.blocks_stock());a.ordinary_tick();self.assertTrue(a.blocks_stock())
        a.submitted(3,'post-decode');a.completed(3,proofs('post-decode','S'))
        self.assertTrue(a.record()['terminal'])

    def test_overlap_arm_has_no_join_fence_and_holds_bucket_after_same_prefix(self):
        c=command();a=Arm('O',7);a.admission(NS(transaction=c.transaction))
        self.assertIsNone(a.needed_checkpoint());self.assertFalse(a.blocks_stock());self.assertFalse(a.blocks_native())
        a.accepted_prefix(c,accepted(c,pending=True));self.assertEqual(a.stage,'PREFIX')
        a.accepted_prefix(c,accepted(c));self.assertEqual(a.stage,'POST_DECODE')
        self.assertTrue(a.blocks_native());self.assertFalse(a.blocks_stock())
        a.ordinary_tick();a.ordinary_tick();self.assertEqual(a.needed_checkpoint(),'post-decode')
        a.submitted(1,'post-decode');bad=proofs('post-decode','O');bad[2]['qualification']['native34']['both_post_prefix_cohorts_complete']=False
        with self.assertRaises(ProtocolRefused):a.completed(1,bad)
        a.completed(1,proofs('post-decode','O'))
        self.assertEqual([r['phase'] for r in a.records],['post-decode'])

    def test_wrong_epoch_state_or_reply_cannot_resume(self):
        with self.assertRaises(ProtocolRefused):Arm('anything',7)
        c=command();a=Arm('S',8)
        with self.assertRaises(ProtocolRefused):a.admission(NS(transaction=c.transaction))
        a=Arm('S',7)
        with self.assertRaises(ProtocolRefused):a.submitted(1,'after-prefix')
        a.admission(NS(transaction=c.transaction));a.submitted(1,'before-prefix')
        with self.assertRaises(ProtocolRefused):a.completed(2,proofs('before-prefix','S'))
        bad=proofs('before-prefix','S');bad[-1]['worker_rank']=0
        with self.assertRaises(ProtocolRefused):a.completed(1,bad)
        self.assertTrue(a.blocks_stock())


if __name__=='__main__':unittest.main()
