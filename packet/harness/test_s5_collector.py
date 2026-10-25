"""The twelve E0 standalone collector scenarios. No vLLM/GPU dependency."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'patches/s5_engine'))
from collector_prototype import (Collector, FakeTransport, Transaction, Reply,
                                 NotReady, ProtocolError, WorkerError, Backpressure,
                                 head_of_line_demo)


class CollectorCases(unittest.TestCase):
    def setUp(self):
        self.t=FakeTransport();self.c=Collector(self.t)
        self.tx=Transaction(1,'synthetic',1);self.c.open(self.tx)

    def answer(self, call, value=None, now=0):
        for r in self.c.calls[call].ranks:self.t.reply(self.c,call,r,value)
        self.t.advance(now);self.c.poll(64)

    def test_01_empty_poll_is_bounded_and_has_no_result_side_effect(self):
        i=self.c.issue('decode',ranks=(3,))
        self.assertEqual(self.c.poll(),0);self.assertEqual(self.t.reads,4)
        with self.assertRaises(NotReady):self.c.result(i)
        self.assertEqual(self.t.reads,4)
        # One response-queue consumer; stock FutureWrapper cannot race it.
        with self.assertRaises(ProtocolError):Collector(self.t)
        with self.assertRaises(ProtocolError):self.t.try_recv(3,object())

    def test_02_unique_reply_rank_and_legitimate_none_output(self):
        i=self.c.issue('decode',ranks=(3,),aggregate=lambda xs:xs[0])
        self.assertTrue(all(not self.c.expected[r] for r in (0,1,2)))
        with self.assertRaises(ProtocolError):self.t.reply(self.c,i,0)
        self.answer(i,None);self.assertIsNone(self.c.result(i))
        self.assertTrue(self.c.calls[i].ready)

    def test_03_all_rank_ack_is_not_gpu_completion_or_commit(self):
        i=self.c.issue('submit',transaction=self.tx)
        self.answer(i,'submitted')
        self.assertEqual(self.c.result(i),['submitted']*4)
        self.assertEqual(self.c.leased,4)
        with self.assertRaises(NotReady):self.c.commit(self.tx)

    def test_04_slow_rank_does_not_block_ready_later_decode_result(self):
        older=self.c.issue('submit',transaction=self.tx)
        later=self.c.issue('decode',ranks=(3,))
        for r in self.t.ranks:self.t.reply(self.c,older,r,'submitted',ready_at=10 if r==0 else 0)
        self.t.reply(self.c,later,3,'decoded',ready_at=0)
        self.t.advance(0);self.c.poll(64)
        with self.assertRaises(NotReady):self.c.result(older)
        self.assertEqual(self.c.result(later),['decoded'])
        self.t.advance(10);self.c.poll()
        self.assertEqual(self.c.result(older),['submitted']*4)

    def test_05_decode_finishes_before_older_prefill_gpu_completion(self):
        s=self.c.issue('submit',transaction=self.tx);self.answer(s,'submitted')
        d=self.c.issue('decode',ranks=(3,));self.answer(d,'decoded')
        self.assertEqual(self.c.result(d),['decoded']);self.assertEqual(self.c.leased,4)
        p=self.c.issue('poll',transaction=self.tx);self.answer(p,(s,))
        self.assertEqual(self.c.leased,0);self.c.commit(self.tx)
        with self.assertRaises(ProtocolError):self.c.commit(self.tx)

    def test_06_reverse_gpu_completion_does_not_reorder_rpc_queues(self):
        s=self.c.issue('submit',transaction=self.tx);self.answer(s,'submitted')
        # Completion polls and the decode command are issued in this order.
        p=self.c.issue('poll',transaction=self.tx)
        d=self.c.issue('decode',ranks=(3,))
        for r in self.t.ranks:self.t.reply(self.c,p,r,(s,),ready_at=0)
        self.t.reply(self.c,d,3,'decoded',ready_at=10)
        self.t.advance(0);self.c.poll(64);self.c.commit(self.tx)
        with self.assertRaises(NotReady):self.c.result(d)
        self.t.advance(10);self.c.poll();self.assertEqual(self.c.result(d),['decoded'])

    def test_07_async_output_fifo_head_of_line_is_not_cured_by_collector(self):
        r=head_of_line_demo()
        self.assertEqual(r['before']['ready_ack_ranks'],[0,1,2])
        self.assertFalse(r['before']['submit_ack_ready'])
        self.assertTrue(r['after']['submit_ack_ready'])
        self.assertTrue(r['after']['decode_ready'])
        self.assertEqual(r['after']['leased'],4);self.assertFalse(r['after']['committed'])

    def test_08_aggregation_order_errors_and_bounded_submissions(self):
        i=self.c.issue('decode',ranks=(3,1,0,2),aggregate=lambda xs:tuple(xs))
        for r in (2,0,1,3):self.t.reply(self.c,i,r,r)
        self.t.advance(0);self.c.poll(64);self.assertEqual(self.c.result(i),(3,1,0,2))
        fail=self.c.issue('decode',ranks=(3,))
        self.t.reply(self.c,fail,3,failed=True);self.t.advance(0);self.c.poll()
        with self.assertRaises(WorkerError):self.c.result(fail)
        limited=Collector(FakeTransport(),max_pending=1)
        limited.issue('decode',ranks=(3,))
        with self.assertRaises(Backpressure):limited.issue('decode',ranks=(3,))

    def test_09_duplicate_or_stale_generation_cannot_release_leases(self):
        s=self.c.issue('submit',transaction=self.tx);self.answer(s,'submitted')
        p=self.c.issue('poll',transaction=self.tx,ranks=(0,))
        wrong=Transaction(1,'synthetic',0)
        self.t.workers[0].wire.append(Reply(p,0,'poll',wrong,(s,)))
        with self.assertRaises(ProtocolError):self.c.poll()
        self.assertEqual(self.c.leased,4);self.assertTrue(self.c.poisoned)
        with self.assertRaises(ProtocolError):self.c.commit(self.tx)

    def test_10_unexpected_reply_order_type_and_duplicate_completion(self):
        for bad in ('order','type','duplicate'):
            t=FakeTransport();c=Collector(t);c.open(self.tx)
            s=c.issue('submit',transaction=self.tx)
            for r in t.ranks:t.reply(c,s,r,'submitted')
            t.advance(0);c.poll(64)
            p=c.issue('poll',transaction=self.tx,ranks=(0,))
            q=c.issue('decode',ranks=(0,))
            reply=Reply(q,0,'decode') if bad=='order' else Reply(p,0,'submit',self.tx) if bad=='type' else Reply(p,0,'poll',self.tx,(s,s))
            t.workers[0].wire.append(reply)
            with self.assertRaises(ProtocolError):c.poll()
            self.assertEqual(c.leased,4)

    def test_11_cancellation_keeps_leases_until_completed_and_blocks_commit(self):
        s=self.c.issue('submit',transaction=self.tx);self.answer(s,'submitted')
        self.c.cancel(self.tx);self.assertEqual(self.c.leased,4)
        new=Transaction(1,'synthetic',2)
        with self.assertRaises(ProtocolError):self.c.open(new)
        with self.assertRaises(ProtocolError):self.c.issue('submit',transaction=self.tx)
        p=self.c.issue('poll',transaction=self.tx);self.answer(p,(s,))
        self.assertEqual(self.c.leased,0)
        with self.assertRaises(ProtocolError):self.c.commit(self.tx)
        self.c.open(new)

    def test_12_shutdown_is_bounded_and_missing_workers_quarantine_ownership(self):
        s=self.c.issue('submit',transaction=self.tx)
        for r in (0,1,2):self.t.reply(self.c,s,r,'submitted')
        self.t.advance(0);r=self.c.shutdown(polls=2)
        self.assertFalse(r['clean']);self.assertEqual(r['pending_replies'],1)
        self.assertEqual(r['quarantined_leases'],4)
        with self.assertRaises(ProtocolError):self.c.issue('decode',ranks=(3,))
        with self.assertRaises(ProtocolError):self.c.commit(self.tx)
        self.assertTrue(Collector(FakeTransport()).shutdown(polls=0)['clean'])


if __name__=='__main__':unittest.main()
