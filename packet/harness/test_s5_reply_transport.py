"""Owned reply receivers: CPU threads, a CPU subprocess, pinned MQ source.

No GPU/vLLM engine/native-shmem/ZMQ run. Exact source is exercised with controlled
buffers/socket doubles; this does not establish native IPC ABI or performance.
"""
from contextlib import contextmanager
from enum import Enum
import multiprocessing
from pathlib import Path
import pickle
from queue import Empty, Queue
import sys
import threading
import time
from types import MethodType, SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'patches/s5_engine'))
from reply_transport import ReplyTransport, TransportError, NotReady, Backpressure
from test_s5_transport_source import load, MQ


class Status(Enum):
    SUCCESS = 1
    FAILURE = 2


class Reader:
    def __init__(self, q=None):
        self.q = Queue() if q is None else q
        self.reads = []
        self.entered = threading.Event()

    def dequeue(self, timeout):
        self.reads.append((threading.get_ident(), timeout)); self.entered.set()
        try: return self.q.get(timeout=timeout)
        except Empty: raise TimeoutError from None


def cpu_worker(q):
    q.put((Status.SUCCESS, 'cpu-child-result'))


def await_condition(test, condition, bound=3):
    deadline = time.monotonic()+bound
    while time.monotonic() < deadline:
        if condition(): return
        time.sleep(0.002)
    test.fail('bounded CPU condition not reached')


class ReceiverTests(unittest.TestCase):
    def setUp(self):
        self.readers = {r: Reader() for r in range(4)}
        self.t = ReplyTransport(self.readers, Status.SUCCESS, handoff_idle=True, max_calls=4)

    def tearDown(self):
        # Release deliberately blocked test readers without touching a real MQ.
        for r in self.readers.values(): r.q.put((Status.SUCCESS, None))
        self.t.close(3)

    def ready(self, sequence):
        self.t.poll(32)
        return self.t._calls[sequence].ready

    def test_dispatcher_never_dequeues_while_receiver_waits(self):
        i=self.t.submit('decode', lambda _: None, ranks=(3,), timeout_s=2)
        self.assertTrue(self.readers[3].entered.wait(1))
        self.assertEqual(self.t.poll(), 0)
        with self.assertRaises(NotReady): self.t.take(i)
        self.assertEqual(len(self.readers[3].reads),1)
        self.assertNotEqual(self.readers[3].reads[0][0],threading.get_ident())
        self.assertTrue(all(not self.readers[r].reads for r in (0,1,2)))
        self.readers[3].q.put((Status.SUCCESS,None))
        await_condition(self,lambda:self.ready(i))
        values,lag=self.t.take(i)
        self.assertEqual(values,[None]);self.assertGreaterEqual(lag[3],0)
        self.assertTrue(self.t.close(1)['clean'])

    def test_slow_rank_does_not_hold_another_ranks_decode(self):
        a=self.t.submit('prefill-submit',lambda _:None,ranks=(0,),timeout_s=2)
        b=self.t.submit('decode',lambda _:self.readers[3].q.put((Status.SUCCESS,'decode')),ranks=(3,),timeout_s=2)
        await_condition(self,lambda:self.ready(b))
        self.assertEqual(self.t.take(b)[0],['decode'])
        with self.assertRaises(NotReady):self.t.take(a)

    def test_fifo_ack_lag_remains_and_aggregator_runs_on_dispatcher(self):
        a=self.t.submit('decode',lambda _:None,ranks=(3,),timeout_s=2)
        called=[]
        b=self.t.submit('prefill-submit',lambda _:None,timeout_s=2,
                        aggregate=lambda xs:(called.append(threading.get_ident()),xs)[1])
        for r in (0,1,2):self.readers[r].q.put((Status.SUCCESS,r))
        await_condition(self,lambda:(self.t.poll(8),len(self.t._calls[b].replies)==3)[1])
        self.assertFalse(self.t._calls[b].ready)
        self.readers[3].q.put((Status.SUCCESS,'decode'))
        self.readers[3].q.put((Status.SUCCESS,3))
        await_condition(self,lambda:self.ready(b))
        self.assertEqual(self.t.take(a)[0],['decode'])
        self.assertEqual(self.t.take(b)[0],[0,1,2,3])
        self.assertEqual(called,[threading.get_ident()])

    def test_ready_unclaimed_results_keep_credits(self):
        ids=[]
        for _ in range(4):
            ids.append(self.t.submit('decode',lambda i:self.readers[0].q.put((Status.SUCCESS,i)),ranks=(0,),timeout_s=2))
        await_condition(self,lambda:self.ready(ids[-1]))
        with self.assertRaises(Backpressure):self.t.submit('decode',lambda _:None)
        self.t.take(ids[0])
        self.t.submit('decode',lambda _:None,ranks=(0,),timeout_s=2)

    def test_registration_precedes_broadcast_and_send_failure_quarantines(self):
        def send(seq):
            self.assertIn(seq,self.t._calls)
            raise RuntimeError('test-send')
        with self.assertRaisesRegex(RuntimeError,'test-send'):
            self.t.submit('submit',send,timeout_s=0.1)
        with self.assertRaisesRegex(TransportError,'broadcast_failed'):self.t.poll()
        self.assertFalse(self.t.close(0)['clean'])

    def test_second_owner_and_non_dispatcher_access_rejected(self):
        with self.assertRaises(TransportError):ReplyTransport(self.readers,Status.SUCCESS,handoff_idle=True)
        with self.assertRaises(ValueError):ReplyTransport({0:Reader()},Status.SUCCESS,handoff_idle=False)
        errors=[]
        def wrong_thread():
            try:self.t.poll()
            except TransportError:errors.append(True)
        t=threading.Thread(target=wrong_thread);t.start();t.join(1)
        self.assertEqual(errors,[True])

    def test_failed_handoff_does_not_leave_other_reader_claimed(self):
        class FixedReader:
            __slots__=()
            def dequeue(self,timeout):raise AssertionError('must not start')
        r=Reader()
        with self.assertRaises(AttributeError):
            ReplyTransport({0:r,1:FixedReader()},Status.SUCCESS,handoff_idle=True)
        self.assertIsNone(r._flash_s5_receiver_owner)
        with self.assertRaises(ValueError):
            ReplyTransport({0:object()},Status.SUCCESS,handoff_idle=True)

    def test_worker_failure_keeps_payload_out_of_error(self):
        self.t.submit('submit',lambda _:self.readers[0].q.put((Status.FAILURE,'do-not-log-payload')),ranks=(0,),timeout_s=2)
        await_condition(self,lambda:self.t._failure is not None)
        with self.assertRaisesRegex(TransportError,'^worker_failure$'):self.t.poll()

    def test_timeout_never_retries_and_quarantines_even_ready_results(self):
        self.t.submit('submit',lambda _:None,ranks=(0,),timeout_s=0.03)
        await_condition(self,lambda:self.t._failure is not None)
        self.readers[0].q.put((Status.SUCCESS,'late'))
        with self.assertRaisesRegex(TransportError,'reply_timeout'):self.t.poll()
        self.assertEqual(len(self.readers[0].reads),1)
        self.assertGreater(self.readers[0].reads[0][1],0)
        self.assertFalse(self.t.close(1)['clean'])

    def test_bounded_close_does_not_claim_pending_work_retired(self):
        self.t.submit('submit',lambda _:None,ranks=(0,),timeout_s=2)
        self.assertTrue(self.readers[0].entered.wait(1))
        d=self.t.close(0)
        self.assertFalse(d['clean']);self.assertIn(0,d['live_receivers'])
        # A zero-duration join need not schedule the other idle threads yet.
        self.assertEqual(d['inflight_receives'],1)
        self.assertEqual(d['outstanding_calls'],1)
        self.readers[0].q.put((Status.SUCCESS,'late'))
        d=self.t.close(1)
        self.assertEqual(d['live_receivers'],[])
        self.assertFalse(d['clean'])

    def test_local_poll_budget_and_aggregate_failure(self):
        def bad(_):raise ValueError('aggregate failure')
        i=self.t.submit('submit',lambda _:[r.q.put((Status.SUCCESS,0)) for r in self.readers.values()],aggregate=bad,timeout_s=2)
        await_condition(self,lambda:self.t._done.qsize()==4)
        self.assertEqual(self.t.poll(1),1)
        self.assertFalse(self.t._calls[i].ready)
        with self.assertRaises(ValueError):self.t.poll(4)
        with self.assertRaisesRegex(TransportError,'aggregation_failed'):self.t.take(i)

    def test_cpu_process_reply(self):
        ctx=multiprocessing.get_context('spawn');q=ctx.Queue()
        reader=Reader(q);transport=ReplyTransport({0:reader},Status.SUCCESS,handoff_idle=True)
        child=ctx.Process(target=cpu_worker,args=(q,))
        try:
            i=transport.submit('decode',lambda _:child.start(),timeout_s=3)
            def ready():transport.poll();return transport._calls[i].ready
            await_condition(self,ready)
            self.assertEqual(transport.take(i)[0],['cpu-child-result'])
            child.join(3);self.assertEqual(child.exitcode,0)
            self.assertTrue(transport.close(1)['clean'])
        finally:
            transport.close(0)
            if child.pid:child.join(3)
            q.close();q.join_thread()


def source_reader(socket):
    """Two slots: delayed overflow first, immediately-ready inline second."""
    payload=pickle.dumps((Status.SUCCESS,'second'))
    inline=bytes([0,0,1])+len(payload).to_bytes(4,'big')+payload
    metadata=[bytearray([1,0]),bytearray([1,0])]
    @contextmanager
    def meta(i):yield metadata[i]
    @contextmanager
    def data(i):yield [b'\x01',inline][i]
    ns=dict(contextmanager=contextmanager,memory_fence=lambda:None,SPINLOOP_EXT_ENABLED=False,
            Any=object,zmq=SimpleNamespace(Socket=object),pickle=pickle,
            from_bytes_big=lambda b:int.from_bytes(b,'big'))
    recv=load(MQ,'recv',ns)
    ns['MessageQueue']=SimpleNamespace(recv=recv)
    acquire=load(MQ,'acquire_read',ns);dequeue=load(MQ,'dequeue',ns)
    reader=SimpleNamespace(_is_local_reader=True,_is_remote_reader=False,current_idx=0,
        local_reader_rank=0,local_socket=socket,shutting_down=False,
        ReadTimeoutWithWarnings=lambda **_:None,
        buffer=SimpleNamespace(get_metadata=meta,get_data=data,max_chunks=2),
        _spin_condition=SimpleNamespace(record_read=lambda:None))
    reader.acquire_read=MethodType(acquire,reader)
    reader.dequeue=MethodType(dequeue,reader)
    return reader,metadata


class ExactSourceTests(unittest.TestCase):
    def test_delayed_overflow_payload_keeps_fifo_off_dispatcher(self):
        entered=threading.Event();release=threading.Event();threads=[]
        def poll(timeout):
            threads.append(threading.get_ident());entered.set()
            return release.wait(timeout/1000)
        socket=SimpleNamespace(poll=poll,recv_multipart=lambda **_: [pickle.dumps((Status.SUCCESS,'first'))])
        reader,metadata=source_reader(socket)
        t=ReplyTransport({0:reader},Status.SUCCESS,handoff_idle=True)
        try:
            a=t.submit('decode',lambda _:None,timeout_s=2)
            b=t.submit('submit',lambda _:None,timeout_s=2)
            self.assertTrue(entered.wait(1))
            self.assertEqual(reader.current_idx,1);self.assertEqual(metadata[0][1],1)
            self.assertEqual(t.poll(),0)
            self.assertNotEqual(threads,[threading.get_ident()])
            release.set()
            await_condition(self,lambda:(t.poll(8),t._calls[b].ready)[1])
            self.assertEqual(t.take(a)[0],['first']);self.assertEqual(t.take(b)[0],['second'])
            self.assertTrue(t.close(1)['clean'])
        finally:release.set();t.close(1)

    def test_overflow_timeout_cannot_relabel_inline_next_slot(self):
        socket=SimpleNamespace(poll=lambda **_:False)
        reader,metadata=source_reader(socket)
        t=ReplyTransport({0:reader},Status.SUCCESS,handoff_idle=True)
        try:
            t.submit('decode',lambda _:None,timeout_s=1)
            await_condition(self,lambda:t._failure is not None)
            self.assertEqual(reader.current_idx,1)
            self.assertEqual(metadata[1][1],0)
            with self.assertRaisesRegex(TransportError,'reply_timeout'):t.poll()
            with self.assertRaises(TransportError):t.submit('submit',lambda _:None)
            self.assertFalse(t.close(1)['clean'])
        finally:t.close(1)


if __name__=='__main__':unittest.main()
