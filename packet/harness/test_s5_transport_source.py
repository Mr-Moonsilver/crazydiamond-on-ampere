"""Execute pinned, exact-source method excerpts against CPU transport doubles.

No vLLM imports, CUDA, processes, sockets or GPU work. This checks control-flow
hazards, not MessageQueue/native-shmem ABI or real interprocess performance.
"""
import ast
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import pickle
from queue import Queue
import textwrap
import threading
from types import MethodType, SimpleNamespace
import unittest

ROOT=Path(__file__).resolve().parents[1]
SOURCES=json.loads((ROOT/'patches/s5_engine/transport_source.json').read_text())['modules']
MQ='distributed/device_communicators/shm_broadcast.py'
WORKER='v1/executor/multiproc_executor.py'


def load(module,name,namespace):
    row=SOURCES[module]['methods'][name]
    assert hashlib.sha256(row['source'].encode()).hexdigest()==row['sha256']
    exec(compile(row['source'],module+':'+name,'exec'),namespace)
    return namespace[name]


class SourceConstraints(unittest.TestCase):
    def test_fixtures_are_exact_pinned_method_excerpts(self):
        for rel,row in SOURCES.items():
            for name,method in row['methods'].items():
                self.assertEqual(hashlib.sha256(method['source'].encode()).hexdigest(),method['sha256'])
            p=ROOT/'reference/vllm-src'/rel
            if not p.exists():continue # Frozen excerpts still run outside this source checkout.
            text=p.read_text();self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),row['sha256'])
            cls=next(n for n in ast.parse(text).body if isinstance(n,ast.ClassDef) and n.name==row['class'])
            for name,method in row['methods'].items():
                node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
                start=min([node.lineno]+[n.lineno for n in node.decorator_list])
                code=textwrap.dedent(''.join(text.splitlines(keepends=True)[start-1:node.end_lineno]))
                self.assertEqual(code,method['source'])

    def test_zero_timeout_consumes_overflow_marker_before_payload_is_ready(self):
        metadata=[bytearray([1,0]),bytearray([1,0])]
        payload=pickle.dumps({'call':1})
        inline=bytes([0])+bytes([0,1])+len(payload).to_bytes(4,'big')+payload
        data=[b'\x01',inline];reads=[]
        @contextmanager
        def get_meta(i):yield metadata[i]
        @contextmanager
        def get_data(i):yield data[i]
        def recv(socket,timeout):
            reads.append(timeout)
            raise TimeoutError('payload not yet delivered')
        ns=dict(contextmanager=contextmanager,memory_fence=lambda:None,SPINLOOP_EXT_ENABLED=False,
                Any=object,MessageQueue=SimpleNamespace(recv=recv),pickle=pickle,
                from_bytes_big=lambda b:int.from_bytes(b,'big'))
        acquire=load(MQ,'acquire_read',ns);dequeue=load(MQ,'dequeue',ns)
        reader=SimpleNamespace(_is_local_reader=True,_is_remote_reader=False,current_idx=0,
            local_reader_rank=0,local_socket=object(),shutting_down=False,
            ReadTimeoutWithWarnings=lambda **_:None,
            buffer=SimpleNamespace(get_metadata=get_meta,get_data=get_data,max_chunks=2),
            _spin_condition=SimpleNamespace(record_read=lambda:None))
        reader.acquire_read=MethodType(acquire,reader)
        with self.assertRaises(TimeoutError):dequeue(reader,timeout=0)
        self.assertEqual(metadata[0][1],1)
        self.assertEqual(reader.current_idx,1) # Marker consumed even though call 0 has no payload.
        self.assertEqual(dequeue(reader,timeout=0),{'call':1})
        self.assertEqual(reads,[0]) # Blind retry read the NEXT message, not the missing payload.

    def test_actual_handle_and_enqueue_output_hold_ready_ack_behind_async_output(self):
        entered=threading.Event();release=threading.Event();wire=[]
        class AsyncOutput:
            def get_output(self):
                entered.set()
                if not release.wait(2):raise RuntimeError('test release missing')
                return 'decode-result'
        ns=dict(Any=object,AsyncModelRunnerOutput=AsyncOutput,
                WorkerProc=SimpleNamespace(ResponseStatus=SimpleNamespace(SUCCESS=0,FAILURE=1)),
                logger=SimpleNamespace(exception=lambda *_:None))
        enqueue=load(WORKER,'enqueue_output',ns);handle=load(WORKER,'handle_output',ns)
        worker=SimpleNamespace(use_async_scheduling=True,async_output_queue=Queue(),
                               worker_response_mq=SimpleNamespace(enqueue=wire.append))
        worker.enqueue_output=MethodType(enqueue,worker)
        handle(worker,AsyncOutput());handle(worker,'prefill-submitted')
        def loop():
            # Same FIFO get/enqueue pair as async_output_busy_loop, limited to
            # these two messages so the test owns and joins its thread.
            for _ in range(2):worker.enqueue_output(worker.async_output_queue.get())
        thread=threading.Thread(target=loop)
        thread.start()
        try:
            self.assertTrue(entered.wait(1))
            self.assertEqual(wire,[])
            self.assertEqual(worker.async_output_queue.qsize(),1)
        finally:release.set();thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(wire,[(0,'decode-result'),(0,'prefill-submitted')])


if __name__=='__main__':unittest.main()
