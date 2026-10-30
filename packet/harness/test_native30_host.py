"""Passive receipt tests: actual pool algorithm with CPU storage, no device operations."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import native30_host as host
import native30_localize as plugin

HERE=Path(__file__).parent
class Obj:pass
class View:
    def __init__(self,ptr,shape):self.ptr=ptr;self.shape=shape
    def data_ptr(self):return self.ptr
    def __getitem__(self,key):return self
class Buffer:
    counter=0
    def __init__(self,size,dtype):
        Buffer.counter+=1
        import numpy as np
        size=(size,) if isinstance(size,int) else size
        self.cpu=View(Buffer.counter,size);self.uva=View(Buffer.counter,size);self.np=np.zeros(size,dtype='int64')
class CpuTensor:pass

def fixture():
    # Execute the real pinned UvaBufferPool source; only its allocation leaf is replaced.
    root=HERE.parent/'reference/vllm-src/v1/worker/gpu/buffer_utils.py'
    if not root.exists():root=HERE/'e3_reference/v1/worker/gpu/buffer_utils.py'
    import ast
    text=root.read_text();node=next(n for n in ast.parse(text).body if isinstance(n,ast.ClassDef) and n.name=='UvaBufferPool')
    ns=dict(UvaBuffer=Buffer,_DEFAULT_MAX_CONCURRENCY=2,torch=NS(Tensor=CpuTensor,dtype=object),Sequence=list,__name__='vllm.v1.worker.gpu.buffer_utils')
    exec(compile('from __future__ import annotations\n'+ast.get_source_segment(text,node),str(root),'exec'),ns)
    Pool=ns['UvaBufferPool'];host.patch_method(Pool,'__init__',host.pool_init);host.patch_method(Pool,'copy_to_uva',host.pool_copy)
    p=Pool(1,'i');runner=Obj();states=Obj();states.__class__=type('RequestState',(Obj,),{'__module__':'vllm.v1.worker.gpu.states'})
    states.max_num_reqs=10;states.pool=p;states.req_id_to_index={}
    runner.req_states=states
    runner.block_tables=Obj();runner.block_tables.block_tables=[];runner.block_tables.num_blocks=NS(np=None)
    runner.kv_cache_config=NS(num_blocks=12)
    s=host.state(runner,1);s['active']=True;host.discover(runner,s)
    return p,runner,s

class Passive(unittest.TestCase):
    def setUp(self):
        self.rows=[];self.previous=host._EMIT;host._EMIT=lambda event,**kw:self.rows.append(dict(event=event,**kw))
    def tearDown(self):host._EMIT=self.previous

    def test_real_pool_rotation_growth_counts_without_reader_claim(self):
        p,r,s=fixture();obj=Obj();obj.runner=r;obj.events=[]
        fields=dict(rank=1,label='LAYER',call_id=1,op_seq=0)
        with host.boundary(obj,fields):
            p.copy_to_uva([1]);p.copy_to_uva([2]);p.copy_to_uva([3,4,5])
        snap=host.snapshot(r,s);row=snap['pools'][0]
        self.assertEqual((row['slot'],row['depth'],row['rotations']),(1,2,3))
        self.assertTrue(row['constructor_observed']);self.assertEqual(p._uva_bufs[1].cpu.shape,(4,))
        self.assertEqual(snap['reader_coverage'],'coverage-incomplete')
        self.assertEqual(len([x for x in self.rows if x['event']=='host-pool-rotation']),3)

    def test_pending_existing_event_query_uses_weak_reference_only(self):
        p,r,s=fixture();obj=Obj();obj.runner=r;obj.events=[]
        class Event:
            def __init__(self):self.queries=0
            def query(self):self.queries+=1;return False
        ev=Event()
        with host.boundary(obj,dict(rank=1,label='PREPARE',call_id=1,op_seq=0)):
            ctx=host._TLS.context;ctx['descriptor']=True;p.copy_to_uva([3])
        obj.events=[NS(op_seq=0,event=ev)];host.attach_event(obj,0)
        with host.boundary(obj,dict(rank=1,label='STATE_FINALIZE',call_id=2,op_seq=1)):
            p.copy_to_uva([4]);p.copy_to_uva([5])
        self.assertEqual(ev.queries,1)
        row=[x for x in self.rows if x['event']=='host-pool-rotation'][-1]
        self.assertTrue(row['previous_native_descriptor_pending']);self.assertEqual(row['reader_coverage'],'coverage-incomplete')

    def test_current_cpu_decode_mapping_and_violation_are_receipts(self):
        p,r,s=fixture()
        import numpy as np
        expected=(NS(idx_mapping_np=np.array([0,11])),None)
        calls=[]
        result=host.batch_after(lambda *a,**k:(calls.append(1) or expected),r,(),{})
        self.assertIs(result,expected);self.assertEqual(calls,[1]);self.assertTrue(s['last_idx']['bound_violation'])
        self.assertEqual((s['last_idx']['length'],s['last_idx']['min'],s['last_idx']['max']),(2,0,11))

    def test_expanded_block_ids_use_expanded_capacity(self):
        p,r,s=fixture();t=Obj();t.gpu=View(1,(10,20));t._staged_write_indices=[1];t._staged_write_starts=[0];t._staged_write_cu_lens=[2];t._staged_write_contents=[22,23]
        r.block_tables.block_tables=[t];r.block_tables.blocks_per_kv_block=[2];s['table_writers']={id(t):0}
        obj=Obj();obj.runner=r;obj.events=[]
        with host.boundary(obj,dict(rank=1,label='PREPARE',call_id=1,op_seq=0)):
            called=[];answer=object();self.assertIs(host.writer_apply(lambda *a,**k:(called.append(1) or answer),t,(),{}),answer)
        row=next(x for x in self.rows if x['event']=='host-staged-writes')['writes'][0]
        self.assertEqual(row['expanded_block_bound'],24);self.assertFalse(row['block_id_violation']);self.assertEqual(called,[1])
        t._staged_write_contents=[24];self.rows.clear()
        with host.boundary(obj,dict(rank=1,label='PREPARE',call_id=2,op_seq=1)):
            host.writer_apply(lambda *a,**k:None,t,(),{})
        self.assertTrue(next(x for x in self.rows if x['event']=='host-staged-writes')['writes'][0]['block_id_violation'])

    def test_diagnostic_errors_preserve_original_result_and_exception_once(self):
        p,r,s=fixture();obj=Obj();obj.runner=r;obj.events=[]
        with host.boundary(obj,dict(rank=1,label='LAYER',call_id=1,op_seq=0)):
            marker=object();calls=[]
            self.assertIs(host.writer_apply(lambda *a,**k:(calls.append(1) or marker),Obj(),(),{}),marker)
            error=RuntimeError('original')
            def fail(*a,**k):calls.append(2);raise error
            with self.assertRaises(RuntimeError) as caught:host.writer_apply(fail,Obj(),(),{})
            self.assertIs(caught.exception,error);self.assertEqual(calls,[1,2])

    def test_bounded_output_emits_one_limit_and_keeps_execution(self):
        p,r,s=fixture()
        with patch.object(host,'MAX_BYTES_PER_RANK',1000):
            for _ in range(5):host.emit(s,'fixture',value='x'*600)
        self.assertEqual(len([x for x in self.rows if x['event']=='host-receipt-limit']),1)
        p.copy_to_uva([1]);self.assertEqual(host.pool_state(p)['rotations'],1)

    def test_plan_digest_is_canonical_and_payload_is_private(self):
        a=NS(transaction=NS(engine_epoch=1,request_uid='PRIVATE_REQUEST',request_generation=2,txn_seq=3,chunk_start=0,chunk_end=2),prompt_end=3,input_ids=(7,8),ngram_context=(9,),block_tables=((1,2),),new_block_ids_to_zero=(),cow_pairs=(),state_epoch=0)
        one=host.plan(a);self.assertEqual(one,host.plan(a));self.assertNotIn('PRIVATE_REQUEST',json.dumps(one))
        a.block_tables=((1,3),);self.assertNotEqual(one['sha256'],host.plan(a)['sha256'])

    def test_no_device_tensor_read_is_accepted(self):
        class Device:
            def tolist(self):raise AssertionError('readback')
            def cpu(self):raise AssertionError('readback')
            def item(self):raise AssertionError('readback')
        self.assertFalse(host.ints(Device())['available'])
        self.assertEqual(plugin.mode(),'none')

    def test_placement_activation_and_native_index(self):
        p,r,s=fixture();s['active']=False
        answer=object();self.assertIs(host.qualification(lambda *a:answer,r,dict(action='teacher_configure',enabled=True)),answer)
        self.assertTrue(s['active']);self.assertTrue(any(x['event']=='host-placement-baseline' for x in self.rows))
        obj=Obj();obj.runner=r;obj.events=[]
        with host.boundary(obj,dict(rank=1,label='PREPARE',call_id=1,op_seq=0)):
            def add(states,*a,**kw):states.req_id_to_index['private']=4
            host.request_added(add,r.req_states,(),dict(req_id='private'))
        self.assertEqual(s['native_index'],4)
        host.qualification(lambda *a:answer,r,dict(action='teacher_configure',enabled=False));self.assertFalse(s['active'])

if __name__=='__main__':unittest.main()
