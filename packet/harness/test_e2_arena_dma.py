"""CPU buffers, delayed events and pinned-source arithmetic; no GPU tests."""
import ast
from dataclasses import replace
from pathlib import Path
import sys
import unittest
from contextlib import contextmanager
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'patches/s5_engine'))
from arena_dma import ArenaDMA, Fence, TorchArenaBackend
from serial_dma import TensorFacts
from retention_contract import ContractError, Pending
import arena_attention
from resident_attention import function, NAME
from test_e2_mirror import setup, output, LAYERS
from test_sparse_stage import old_plan, staged


class Buffer:
    def __init__(self, shape, device, address, data=None):
        self.facts=TensorFacts(device,shape,(shape[2]*512,shape[2]*512,512,1),
                              'torch.bfloat16',2,device=='cpu',True,address)
        self.data=bytearray(self.facts.nbytes) if data is None else data
    def first(self,n):
        return Buffer((n,*self.facts.shape[1:]),self.facts.device,self.facts.address,
                      memoryview(self.data)[:n*self.facts.shape[2]*1024])


class Backend:
    device='cuda:0'
    def __init__(self):
        self.stream=10;self.events=[];self.exec_actions=[];self.fail_submit=False
        self.fail_wait=None;self.on_wait=lambda _:None
    def facts(self,x):return x.facts
    def execution_stream(self):return self.stream
    def execution_fence(self):
        self.events.append(('execution-fence',self.stream))
        actions=self.exec_actions;self.exec_actions=[]
        return Fence(self.device,self.stream,actions)
    def copy_stream_token(self):return 20
    def submit(self,host,arena,pieces):
        self.events.append(('submit',[(p.source_page,p.arena_page,p.pages) for p in pieces]))
        if self.fail_submit:raise RuntimeError('submit failed')
        size=host.facts.shape[2]*1024
        def do():
            for p in pieces:
                arena.data[p.arena_page*size:(p.arena_page+p.pages)*size]=host.data[p.source_page*size:(p.source_page+p.pages)*size]
        return Fence(self.device,20,[do])
    def wait(self,fence,stream):
        self.events.append(('wait',stream));self.on_wait(stream)
        if self.fail_wait==stream:raise RuntimeError('wait failed')
        if fence.device!=self.device or fence.stream!=stream:raise ContractError('foreign event')
        for action in fence.event:action()


def fixture(ids=None,tokens=None):
    ids=list(range(1,51)) if ids is None else ids
    tokens=4*len(ids) if tokens is None else tokens
    j,m,t=setup();m.arena.arena_pages=23
    j.allocated(ids);t.append(0,(ids,),True)
    f=j.frame(output(new=[('a',(ids,),0)],scheduled={'a':tokens}),lambda _:(ids,))
    m.enter(f,{'a':0},t)
    b=Backend();hosts={}
    for index,name in enumerate(LAYERS):
        host=Buffer((100,1,4,512),'cpu',1_000_000*(index+1))
        for page in range(100):host.data[page*4096:(page+1)*4096]=bytes([(index*11+page)%256])*4096
        hosts[name]=host
    arena=Buffer((23,1,4,512),b.device,8_000_000)
    table=type('Table',(),{})()
    table.facts=TensorFacts(b.device,(100,),(1,),'torch.int32',4,False,True,9_000_000)
    return m,b,hosts,arena,table,ArenaDMA(m,hosts,b),tokens


class DMATests(unittest.TestCase):
    def test_direct_three_buckets_byte_equal_fragmented_and_six_independent_histories(self):
        for ids in (list(range(1,51)),list(range(50,0,-1)),list(range(1,51,2))+list(range(2,51,2))):
            m,b,hosts,arena,table,a,tokens=fixture(ids)
            for name in LAYERS:
                with a.session('a',name,arena,table,tokens) as s:
                    s.route('dense_dma',50)
                    for lo,hi in ((0,23),(23,46),(46,50)):
                        s.fill(arena.first(hi-lo),lo,hi,table)
                        expected=b''.join(hosts[name].data[p*4096:(p+1)*4096] for p in ids[lo:hi])
                        self.assertEqual(bytes(arena.data[:len(expected)]),expected)
                        s.consume_begin();s.consume_end()
                self.assertEqual(m.pool.held_count(),0)
            self.assertEqual(a.stats['bytes'],6*50*4096)
            self.assertEqual(a.stats['buckets'],18)
            self.assertTrue(all(n<=4 for kind,items in b.events if kind=='submit' for _,_,n in items))
            self.assertIs(a.arena,arena)

    def test_producer_then_dma_then_consumer_order_and_no_early_reuse(self):
        m,b,hosts,arena,table,a,n=fixture([1,2],6)
        b.exec_actions.append(lambda:hosts['qsa0'].data.__setitem__(slice(4096,8192),bytes([77])*4096))
        def inspect(stream):
            if stream==20:
                self.assertEqual(m.arena.state,'filling')
                with self.assertRaises(Pending):m.leave()
        b.on_wait=inspect
        with a.session('a','qsa0',arena,table,n) as s:
            s.route('dense_dma',2);s.fill(arena.first(2),0,2,table)
            self.assertEqual(bytes(arena.data[:4096]),bytes([77])*4096)
            with self.assertRaises(Pending):m.arena.begin_fill()
            s.consume_begin()
            b.exec_actions.append(lambda:None)
            s.consume_end()
        self.assertEqual([x[0] for x in b.events],['execution-fence','wait','submit','wait','execution-fence','wait'])

    def test_sparse_and_empty_use_no_dma_and_no_extra_cuda_fence(self):
        for route,pages in (('sparse_uva',1),('empty',0)):
            m,b,hosts,arena,table,a,n=fixture([1,2],6)
            with a.session('a','qsa0',arena,table,n) as s:s.route(route,pages)
            self.assertEqual(a.stats[route],1);self.assertEqual(a.stats['bytes'],0)
            self.assertFalse(b.events);self.assertIsNone(a.current)
            self.assertEqual(m.pool.held_count(),0)

    def test_submit_or_wait_failure_keeps_bindings_and_lease(self):
        for failure in ('submit','copy','consumer'):
            m,b,hosts,arena,table,a,n=fixture([1])
            with self.assertRaises(RuntimeError):
                with a.session('a','qsa0',arena,table,n) as s:
                    s.route('dense_dma',1)
                    if failure=='submit':b.fail_submit=True
                    if failure=='copy':b.fail_wait=20
                    s.fill(arena.first(1),0,1,table);s.consume_begin()
                    if failure=='consumer':b.fail_wait=10
                    s.consume_end()
            self.assertTrue(m.failed);self.assertIsNotNone(a.current)
            self.assertEqual(m.pool.held_count(),1)
            self.assertIs(a.hosts['qsa0'],hosts['qsa0'])
            with self.assertRaises(Pending):m.leave()

    def test_producer_fence_failure_never_acquires_cpu_read_lease(self):
        m,b,hosts,arena,table,a,n=fixture([1]);b.fail_wait=10
        with self.assertRaises(RuntimeError):
            with a.session('a','qsa0',arena,table,n) as s:s.route('dense_dma',1)
        self.assertTrue(m.failed);self.assertEqual(m.pool.held_count(),0)
        self.assertFalse(any(x[0]=='submit' for x in b.events))

    def test_layout_uva_unpinned_alias_and_wrong_capacity_rejected(self):
        for mutation in (dict(device='cuda:0'),dict(pinned=False),dict(strides=(1,1,1,1)),dict(dtype='torch.float16')):
            m,b,hosts,arena,table,a,n=fixture([1]);hosts['qsa0'].facts=replace(hosts['qsa0'].facts,**mutation)
            with self.assertRaises(ContractError):ArenaDMA(m,hosts,b)
        m,b,hosts,arena,table,a,n=fixture([1]);hosts['qsa1']=hosts['qsa0']
        with self.assertRaises(ContractError):ArenaDMA(m,hosts,b)
        m,b,hosts,arena,table,a,n=fixture([1])
        with self.assertRaises(ContractError):
            with a.session('a','qsa0',arena.first(22),table,n):pass

    def test_wrong_context_bucket_order_and_table_fail_before_submission(self):
        for bad in ('stream','table','bucket','span'):
            m,b,hosts,arena,table,a,n=fixture([1,2])
            with self.assertRaises(ContractError):
                with a.session('a','qsa0',arena,table,n) as s:
                    s.route('dense_dma',2)
                    if bad=='stream':b.stream=11
                    if bad=='table':table.facts=replace(table.facts,address=table.facts.address+4)
                    if bad=='bucket':s.fill(arena.first(1),1,2,table)
                    elif bad=='span':s.fill(arena.first(1),0,2,table)
                    else:s.fill(arena.first(2),0,2,table)
            self.assertFalse(any(x[0]=='submit' for x in b.events))

    def test_sparse_does_not_allow_late_dense_fill_or_double_route(self):
        for action in ('fill','route'):
            m,b,hosts,arena,table,a,n=fixture([1])
            with self.assertRaises(ContractError):
                with a.session('a','qsa0',arena,table,n) as s:
                    s.route('sparse_uva',1)
                    if action=='fill':s.fill(arena.first(1),0,1,table)
                    else:s.route('dense_dma',1)

    def test_no_route_or_unconsumed_bucket_cannot_be_successful(self):
        for state in ('none','owned','ready','consuming'):
            m,b,hosts,arena,table,a,n=fixture([1])
            with self.assertRaises((ContractError,Pending)):
                with a.session('a','qsa0',arena,table,n) as s:
                    if state!='none':s.route('dense_dma',1)
                    if state in ('ready','consuming'):s.fill(arena.first(1),0,1,table)
                    if state=='consuming':s.consume_begin()
            self.assertTrue(m.failed)


class AttentionTests(unittest.TestCase):
    def source(self):return (ROOT/'patches/retain/ops_qsa.py.v8.1').read_text()

    def test_original_unchanged_and_candidate_diff_exactly_callbacks_and_guards(self):
        src=self.source();result=arena_attention.generate(src)
        self.assertTrue(result.startswith(src+'\n\n'))
        original=function(src,NAME);candidate=function(result,'flash_e2_arena_attention')
        self.assertEqual(original,function(result,NAME))
        # Reverse the complete generated delta: every sparse/remap expression,
        # kernel argument, accumulator update and FIRST/LAST order must match.
        c=candidate.replace('flash_e2_arena_attention',NAME,1).replace('    *, e2,\n','',1)
        c=c.replace('    if retained is not None or retained_copied != 0:\n'
                    '        raise RuntimeError("E2 arena rejects retained-bucket shortcut")\n','',1)
        c=c.replace('        e2.route("empty", 0)\n','',1)
        c=c.replace('            e2.route("empty", 0)\n','',1)
        c=c.replace('    e2.route("sparse_uva" if unique_tokens is not None else "dense_dma", num_pages)\n','',1)
        c=c.replace('            e2.fill(staged, page_lo, page_hi, block_table_row)\n',
                    '            physical = block_table_row[page_lo:page_hi].to(torch.int64)\n'
                    '            torch.index_select(kv_cache, 0, physical, out=staged)\n',1)
        c=c.replace('        if unique_tokens is None:\n            e2.consume_begin()\n','',1)
        c=c.replace('        if unique_tokens is None:\n            e2.consume_end()\n','',1)
        self.assertEqual(c,original)
        with self.assertRaises(ValueError):arena_attention.generate(src+'\n')
        ast.parse(result)

    def test_dense_sparse_empty_buckets_preserve_byte_arithmetic_model(self):
        rng=np.random.default_rng(33);page,arena,pages=8,3,7;n=page*pages
        physical=np.array([11,3,7,9,1,5,10]);host=rng.normal(size=(12,page,8)).astype('f')
        q=rng.normal(size=(3,8)).astype('f');rows=np.array([2,0])
        for count in (0,1,23,24,25,44,45,56):
            ids=np.full((3,65),-1,np.int32)
            for r in rows:ids[r,:count]=rng.choice(n,count,replace=False);ids[r,-1]=count
            plan=old_plan(ids,rows,n,page,arena)
            # CPU DMA/gather source identity at every original dense bucket.
            copied=np.empty_like(host[physical])
            for lo in range(0,pages,arena):copied[lo:lo+arena]=host[physical[lo:lo+arena]]
            expected=staged(plan,ids,rows,host[physical].reshape(n,8),q,page)
            actual=staged(plan,ids,rows,copied.reshape(n,8),q,page)
            self.assertEqual(actual.tobytes(),expected.tobytes())


class NativeBackendTests(unittest.TestCase):
    def fake_torch(self):
        streams={};current=[10];device=[0];capture=[False];operations=[];queues={10:[],20:[]}
        class Device:
            def __init__(self,name):self.name=str(name);self.index=int(self.name.split(':')[1])
            def __str__(self):return self.name
        @contextmanager
        def use_device(name):
            old=device[0];device[0]=Device(name).index
            try:yield
            finally:device[0]=old
        class Stream:
            def __init__(self,device=None):self.cuda_stream=20;streams[20]=self
        streams[10]=NS(cuda_stream=10)
        @contextmanager
        def use_stream(stream):
            old=current[0];current[0]=stream.cuda_stream
            try:yield
            finally:current[0]=old
        class Event:
            def __init__(self):self.device=None;self.stream=None;self.target=0
            def record(self,stream):
                self.device=Device('cuda:'+str(device[0]));self.stream=stream.cuda_stream
                self.target=len(queues[self.stream]);operations.append(('record',self.stream))
            def synchronize(self):
                operations.append(('fence',self.stream))
                for i in range(self.target):
                    action=queues[self.stream][i]
                    if action is not None:action();queues[self.stream][i]=None
        class Tensor:
            def __init__(self,data,kind):self.data=data;self.kind=kind
            def __getitem__(self,key):return Tensor(self.data[key],self.kind)
            def copy_(self,src,non_blocking):
                operations.append(('copy',current[0],src.kind,self.kind,non_blocking,len(src.data)))
                queues[current[0]].append(lambda:self.data.__setitem__(slice(None),src.data))
        fake=NS(device=Device,cuda=NS(Stream=Stream,Event=Event,device=use_device,stream=use_stream,
                current_stream=lambda _:streams[current[0]],current_device=lambda:device[0],
                is_current_stream_capturing=lambda:capture[0]))
        return fake,Tensor,operations,queues,current,device,capture

    def test_real_backend_batches_all_pieces_on_copy_stream_then_one_terminal_fence(self):
        fake,T,ops,queues,current,device,capture=self.fake_torch()
        with patch.dict(sys.modules,{'torch':fake}):b=TorchArenaBackend('cuda:0')
        host=T(np.arange(20*4,dtype=np.uint16).reshape(20,4),'cpu')
        dst=T(np.zeros((5,4),dtype=np.uint16),'cuda:0')
        from arena_contract import Piece
        before=dst.data.copy();f=b.submit(host,dst,(Piece(5,0,4),Piece(12,4,1)))
        np.testing.assert_array_equal(dst.data,before)
        b.wait(f,b.copy_stream_token())
        np.testing.assert_array_equal(dst.data,np.concatenate((host.data[5:9],host.data[12:13])))
        self.assertEqual(ops,[('copy',20,'cpu','cuda:0',True,4),('copy',20,'cpu','cuda:0',True,1),
                              ('record',20),('fence',20)])
        self.assertEqual(current[0],10)

    def test_real_backend_execution_fence_rejects_copy_fence_capture_and_device_switch(self):
        fake,T,ops,queues,current,device,capture=self.fake_torch()
        with patch.dict(sys.modules,{'torch':fake}):b=TorchArenaBackend('cuda:0')
        f=b.execution_fence();self.assertEqual(f.stream,10)
        with self.assertRaises(ContractError):b.wait(f,20)
        b.wait(f,10)
        capture[0]=True
        with self.assertRaises(ContractError):b.execution_fence()
        capture[0]=False;device[0]=1
        with self.assertRaises(ContractError):b.execution_fence()


if __name__=='__main__':unittest.main()
