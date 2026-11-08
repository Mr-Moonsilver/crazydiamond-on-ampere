"""Real indexer host functions with CPU score/top-k kernel ports; no CUDA proof."""
import ast
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
import weakref

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'patches/s5_engine'))
import make_s5_overlap as build
import make_e2_arena as e2
from native_measurement import INDEXER_PHASE_CONTEXT

RECEIPTS=[]


def source():
    return (ROOT/'reference/vllm-src'/build.INDEXER_PATH).read_text()


def nodes(text):
    return ast.parse(text).body


class Transform(unittest.TestCase):
    def test_the_only_nonprobe_change_is_delete_after_the_last_topk_consumer(self):
        old=source();new=build.compose_indexer(old)
        tree=ast.parse(new)
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='qsa_select_paged_prefill')
        loop=next(n for n in fn.body if isinstance(n,ast.For))
        self.assertIsInstance(loop.body[-1],ast.Delete)
        self.assertEqual(ast.unparse(loop.body[-1]),'del logits')
        loop.body.pop()
        # Strip only the observational context lookup/guards and its import.
        tree.body = [n for n in tree.body if not (isinstance(n, ast.ImportFrom) and n.module == 'vllm.v1.flash_e3.native_measurement')]
        fn.body = [n for n in fn.body if not (isinstance(n, ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='_phase')]
        loop.body = [n for n in loop.body if not (isinstance(n,ast.If) and ast.unparse(n.test)=='_phase is not None')]
        self.assertEqual(ast.unparse(loop.body[-1].value.func),'_topk')
        self.assertEqual(ast.dump(tree,include_attributes=False),ast.dump(ast.parse(old),include_attributes=False))
        self.assertEqual(new.count('del logits'),1)
        with self.assertRaisesRegex(ValueError,'unreviewed'):
            build.compose_indexer(new)
        with self.assertRaisesRegex(ValueError,'unreviewed'):
            build.compose_indexer(old+'# drift\n')

    def test_launcher_checks_and_mounts_the_generated_ops_file_only_under_s5(self):
        launcher=build._s5_launcher(e2.launcher((ROOT/'deploy/run-flash-prod.sh').read_text()))
        self.assertIn('qsa.py ops_qsa.py ops_qsa_indexer.py; do',launcher)
        self.assertIn('$OVERLAY/ops_qsa_indexer.py:/usr/local/lib/python3.12/dist-packages/vllm/'+build.INDEXER_PATH+':ro',launcher)
        self.assertLess(launcher.index('[[ ${S5:-0} == 1 ]]'),launcher.index('$OVERLAY/ops_qsa_indexer.py:'))
        self.assertEqual(build.REFERENCE_INPUTS[build.INDEXER_PATH],hashlib.sha256(source().encode()).hexdigest())


class Ledger:
    def __init__(self):self.live={};self.peak_bytes=0;self.peak_slices=0;self.events=[]
    def allocated(self,tensor):
        token=id(tensor);size=tensor.numel()*tensor.element_size();self.live[token]=size
        self.peak_bytes=max(self.peak_bytes,sum(self.live.values()));self.peak_slices=max(self.peak_slices,len(self.live))
        self.events.append(('allocate',size,len(self.live)))
        weakref.finalize(tensor,self.retire,token)
    def retire(self,token):self.events.append(('retire',self.live.pop(token),len(self.live)))


class Ports:
    """Real torch CPU allocations; replace just the CUDA score/top-k launches."""
    def __init__(self,torch):
        self.torch=torch;self.ledger=Ledger();self.calls=[]
        self.ops=NS(_C=NS(persistent_topk=self.topk,cooperative_topk=self.topk))
    def __getattr__(self,name):return getattr(self.torch,name)
    def empty(self,*args,**kwargs):
        t=self.torch.empty(*args,**kwargs)
        if t.dtype==self.torch.float32:self.ledger.allocated(t)
        return t
    def __getitem__(self,grid):
        def kernel(q,cache,table,starts,visible,logits,*args,**kw):
            # The pinned launch's six strides, then num_queries/query_offset.
            count,offset=args[6:8]
            self.calls.append(('score',tuple(grid),count,offset,tuple(logits.shape),tuple(sorted(kw.items()))))
            keys=cache[table[0].long()].reshape(-1,cache.shape[-1])[:logits.shape[1]]
            self.torch.mm(q[offset:offset+count,0].float(),keys.float().T,out=logits)
        return kernel
    def topk(self,logits,visible,out,workspace,k,width):
        digest=hashlib.sha256(logits.numpy().tobytes()).hexdigest()
        self.calls.append(('topk',tuple(logits.shape),width,k,digest))
        columns=self.torch.arange(width)
        logits.masked_fill_(columns[None,:]>=visible[:,None],-float('inf'))
        out.copy_(self.torch.topk(logits,k,dim=1,sorted=True).indices)


def function(text,ports):
    keep={'_prefill_logits','_topk','qsa_select_paged_prefill'}
    module=ast.Module(body=[n for n in nodes(text) if isinstance(n,ast.FunctionDef) and n.name in keep],type_ignores=[])
    ns=dict(torch=ports,triton=NS(cdiv=lambda a,b:-(-a//b)),envs=NS(VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=64),
            _TOPK_WORKSPACE_BYTES=1024*1024,_qsa_mqa_paged_prefill_kernel=ports,
            INDEXER_PHASE_CONTEXT=INDEXER_PHASE_CONTEXT,
            current_platform=NS(has_device_capability=lambda n:False,is_device_capability_family=lambda n:False))
    exec(compile(module,'actual-pinned-qsa-indexer-host-functions','exec'),ns)
    return ns['qsa_select_paged_prefill']


class CpuLifetime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch=torch
        torch.set_num_threads(4)
        assert not torch.cuda.is_initialized()
        RECEIPTS.clear()

    def run_case(self,text,rows,history):
        t=self.torch;p=Ports(t);f=function(text,p);pages=(history+4047)//4048+1
        q=(t.arange(rows*4).reshape(rows,1,4).remainder(67).float()/32-1).to(t.bfloat16)
        cache=(t.arange(pages*1012*4).reshape(pages,1012,1,4).remainder(103).float()/64-1).to(t.bfloat16)
        table=t.arange(pages,dtype=t.int32)[None,:]
        starts=t.tensor([0,rows],dtype=t.int32)
        visible=(t.arange(history-rows+1,history+1,dtype=t.int32)+3)//4
        out=t.empty((rows,512),dtype=t.int32)
        with t.inference_mode():f(q,cache,table,starts,visible,2048,4,rows,out,history)
        self.assertEqual(p.ledger.live,{})
        self.assertFalse(t.cuda.is_initialized())
        return out,p

    def test_single_live_slice_and_bit_equal_outputs_across_histories_and_quanta(self):
        old=source();new=build.compose_indexer(old)
        for history in (97151,97152,97153,133679,201261):
            for rows in (1024,1352):
                with self.subTest(history=history,rows=rows):
                    expected,before=self.run_case(old,rows,history)
                    got,after=self.run_case(new,rows,history)
                    self.assertTrue(self.torch.equal(expected,got))
                    self.assertEqual(before.calls,after.calls,'values and launch order/geometry must match')
                    self.assertEqual(before.ledger.peak_slices,2)
                    self.assertEqual(after.ledger.peak_slices,1)
                    self.assertLess(after.ledger.peak_bytes,before.ledger.peak_bytes)
                    RECEIPTS.append(dict(history=history,rows=rows,peak_before_bytes=before.ledger.peak_bytes,
                        peak_after_bytes=after.ledger.peak_bytes,saved_MiB=(before.ledger.peak_bytes-after.ledger.peak_bytes)/2**20,
                        output_sha256=hashlib.sha256(got.numpy().tobytes()).hexdigest(),
                        calls_equal=True,outputs_bit_equal=True,live_after_return=0,scope='CPU score tensors; CUDA kernels are ports'))

    def test_zero_and_single_tile_still_work(self):
        for rows in (0,32):
            old,a=self.run_case(source(),rows,201261);new,b=self.run_case(build.compose_indexer(source()),rows,201261)
            self.assertTrue(self.torch.equal(old,new));self.assertEqual(a.calls,b.calls)
            self.assertEqual(a.ledger.peak_bytes,b.ledger.peak_bytes)
            self.assertLessEqual(b.ledger.peak_slices,1)


if __name__=='__main__':unittest.main()
