import ctypes
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'patches/s5_engine'))
import route_invariant as ri

class Sources:
    total_bytes=4
    value=b'abcd'
    def chunks(self):yield self.value
    def metadata(self):return dict(local_layers=6,cohort_size=8,unique_layer_pages=1,history_bytes=4)
class SnapshotTests(unittest.TestCase):
    def make(self):
        self.sources=Sources();self.graph=object();self.value=b'captured arguments'
        self.probe=types.SimpleNamespace(storage_epoch=1,route_generation=4,
            bank=types.SimpleNamespace(manager=types.SimpleNamespace(graphs={1:self.graph})))
        return ri.Snapshot(self.probe,self.sources,lambda g:self.value)
    def test_private_exact_bytes_through_three_boundaries(self):
        s=self.make()
        for phase in ('entered','measured','returned'):
            self.assertEqual(s.check(phase)['verdict'],'PASS')
        self.assertEqual(s.history,[])
        self.assertNotIn('abcd',str(s.checks))
    def test_one_bit_corruption_fails_and_stays_failed(self):
        for bank in (False,True):
            s=self.make();s.check('entered')
            if bank:self.value=b'captured argumenus'
            else:self.sources.value=b'abce'
            self.assertEqual(s.check('measured')['verdict'],'FAIL')
            self.sources.value=b'abcd';self.value=b'captured arguments'
            self.assertEqual(s.check('returned')['verdict'],'FAIL')
    def test_bank_replacement_and_phase_skip_refused(self):
        s=self.make()
        with self.assertRaises(ri.InvariantUnavailable):s.check('returned')
        self.probe.bank.manager.graphs[1]=object()
        self.assertEqual(s.check('entered')['verdict'],'FAIL')
    def test_limit_precedes_history_copy(self):
        self.make();self.sources.total_bytes=ri.MAX_SNAPSHOT_BYTES
        self.sources.chunks=lambda:(_ for _ in ()).throw(AssertionError('must not allocate'))
        with self.assertRaisesRegex(ri.InvariantUnavailable,'budget'):
            ri.Snapshot(self.probe,self.sources,lambda g:b'x')

class PrefixSelectionTests(unittest.TestCase):
    def test_real_prefix_selector_deduplicates_and_excludes_written_tail(self):
        import numpy as np
        from unittest.mock import patch
        class Tensor:
            def __init__(self,a):self.a=a;self.device=types.SimpleNamespace(type='cpu');self.dtype=a.dtype;self.shape=a.shape
            def is_contiguous(self):return self.a.flags.c_contiguous
            def data_ptr(self):return self.a.__array_interface__['data'][0]
            def __getitem__(self,k):return Tensor(self.a[k])
            def view(self,t):return Tensor(self.a.view(t))
            def reshape(self,*shape):return Tensor(self.a.reshape(*shape))
            def numel(self):return self.a.size
            def numpy(self):return self.a
            def cpu(self):return self
            def tolist(self):return self.a.tolist()
        layers={f'layer{i}':types.SimpleNamespace(_qsa_host_kv=Tensor(np.zeros((3,1,4048,512),dtype=np.uint16))) for i in range(6)}
        table=Tensor(np.ones((8,1),dtype=np.int32))
        runner=types.SimpleNamespace(req_states=types.SimpleNamespace(req_id_to_index={str(i):i for i in range(8)},prompt_len=types.SimpleNamespace(np=np.full(8,3))),
            compilation_config=types.SimpleNamespace(static_forward_context=layers),
            kv_cache_config=types.SimpleNamespace(kv_cache_groups=[types.SimpleNamespace(layer_names=list(layers))]),
            block_tables=types.SimpleNamespace(block_sizes=[4048],blocks_per_kv_block=[1],num_blocks=types.SimpleNamespace(np=np.ones((1,8))),block_tables=[types.SimpleNamespace(gpu=table)]))
        with patch.dict(sys.modules,torch=types.SimpleNamespace(bfloat16=np.dtype('uint16'),uint8=np.uint8)):
            source=ri.PrefixSources(runner)
        self.assertEqual(source.total_bytes,6*3*1024)
        before=list(source.chunks())
        layers['layer0']._qsa_host_kv.a[1,0,3,0]=7
        self.assertEqual(before,list(source.chunks()))
        layers['layer0']._qsa_host_kv.a[1,0,2,0]=1
        self.assertNotEqual(before,list(source.chunks()))
        table.a[0,0]=2
        with self.assertRaisesRegex(ri.InvariantUnavailable,'mapping'):list(source.chunks())

STUB=r'''
#include <cuda.h>
#include <cstring>
static unsigned value=123;static int kind=0;static unsigned char port=0;
extern "C" void change(unsigned v,int k,unsigned char p){value=v;kind=k;port=p;}
extern "C" CUresult CUDAAPI cuGraphGetNodes(CUgraph,CUgraphNode*n,size_t*c){if(n)n[0]=(CUgraphNode)0x20;*c=1;return CUDA_SUCCESS;}
extern "C" CUresult CUDAAPI cuGraphNodeGetType(CUgraphNode,CUgraphNodeType*t){*t=kind?CU_GRAPH_NODE_TYPE_HOST:CU_GRAPH_NODE_TYPE_KERNEL;return CUDA_SUCCESS;}
extern "C" CUresult CUDAAPI cuGraphNodeGetDependencies(CUgraphNode,CUgraphNode*n,CUgraphEdgeData*e,size_t*c){if(n){n[0]=(CUgraphNode)0x10;e[0]={};e[0].from_port=port;}*c=1;return CUDA_SUCCESS;}
extern "C" CUresult CUDAAPI cuGraphKernelNodeGetParams(CUgraphNode,CUDA_KERNEL_NODE_PARAMS*p){*p={};p->func=(CUfunction)0x30;p->gridDimX=4;p->blockDimX=32;static void*args[]={&value};p->kernelParams=args;return CUDA_SUCCESS;}
extern "C" CUresult CUDAAPI cuFuncGetParamInfo(CUfunction,size_t i,size_t*o,size_t*s){if(i)return CUDA_ERROR_INVALID_VALUE;*o=0;*s=4;return CUDA_SUCCESS;}
extern "C" CUresult CUDAAPI cuKernelGetParamInfo(CUkernel,size_t,size_t*,size_t*){return CUDA_ERROR_INVALID_HANDLE;}
extern "C" CUresult CUDAAPI cuGraphMemcpyNodeGetParams(CUgraphNode,CUDA_MEMCPY3D*){return CUDA_ERROR_NOT_SUPPORTED;}
extern "C" CUresult CUDAAPI cuGraphMemsetNodeGetParams(CUgraphNode,CUDA_MEMSET_NODE_PARAMS*){return CUDA_ERROR_NOT_SUPPORTED;}
extern "C" CUresult CUDAAPI cuGraphChildGraphNodeGetGraph(CUgraphNode,CUgraph*){return CUDA_ERROR_NOT_SUPPORTED;}
extern "C" CUresult CUDAAPI cuGraphEventWaitNodeGetEvent(CUgraphNode,CUevent*){return CUDA_ERROR_NOT_SUPPORTED;}
extern "C" CUresult CUDAAPI cuGraphEventRecordNodeGetEvent(CUgraphNode,CUevent*){return CUDA_ERROR_NOT_SUPPORTED;}
'''
class NativeBytesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory(prefix='s5-bank-bytes-')
        p=Path(cls.tmp.name);(p/'stub.cpp').write_text(STUB)
        subprocess.run(['g++','-std=c++17','-shared','-fPIC','-I',str(ROOT/'reference/cuda-13'),
            str(ROOT/'patches/s5_engine/native/flash_s5_graph_bytes.cpp'),str(p/'stub.cpp'),'-o',str(p/'bytes.so')],check=True,capture_output=True)
        cls.lib=ctypes.CDLL(str(p/'bytes.so'));cls.lib.change.argtypes=[ctypes.c_uint,ctypes.c_int,ctypes.c_ubyte]
    def test_real_export_preserves_scalar_bytes_and_edges_and_refuses_unknown(self):
        read=ri.GraphBytes(self.lib);g=types.SimpleNamespace(raw_cuda_graph=lambda:0x50)
        self.lib.change(123,0,0);before=read(g);self.assertEqual(before,read(g))
        self.lib.change(124,0,0);self.assertNotEqual(before,read(g))
        self.lib.change(123,0,1);self.assertNotEqual(before,read(g))
        self.lib.change(123,1,0)
        with self.assertRaisesRegex(ri.InvariantUnavailable,'definition'):read(g)
if __name__=='__main__':unittest.main()
