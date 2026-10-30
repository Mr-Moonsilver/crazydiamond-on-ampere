"""Execute the pinned real pre-indexer host body up to its grid-launch leaf."""
import ast
import ctypes
import hashlib
from pathlib import Path
import types
import unittest

from native34_indexer import operands,direct_bounds,ORIGINS

PIN='8af0b1979e5896569d3a4c11b0708013f1f93bb4c5cb1d98dafdd7fb52c64b70'
REL='models/qwen4_exp/nvidia/ops/qsa_pre_indexer.py'


class Storage:
    def __init__(self,n):self.memory=ctypes.create_string_buffer(n);self._cdata=id(self)
    def data_ptr(self):return ctypes.addressof(self.memory)
    def nbytes(self):return len(self.memory)


class Tensor:
    device='cuda:0'
    def __init__(self,shape,item=2,dtype='torch.bfloat16'):
        self.shape=tuple(shape);self.ndim=len(shape);self.bytes_per_item=item;self.dtype=dtype
        steps=[];n=1
        for size in reversed(shape):steps.insert(0,n);n*=size
        self.steps=tuple(steps);self.storage=Storage(n*item)
    def stride(self,n=None):return self.steps if n is None else self.steps[n]
    def is_contiguous(self):return True
    def untyped_storage(self):return self.storage
    def data_ptr(self):return self.storage.data_ptr()
    def element_size(self):return self.bytes_per_item
    def storage_offset(self):return 0
    def item(self):raise AssertionError('device-value-readback-forbidden')
    def cpu(self):raise AssertionError('device-value-readback-forbidden')


def host_body():
    root=Path(__file__).resolve().parents[1]
    path=root/'reference/vllm-src'/REL
    if not path.exists():path=Path('/usr/local/lib/python3.12/dist-packages/vllm')/REL
    source=path.read_bytes();assert hashlib.sha256(source).hexdigest()==PIN
    tree=ast.parse(source);entry=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='qsa_pre_indexer')
    kernel=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_qsa_pre_indexer_kernel')
    names=[n.arg for n in kernel.args.args]
    launches=[]
    class Grid:
        def __getitem__(self,grid):
            def run(*args,**kwargs):
                bound=dict(zip(names,args));bound.update({k:v for k,v in kwargs.items() if k in names})
                launches.append(dict(grid=grid,bound=bound))
            return run
    # Annotations alone are removed; every host assertion, shape/stride access,
    # branch and launch argument in the actual function body executes unchanged.
    entry.returns=None
    for n in entry.args.args+entry.args.kwonlyargs:n.annotation=None
    module=ast.fix_missing_locations(ast.Module(body=[entry],type_ignores=[]))
    ns=dict(triton=types.SimpleNamespace(cdiv=lambda a,b:(a+b-1)//b),_qsa_pre_indexer_kernel=Grid())
    exec(compile(module,str(path),'exec'),ns)
    return ns['qsa_pre_indexer'],launches


def inputs(mrope=False):
    n,d,h=8,8,2
    return [Tensor((n,h*d)),Tensor((n,d)),Tensor((3,n) if mrope else (n,),8,'torch.int64'),
        Tensor((32,d//2)),Tensor((d,),4,'torch.float32'),Tensor((d,),4,'torch.float32'),1e-6,
        Tensor((n,h,d)),Tensor((4,4,d+12)),Tensor((n,),8,'torch.int64'),
        Tensor((1,1),4,'torch.int32'),Tensor((2,),4,'torch.int32'),Tensor((n,),8,'torch.int64'),
        Tensor((4,4,d)),Tensor((n,),8,'torch.int64'),Tensor((2,2),4,'torch.int32')]


class IndexerLaunch(unittest.TestCase):
    def launch(self,values,mrope=False):
        fn,calls=host_body();self.assertIsNone(fn(*values,compress_ratio=4,
            mrope_section=(1,1,1) if mrope else None,rope_pos_offset=8))
        self.assertEqual(len(calls),1)
        bound=calls[0]['bound'];rows=operands(bound);checks=direct_bounds(bound,rows)
        self.assertEqual(set(rows),set(ORIGINS));self.assertEqual(len(rows),15)
        self.assertEqual(set(ORIGINS),{r['argument'] for r in checks['direct_checks']})
        return rows,checks

    def test_real_host_launch_every_operand_and_both_position_branches(self):
        for mrope in (False,True):
            with self.subTest(mrope=mrope):
                rows,r=self.launch(inputs(mrope),mrope)
                self.assertEqual({v['verdict'] for v in r['direct_checks']},{'PASS'})
                self.assertEqual(r['verdict'],'UNPROVEN');self.assertFalse(r['device_values_observed'])
                self.assertTrue(all(not row['last_writer']['observed'] for row in rows.values()))

    def test_short_operand_and_missing_position_tail_are_reported(self):
        for index,bad,operand in ((4,Tensor((7,),4),'q_norm_weight_ptr'),
                                  (8,Tensor((4,4,8)),'state_cache_ptr'),
                                  (9,Tensor((7,),8),'state_slots_ptr')):
            with self.subTest(operand=operand):
                values=inputs();values[index]=bad;_,r=self.launch(values)
                self.assertEqual(r['verdict'],'VIOLATION')
                self.assertIn(operand,[v['argument'] for v in r['direct_checks'] if v['verdict']=='VIOLATION'])

    def test_original_host_validation_still_refuses_bad_shape_before_launch(self):
        fn,calls=host_body();values=inputs();values[0]=Tensor((8,15))
        with self.assertRaises(AssertionError):fn(*values,compress_ratio=4,mrope_section=None,rope_pos_offset=8)
        self.assertEqual(calls,[])


if __name__=='__main__':unittest.main()
