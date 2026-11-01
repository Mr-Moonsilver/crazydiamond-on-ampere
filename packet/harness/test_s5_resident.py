import copy
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from s5_resident import CASES,PLETransport,ledger,validate_prefetch,dma_window_proven
from s5_value import grade,summarize
from test_s5_value import Value


class Resident(unittest.TestCase):
    def test_exact_memory_and_slice_sharing(self):
        r=ledger(200000,4048,1024,207257600,838860,160,1)
        self.assertEqual(r['pages'],50);self.assertEqual(r['kv_added_bytes'],207257600)
        self.assertEqual(r['engine_full_kv_bytes'],1243545600)
        self.assertEqual(r['engine_s3_24_bytes'],596901888)
        self.assertEqual(r['engine_s3_40_bytes'],994836480)
        self.assertEqual(r['ple_chunk_bytes'],2621440)
        self.assertEqual(r['slice_unique_kv_sources'],1)
        self.assertEqual(r['ple_added_device_buffer_bytes'],0)
        self.assertAlmostEqual(r['engine_fp8_ple_host_gib'],48.6)
        ple=ledger(200000,4048,1024,207257600,838860,160,2,retain_kv=False)
        self.assertEqual(ple['kv_added_bytes'],0);self.assertEqual(ple['ple_chunk_bytes'],5242880)
        with self.assertRaises(ValueError):ledger(200000,4048,1024,207257601,838860,160,1)

    def test_pinned_buffer_ownership_rotating_ids_and_reset(self):
        # Execute the actual lookup/reset control flow with an asynchronous
        # copy mock. This proves host buffer fencing, not CUDA/FP8 numerics.
        events=[]
        class Vec(list):
            def __add__(self,v):return Vec(x+v for x in self)
            def __mod__(self,v):return Vec(x%v for x in self)
        class Tensor:
            def __init__(self,data):self.data=copy.deepcopy(data);self.shape=(len(data),len(data[0]))
            def copy_(self,source,non_blocking):
                self.data=copy.deepcopy(source.data);events.append(('copy',non_blocking));return self
            def numel(self):return self.shape[0]*self.shape[1]
            def element_size(self):return 2
        class Event:
            pending=False
            def record(self,stream):self.pending=True;events.append(('record',stream))
            def synchronize(self):self.pending=False;events.append(('fence',))
        o=object.__new__(PLETransport);o.copy_done=Event();o.pending=False;o.prefetch=True
        o.itemsize=2;o.step=0;o.calls=0;o.bytes=0;o.pack_seconds=0.;o.base=Vec([0,4])
        o.packed=Tensor([[0,0],[0,0]]);o.raw=Tensor([[0,0],[0,0]])
        o.lane=SimpleNamespace(pre=True,ple_host=Tensor([[i,i+5] for i in range(5)]),ple_output=o.raw)
        o.original_reset=lambda:events.append(('reset',))
        def select(table,dim,ids,out):
            self.assertFalse(o.copy_done.pending,'CPU overwrote in-flight pinned memory')
            out.data=[list(table.data[i]) for i in ids]
        torch=SimpleNamespace(index_select=select,cuda=SimpleNamespace(current_stream=lambda:17))
        ngram=SimpleNamespace(_lookup_ple_embedding_from_pinned_kernel=None)
        with patch.dict(sys.modules,{'torch':torch,'vllm.models.qwen4_exp.nvidia.ngram_embedding':ngram}):
            o.lookup();self.assertEqual(o.raw.data,[[0,5],[4,9]])
            o.lookup();self.assertEqual(o.raw.data,[[4,9],[3,8]])
            self.assertEqual(events[2],('fence',))
            self.assertEqual(o.stats()['h2d_bytes'],16)
            o.reset();self.assertEqual(o.step,0);self.assertEqual(events[-2:],[('fence',),('reset',)])
            o.lookup();self.assertEqual(o.raw.data,[[0,5],[4,9]])

    def fixture(self):
        d=Value().fixture()
        for r in d['ranks']:
            r['value_probe'].update(prefill_noop=False,resident_kind='kv_ple_prefetch',ple_itemsize=1)
            r['resident_source_equality']=True
            r['resident_ledger']=dict(kind='kv_ple_prefetch',ple_chunk_bytes=2621440)
            r['ple_prefetch_stats']=dict(calls=3,h2d_bytes=3*2621440,pack_seconds=.02)
        return d

    def test_prefetch_accounting_and_equality_fail_closed(self):
        d=self.fixture();self.assertEqual(grade(d)['verdict'],'PASS')
        d['ranks'][0]['ple_prefetch_stats']['h2d_bytes']-=1
        self.assertEqual(grade(d)['verdict'],'INCONCLUSIVE')
        d=self.fixture();d['ranks'][1]['resident_source_equality']=False
        self.assertNotEqual(grade(d)['verdict'],'PASS')
        d=self.fixture();d['ranks'][1]['resident_correctness']['passed']=False
        self.assertEqual(grade(d)['verdict'],'FAIL')
        with self.assertRaises(ValueError):validate_prefetch(dict(calls=2,h2d_bytes=4,pack_seconds=0),3,2)
        with self.assertRaises(ValueError):validate_prefetch(dict(calls=3,h2d_bytes=6,pack_seconds=math.nan),3,2)

    def test_matrix_missing_cases_and_lifecycle(self):
        h=Path(__file__).resolve().parent;root=h.parent.parent/'tmp';root.mkdir(exist_ok=True)
        out=Path(tempfile.mkdtemp(prefix='s5-resident-',dir=root))
        self.addCleanup(shutil.rmtree,out,ignore_errors=True)
        r=summarize(out,CASES);self.assertEqual([x['case'] for x in r['cases']],list(CASES))
        self.assertEqual(r['verdict'],'INCONCLUSIVE')
        s=(h/'runs/q_s5_resident.draft.sh').read_text()
        self.assertEqual(subprocess.run(['bash','-n'],input=s,text=True).returncode,0)
        for bad in ('systemctl stop','systemctl start','docker rm','--rm'):self.assertNotIn(bad,s)
        for name in CASES:self.assertIn('run_case '+name,s)
        self.assertIn('S5_CONNECTIONS=8',s);self.assertIn('S5_PLE_ITEMSIZE=1',s)

    def test_dma_must_land_inside_decode_window(self):
        def row(a,b):return dict(submitted_s=a,observed_s=b)
        m=dict(raw_decode=[row(0,.01),row(1,1.01)],raw_prefill=[row(-.01,.03),row(.04,.4)])
        self.assertTrue(dma_window_proven(m))
        m['raw_prefill'][1]=row(-.02,.4);self.assertFalse(dma_window_proven(m))
        m['raw_prefill'][1]=row(.04,1.1);self.assertFalse(dma_window_proven(m))
        d=self.fixture()
        for r in d['ranks']:
            r['value_probe']['resident_dma']=True
            r['resident_ledger'].update(kv_source_bytes=207257600,pages=50)
            r['resident_dma']=dict(bytes=0,chunks=13,max_outstanding=1,gpu_copy_ms=16,inside_decode_window=True)
        self.assertEqual(grade(d)['verdict'],'INCONCLUSIVE')


if __name__=='__main__':unittest.main()
