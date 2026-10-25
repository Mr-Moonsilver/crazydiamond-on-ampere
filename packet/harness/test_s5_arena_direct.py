"""No GPU: pinned-source transform, actual callback copy/fence and strict grading."""
import ast
import copy
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import numpy as np
from unittest.mock import patch
import sys
from s5_arena_direct import BASE_SHA,GATHER,NAME,ArenaDirect,copy_plan,generate,slice_function,validate
from test_s5_dma_window import DMAWindow,evidence
from s5_value import grade,monitor_lines

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'reference/vllm-src/models/qwen4_exp/nvidia/ops/qsa.py'


def fixture():
    d=DMAWindow().fixture()
    for r in d['ranks']:
        r['value_probe'].update(resident_kind='arena_direct',resident_dma=False,sparse_prefill='0')
        m=r['resident_ledger'];m.update(kind='arena_direct',kv_added_bytes=0,added_kv_window_bytes=0,
             expected_resident_allocation_delta=0,arena_reused=True,arena_allocated_bytes=23*1024)
        r['resident_substitution']['arena_bucket_exact']=dict(passed=True,checks=162,expected=162)
        r['arena_mapping_exact']=dict(passed=True,checks=8,expected=8,cases=[
            dict(name=n,pages=p,buckets=b,passed=True) for n,p,b in
            [('identity',50,3),('reverse',50,3),('fragmented_short',24,2)]])
        for key,mode in [('resident_dma','overlap_resident'),('resident_dma_solo','resident_prefill')]:
            rows=[]
            for step in range(3):
                for layer in range(6):
                    for bucket,(lo,hi) in enumerate(((0,23),(23,46),(46,50))):
                        a=step+.01+layer*.1+bucket*.02
                        rows.append(dict(step=step,layer=layer,page_lo=lo,page_hi=hi,start_s=a,end_s=a+.01,
                                         bytes=(hi-lo)*1024,chunks=(hi-lo+3)//4,gpu_copy_ms=.5))
            rec=dict(mode='every_qsa_bucket',layers=6,windows=0,arena_pages=23,chunk_cap_pages=4,
                max_outstanding_bytes=23*1024,max_buckets_in_flight=1,timing_basis='whole_bucket_stream_span',
                refills=rows,bytes=sum(x['bytes'] for x in rows),chunks=sum(x['chunks'] for x in rows),
                gpu_copy_ms=sum(x['gpu_copy_ms'] for x in rows))
            rec.update(validate(rec,r['modes'][mode],50,1024));r[key]=rec
    return d


class ArenaTests(unittest.TestCase):
    def test_pinned_generator_changes_gather_only_and_rejects_drift(self):
        source=SOURCE.read_text();candidate=generate(source)
        node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==NAME)
        orig=''.join(source.splitlines(keepends=True)[node.lineno-1:node.end_lineno])
        tail=candidate[candidate.index('    num_rows = rows.numel()'):]
        self.assertEqual(tail.replace('            dma_fill(staged, page_lo, page_hi, block_table_row)',GATHER),
                         orig[orig.index('    num_rows = rows.numel()'):])
        self.assertIn('arena-direct is dense-only',candidate)
        self.assertIn('arena-direct rejects retained shortcut',candidate)
        with self.assertRaises(ValueError):generate(source+'\n')

    def test_slice_clone_redirects_only_staged_attention(self):
        calls=[]
        source='''def qsa(self,x,i):
    if self.pre:
        return qsa_sparse_paged_attention_staged(x,i)
    return decode(x,i)
'''
        ns=dict(_s5_arena_staged=lambda *a:calls.append(('dma',a)),decode=lambda *a:calls.append(('decode',a)))
        exec(slice_function(source),ns)
        ns['qsa'](NS(pre=True),3,4);ns['qsa'](NS(pre=False),5,6)
        self.assertEqual(calls,[('dma',(3,4)),('decode',(5,6))])
        with self.assertRaises(ValueError):slice_function('def qsa(self,x,i):return x')

    def test_copy_plan_preserves_mapping_duplicates_short_tail_and_bounds(self):
        self.assertEqual(copy_plan(list(range(50)),23,46,50,23),[(23,0,4),(27,4,4),(31,8,4),(35,12,4),(39,16,4),(43,20,3)])
        self.assertEqual(copy_plan([8,9,3,3,4],0,5,10,23),[(8,0,2),(3,2,1),(3,3,2)])
        self.assertEqual(copy_plan(list(range(50)),46,50,50,23),[(46,0,4)])
        for ids,lo,hi in [([0,99],0,2),([-1],0,1),([0],0,2),([True],0,1),(list(range(50)),0,24)]:
            with self.assertRaises(ValueError):copy_plan(ids,lo,hi,50,23)

    def test_strict_grader_counts_all_buckets_solo_and_inside_window(self):
        d=fixture();g=grade(d);self.assertEqual(g['verdict'],'PASS')
        lines='\n'.join(monitor_lines('arena-direct34',g))
        for text in ('S5_ARENA','windows=0','layer_exact=54/54','bucket_exact=162/162','mapping_exact=8/8','bytes_per_chunk=307200'):
            self.assertIn(text,lines)
        for change in ('missing','solo','bytes','allocation','oracle','mapping','sparse','policy'):
            bad=copy.deepcopy(d);r=bad['ranks'][0]
            if change=='missing':r['resident_dma']['refills'].pop()
            elif change=='solo':r['resident_dma_solo']['refills'].pop()
            elif change=='bytes':r['resident_dma']['refills'][0]['bytes']-=1
            elif change=='allocation':r['resident_ledger']['kv_added_bytes']=1
            elif change=='oracle':r['resident_substitution']['arena_bucket_exact']['checks']=161
            elif change=='mapping':r['arena_mapping_exact']['cases'][2]['buckets']=1
            elif change=='sparse':r['value_probe']['sparse_prefill']='1'
            else:r['resident_dma']['max_buckets_in_flight']=2
            self.assertEqual(grade(bad)['verdict'],'INCONCLUSIVE',change)
        bad=copy.deepcopy(d);del bad['ranks'][0]['modes']
        bad['ranks'][1]['resident_substitution']['arena_bucket_exact']['passed']=False
        self.assertEqual(grade(bad)['verdict'],'FAIL')

    def fake(self,mapping):
        actions=[];events=[];active=[None]
        class T:
            def __init__(self,data):self.a=data;self.shape=data.shape
            def data_ptr(self):return self.a.ctypes.data
            def __getitem__(self,x):return T(self.a[x])
            def to(self,dtype):return T(self.a.astype(dtype))
            def cpu(self):
                if actions:raise AssertionError('CPU snapshot before DMA fence')
                events.append('snapshot');return T(self.a.copy())
            def view(self,dtype):return T(self.a.view(dtype))
            def copy_(self,source,non_blocking):
                self_ref=self;src=source
                if active[0] is None:raise AssertionError('wrong copy stream')
                actions.append(lambda:self_ref.a.__setitem__(slice(None),src.a))
                events.append(('copy',self.shape[0],non_blocking));return self
        class Stream:
            def wait_stream(self,other):events.append('wait-consumer')
        class Context:
            def __enter__(self):active[0]=stream
            def __exit__(self,*args):active[0]=None
        class Event:
            def record(self):events.append('record')
            def synchronize(self):
                events.append('fence')
                for fn in actions:fn()
                actions.clear()
            def elapsed_time(self,end):return 1.
        stream=Stream();current=NS(wait_event=lambda e:events.append('wait-copy'))
        def index_select(source,axis,physical,out):
            events.append('reference');out.a[:]=np.take(source.a,physical.a,axis=axis)
        torch=NS(cuda=NS(current_stream=lambda:current,stream=lambda s:Context()),int64=np.int64,uint8=np.uint8,
                 equal=lambda a,b:np.array_equal(a.a,b.a),index_select=index_select)
        host=T(np.arange(50*8,dtype=np.uint16).reshape(50,1,2,4))
        arena=T(np.zeros((23,1,2,4),dtype=np.uint16));table=T(np.array([mapping],dtype=np.int32))
        pre=NS(host=host,arena=arena,table=table)
        o=object.__new__(ArenaDirect);o.pre=pre;o.torch=torch;o.original_kv=host
        o.mapping=mapping;o.arena_id=arena.data_ptr();o.table_id=table.data_ptr()
        o.stream=stream;o.begin=Event();o.end=Event();o.verify_layers=True;o.bucket_checks=0
        o.page_bytes=16;o.step=0;o.layer=0;o.refills=[]
        return o,host,arena,table,events

    def test_real_callback_byte_oracle_fragmented_mapping_and_arena_reuse(self):
        for mapping in (list(range(50)),list(reversed(range(50))),list(range(0,50,2))+list(range(1,50,2))):
            o,host,arena,table,events=self.fake(mapping)
            for lo,hi in ((0,23),(23,46),(46,50)):
                o.fill(arena[:hi-lo],lo,hi,table[0])
                np.testing.assert_array_equal(arena.a[:hi-lo],host.a[mapping[lo:hi]])
            self.assertEqual(o.bucket_checks,3)
            self.assertEqual(sum(r['bytes'] for r in o.refills),50*16)
            self.assertEqual(events.count('fence'),3);self.assertEqual(events.count('wait-consumer'),3)
            self.assertEqual(events.count('reference'),3)
            self.assertTrue(all(e[1]<=4 for e in events if isinstance(e,tuple) and e[0]=='copy'))
            self.assertEqual(o.refills[-1]['page_hi']-o.refills[-1]['page_lo'],4)
            # Measurement has no CPU snapshots and no UVA gather.
            o.verify_layers=False;events.clear();o.fill(arena[:23],0,23,table[0])
            self.assertNotIn('snapshot',events);self.assertNotIn('reference',events)

    def test_bucket_corruption_is_hard_and_bad_mapping_is_rejected_before_copy(self):
        class CorrectnessError(RuntimeError):pass
        o,host,arena,table,events=self.fake(list(range(50)))
        original=o.torch.index_select
        def wrong(*a,**kw):
            original(*a,**kw);kw['out'].a[0,0,0,0]^=1
        o.torch.index_select=wrong
        with patch.dict(sys.modules,{'s5_segment_gpu':NS(CorrectnessError=CorrectnessError)}):
            with self.assertRaises(CorrectnessError):o.fill(arena[:23],0,23,table[0])
        self.assertEqual(o.bucket_checks,0)
        o,host,arena,table,events=self.fake([0]*49+[99])
        with self.assertRaises(ValueError):o.fill(arena[:4],46,50,table[0])
        self.assertFalse(events)


if __name__=='__main__':unittest.main()
