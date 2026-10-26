import ast
import copy
import hashlib
from pathlib import Path
import subprocess
import unittest
import vmid_job4 as job
from vmid_qsa_dma import BucketLease, generate, GATHER, SOURCE_SHA, NAME
from vmid_sparse_plan import cases,make_plan,ROW_BYTES

ROOT=Path(__file__).resolve().parents[1]

class Event:
    def __init__(self,ready=False):self.ready=ready
    def query(self):return self.ready


def good():
    r=dict(source_sha256=SOURCE_SHA,geometry=dict(layers=4,query_rows=80,query_heads=12,head_dim=256,topk=2051,
           page_rows=4048,arena_pages=24,piece_pages=4,descriptor_cap=256),cases=[],cancellation=[],
           allocator_peak_allocated=400*1024**2,allocator_peak_reserved=450*1024**2,
           distinct_host_layers=True,cleanup_ok=True)
    for name,sel,mapping,_ in cases():
        p=make_plan(sel,mapping,(1,)*52,1)
        for layer in range(4):
            r['cases'].append(dict(name=name,layer=layer,selected_rows=len(sel),fallback=p.fallback,
                reference_calls=0 if p.fallback else 1,candidate_calls=0 if p.fallback else 1,
                bucket_checks=len(p.buckets),candidate_kernels=len(p.buckets),
                dma_pieces=sum(len(b.pieces) for b in p.buckets),bytes_checked=sum(b.rows for b in p.buckets)*ROW_BYTES,
                **{k:(None if p.fallback and k=='output_exact' else True) for k in job.BOOLS}))
    for layer in range(4):
        r['cancellation'].append(dict(layer=layer,cancelled=True,partial_prefix_exact=True,partial_tail_poison=True,
            consumer_refused=True,reuse_refused=True,output_untouched=True,drained=True,refill_output_exact=True,
            kernels_before_refill=0,published_before_refill=0,submitted_pieces=12,planned_pieces=24))
    return r


class SourceTests(unittest.TestCase):
    def test_pinned_gather_only_inverse(self):
        source=(ROOT/'patches/retain/ops_qsa.py.v8.1').read_text()
        generated=generate(source)
        recovered=generated.replace('def dma_attention(', 'def '+NAME+'(',1).replace('    *, dma_fill,\n','',1)
        recovered=recovered.replace('            dma_fill(staged, selected, page_lo * page_size)',GATHER,1)
        node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==NAME)
        self.assertEqual(recovered,''.join(source.splitlines(keepends=True)[node.lineno-1:node.end_lineno]))
        args=ast.parse(generated).body[0].args
        self.assertEqual([x.arg for x in args.args][:5],['q','kv_cache','logical_indices','block_table_row','rows'])
        self.assertEqual(hashlib.sha256(source.encode()).hexdigest(),SOURCE_SHA)

    def test_wrong_source_refused(self):
        with self.assertRaises(ValueError):generate('def qsa_sparse_paged_attention_staged(): pass')

    def test_callback_before_attention(self):
        source=(ROOT/'patches/retain/ops_qsa.py.v8.1').read_text()
        g=generate(source)
        self.assertLess(g.index('dma_fill(staged'),g.index('_qsa_sparse_staged_gqa_kernel['))
        self.assertIn('FIRST_BUCKET=bucket == 0',g)
        self.assertIn('LAST_BUCKET=bucket == num_buckets - 1',g)

    def test_selection_capacity_and_sparse_route(self):
        for name,sel,mapping,_ in cases():
            self.assertLessEqual(len(sel),job.QUERY_ROWS*job.TOPK,name)
            self.assertLess(len(sel),.8*len(mapping)*4048,name)
        # This test is deliberately QSA-only. No accidental collective/engine imports.
        source=(ROOT/'harness/vmid_job4.py').read_text()
        self.assertNotIn('import s5_segment_gpu',source)
        self.assertIn('qr,qc=ref,actual',source)

    def test_queue_bounds_and_names(self):
        path=ROOT/'harness/runs/vmid_job4.sh'
        subprocess.run(['bash','-n',str(path)],check=True)
        text=path.read_text()
        for token in ('runs/job4','flashnext-job4','480s','--memory 4g','--network none','VLLM_QSA_KVO_SPARSE_PREFILL=1','QSA_STAGE_TIMING=0'):
            self.assertIn(token,text)
        self.assertNotIn('systemctl',text)
        self.assertIn('set_per_process_memory_fraction(.125,0)',(ROOT/'harness/vmid_job4.py').read_text())


class LeaseTests(unittest.TestCase):
    def test_no_consumer_or_reuse_during_partial_fill(self):
        l=BucketLease();l.begin()
        with self.assertRaises(RuntimeError):l.consume()
        with self.assertRaises(RuntimeError):l.begin()
        with self.assertRaises(RuntimeError):l.publish(Event(False))
        self.assertEqual(l.published,0)

    def test_cancel_requires_drain_before_reuse(self):
        l=BucketLease();e=Event();l.begin();l.abort(e)
        for action in (l.consume,l.begin,l.drain):
            with self.assertRaises(RuntimeError):action()
        e.ready=True;l.drain();l.begin();l.publish(e);l.consume();l.complete(e)
        self.assertEqual(l.state,'free');self.assertEqual(l.published,1)

    def test_attention_completion_protects_reuse(self):
        l=BucketLease();l.begin();l.publish(Event(True));l.consume()
        with self.assertRaises(RuntimeError):l.begin()
        e=Event()
        with self.assertRaises(RuntimeError):l.complete(e)
        e.ready=True;l.complete(e);l.begin()


class GradeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.fixture=good()
    def fresh(self):return copy.deepcopy(self.fixture)
    def test_complete(self):self.assertEqual(job.grade(self.fresh())['verdict'],'PASS')
    def test_output_mismatch_fails(self):
        r=self.fresh();r['cases'][4]['output_exact']=False
        self.assertEqual(job.grade(r)['verdict'],'FAIL')
    def test_missing_output_inconclusive(self):
        r=self.fresh();r['cases'][4].pop('output_exact')
        self.assertEqual(job.grade(r)['verdict'],'INCONCLUSIVE')
    def test_partial_layer_coverage_not_pass(self):
        r=self.fresh();r['cases'].pop()
        self.assertEqual(job.grade(r)['verdict'],'INCONCLUSIVE')
    def test_missing_multibucket_not_pass(self):
        r=self.fresh();x=next(x for x in r['cases'] if x['name']=='arena_25_pages_tail');x['candidate_kernels']=1
        self.assertEqual(job.grade(r)['verdict'],'INCONCLUSIVE')
    def test_fallback_cannot_claim_attention(self):
        r=self.fresh();r['cases'][-1]['output_exact']=True
        self.assertEqual(job.grade(r)['verdict'],'FAIL')
    def test_cancellation_false_publication_fails(self):
        r=self.fresh();r['cancellation'][0]['kernels_before_refill']=1
        self.assertEqual(job.grade(r)['verdict'],'FAIL')
    def test_missing_cancellation_proof(self):
        r=self.fresh();r['cancellation'][0].pop('reuse_refused')
        self.assertEqual(job.grade(r)['verdict'],'INCONCLUSIVE')
    def test_memory_growth_fails(self):
        r=self.fresh();r['allocator_peak_reserved']=1024**3+1
        self.assertEqual(job.grade(r)['verdict'],'FAIL')
    def test_byte_mismatch_fails(self):
        r=self.fresh();r['cases'][4]['bytes_exact']=False
        self.assertEqual(job.grade(r)['verdict'],'FAIL')

if __name__=='__main__':unittest.main()
