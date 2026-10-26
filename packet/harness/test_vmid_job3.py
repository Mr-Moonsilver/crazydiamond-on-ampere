"""Independent CPU oracle for the compact sparse DMA plan; no torch/GPU import."""
import copy
from pathlib import Path
import random
import subprocess
import unittest
import vmid_job3 as j
import vmid_sparse_plan as p


class PlanTests(unittest.TestCase):
    def assert_plan(self, chosen, mapping, *, rows=7, arena_pages=3, cap=256):
        plan=p.make_plan(chosen,mapping,(1,)*12,1,page_rows=rows,
                         arena_pages=arena_pages,max_pieces=cap)
        self.assertIsNone(plan.fallback)
        # Independent row-number/byte-pattern oracle, not reconstructed from
        # the DMA plan. Duplicate physical page mappings remain legal.
        source=[bytes((i%251,(i//251)%251,19,83)) for i in range(12*rows)]
        expected=[source[mapping[t//rows]*rows+t%rows] for t in chosen]
        copied=[]
        for b in plan.buckets:
            dst=[b'poison']*(arena_pages*rows)
            pos=0
            for x in b.pieces:
                self.assertEqual(x.arena_row,pos)
                self.assertGreater(x.rows,0)
                self.assertLessEqual(x.rows,4*rows)
                self.assertLessEqual((x.source_row+x.rows-1)//rows-x.source_row//rows+1,4)
                dst[x.arena_row:x.arena_row+x.rows]=source[x.source_row:x.source_row+x.rows]
                pos+=x.rows
            self.assertEqual(pos,b.rows)
            self.assertEqual(dst[b.rows:],[b'poison']*(arena_pages*rows-b.rows))
            copied+=dst[:b.rows]
        self.assertEqual(copied,expected)
        return plan

    def test_random_permuted_and_aliased_sources_match_direct_cpu_selection(self):
        rng=random.Random(7126)
        for _ in range(100):
            mapping=tuple(rng.randrange(12) for _ in range(9))
            chosen=tuple(sorted(rng.sample(range(63),rng.randrange(64))))
            self.assert_plan(chosen,mapping)

    def test_unaligned_run_never_spans_five_physical_pages(self):
        plan=self.assert_plan(tuple(range(6,45)),tuple(range(9)),arena_pages=8)
        self.assertEqual(plan.buckets[0].pieces[0].rows,22)
        self.assertEqual(plan.buckets[0].pieces[1].source_row,28)

    def test_buckets_compact_after_arena_boundary(self):
        plan=self.assert_plan(tuple(range(31)),(4,2,2,8,1,0,6,3,7))
        self.assertEqual([(b.compact_start,b.rows) for b in plan.buckets],[(0,21),(21,10)])

    def test_descriptor_cap_rejects_whole_plan_even_if_first_bucket_fits(self):
        # First bucket is contiguous; second needs two pieces. No prefix plan
        # survives a whole-operation refusal.
        plan=p.make_plan((0,1,2,3,5,7),(0,1),(1,1),1,
                         page_rows=4,arena_pages=1,max_pieces=1)
        self.assertEqual(plan.fallback,'descriptor_cap')
        self.assertEqual(plan.buckets,())

    def test_invalid_selection_and_sources_are_refused(self):
        for chosen in ((1,1),(2,1),(-1,), (8,), (True,)):
            with self.subTest(chosen=chosen),self.assertRaises(ValueError):
                p.make_plan(chosen,(0,1),(1,1),1,page_rows=4)
        for mapping,versions,epoch in (((2,),(1,1),1),((True,),(1,),1),((0,),(),1),((0,),(True,),1),((0,),(1,),True)):
            with self.assertRaises(ValueError):p.make_plan((),mapping,versions,epoch)

    def test_stale_epoch_table_or_source_version_refused(self):
        plan=p.make_plan((1,),(0,1),(2,3),5,page_rows=4)
        p.validate_source(plan,(0,1),(2,3),5)
        for mapping,versions,epoch in (((1,0),(2,3),5),((0,1),(2,4),5),((0,1),(2,3),6),((False,1),(2,3),5)):
            with self.assertRaises(ValueError):p.validate_source(plan,mapping,versions,epoch)

    def test_declared_cases_and_piece_budgets(self):
        for name,chosen,mapping,fallback in p.cases():
            plan=p.make_plan(chosen,mapping,(1,)*52,1)
            self.assertEqual(bool(plan.fallback),fallback,name)
            if fallback:
                with self.assertRaises(ValueError):p.validate_source(plan,mapping,(1,)*52,1)
            else:
                self.assertEqual(sum(b.rows for b in plan.buckets),len(chosen))
                self.assertTrue(all(len(b.pieces)<=256 for b in plan.buckets))


def complete():
    rows=[]
    for name,chosen,mapping,fallback in p.cases():
        n=len(chosen);b=(n+p.ARENA_PAGES*p.PAGE_ROWS-1)//(p.ARENA_PAGES*p.PAGE_ROWS)
        rows.append(dict(name=name,selected_rows=n,expected_fallback=fallback,
            fallback='descriptor_cap' if fallback else None,
            dma_pieces=0 if fallback else min(n,b*24),
            dma_bytes=0 if fallback else n*p.ROW_BYTES,
            bytes_checked=0 if fallback else n*p.ROW_BYTES,
            buckets=0 if fallback else b,
            max_piece_bytes=0 if fallback or not n else min(n,p.PAGE_ROWS)*p.ROW_BYTES,
            max_source_page_span=0 if fallback or not n else 1,
            max_bucket_pieces=0 if fallback or not n else min(n,24),
            bytes_exact=True,tail_unchanged=True,copy_and_consumer_complete=True))
    return dict(cases=rows,failures=[],missing=[],cleanup_ok=True,
        allocator_peak_allocated=100*1024**2,allocator_peak_reserved=100*1024**2,
        arena_bytes=p.ARENA_PAGES*p.PAGE_ROWS*p.ROW_BYTES,page_rows=p.PAGE_ROWS,
        row_bytes=p.ROW_BYTES,arena_pages=p.ARENA_PAGES,piece_pages=p.PIECE_PAGES,
        descriptor_cap=p.MAX_PIECES)


class GradeTests(unittest.TestCase):
    def test_complete(self):self.assertEqual(j.grade(complete())['verdict'],'PASS')

    def test_mismatch_and_unsafe_reuse_fail(self):
        for key in ('bytes_exact','tail_unchanged','copy_and_consumer_complete'):
            r=complete();r['cases'][1][key]=False
            self.assertEqual(j.grade(r)['verdict'],'FAIL')

    def test_missing_byte_or_piece_proof_not_pass(self):
        for key in ('bytes_checked','dma_bytes','buckets','max_piece_bytes','max_source_page_span','max_bucket_pieces'):
            r=complete();del r['cases'][1][key]
            self.assertEqual(j.grade(r)['verdict'],'INCONCLUSIVE')

    def test_transfer_bounds_fail(self):
        for key,value in (('max_piece_bytes',4*p.PAGE_ROWS*p.ROW_BYTES+1),('max_source_page_span',5),('max_bucket_pieces',257)):
            r=complete();r['cases'][4][key]=value
            self.assertEqual(j.grade(r)['verdict'],'FAIL')

    def test_fallback_cannot_submit_a_prefix(self):
        r=complete();r['cases'][-1]['dma_pieces']=1
        self.assertEqual(j.grade(r)['verdict'],'FAIL')

    def test_empty_and_missing_cases_do_not_claim_coverage(self):
        r=complete();r['cases']=r['cases'][:-1]
        self.assertEqual(j.grade(r)['verdict'],'INCONCLUSIVE')
        r=complete();r['cases'][0]['dma_pieces']=1
        self.assertEqual(j.grade(r)['verdict'],'FAIL')

    def test_cleanup_failure_is_failure(self):
        r=complete();r['cleanup_ok']=False
        self.assertEqual(j.grade(r)['verdict'],'FAIL')

    def test_queue_is_vm_only_and_uses_fraction_before_cuda_allocations(self):
        root=Path(__file__).resolve().parent
        subprocess.run(['bash','-n',str(root/'runs/vmid_job3.sh')],check=True)
        text=(root/'runs/vmid_job3.sh').read_text()
        self.assertIn('--rm --gpus all --name flashnext-job3',text)
        self.assertIn('Q=/opt/flashnext/runs/job3',text)
        self.assertNotIn('systemctl',text)
        self.assertNotIn('flashnext-job2',text)
        source=(root/'vmid_job3.py').read_text()
        self.assertLess(source.index('set_per_process_memory_fraction(0.125'),source.index('torch.empty('))
        self.assertIn('consumed.synchronize()',source)
        self.assertNotIn('LLM(',source)


if __name__=='__main__':unittest.main()
