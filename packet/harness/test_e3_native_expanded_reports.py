"""Actual driver/bridge source stream with synthetic device leaves; CPU only."""
import copy
import unittest
import test_e3_native_measurement as fixtures

nm=fixtures.nm
ed=fixtures.bridge_suite.ed


def expanded_stack(pauses=200):
    s=fixtures.make_long_stack(tokens=2,layers=48)
    entries=fixtures.capture_record(s)
    s.drive(lambda:s.driver._split_ready)
    for rank in (0,1):s.worker(rank).lane.pause_counts[0]=pauses
    s.drive(lambda:s.driver.phase=='finalize',limit=pauses*5+500)
    s.complete_everything();s.drive(lambda:s.driver.tx is None)
    start,end=fixtures.window_of(entries)
    evidence=s.driver.measurement_census.evidence(start,end,now_ns=end+1)
    return s,entries,evidence


def regrade(s,entries,**kw):
    census=fixtures.census_for(s,pool_chunk_capacity=True,**kw)
    for entry in entries:census.ingest(entry)
    start,end=fixtures.window_of(entries)
    return census,census.evidence(start,end,now_ns=end+1)


class VectorTests(unittest.TestCase):
    def test_exact_vector_boundaries_and_three_level_tree(self):
        for count in (0,64,65,4096,4097,nm.MAX_JOURNAL_VECTOR_ITEMS):
            for proofs in (False,True):
                values=([dict(kind='gpu_event',op_seq=i,completion_seq=i+1) for i in range(count)]
                        if proofs else list(range(count)))
                packed=nm.encode_report_vector(values)
                self.assertEqual(nm.decode_report_vector(packed),values)
                if count<=64:self.assertEqual(packed,values)
                else:
                    def width(node):
                        if isinstance(node,list):
                            self.assertLessEqual(len(node),64)
                            for n in node:width(n)
                    width(packed['items'])
        with self.assertRaises(ValueError):
            nm.encode_report_vector([0]*(nm.MAX_JOURNAL_VECTOR_ITEMS+1))

    def test_malformed_tree_refused(self):
        base=nm.encode_report_vector(list(range(65)))
        for kind in ('length','levels','extra','short','wide'):
            v=copy.deepcopy(base)
            if kind=='length':v['length']-=1
            if kind=='levels':v['levels']=True
            if kind=='extra':v['unknown']=0
            if kind=='short':v['items'][0].pop()
            if kind=='wide':v['items'][0].append(64)
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                nm.decode_report_vector(v)

    def test_general_metadata_limits_preserved(self):
        with self.assertRaisesRegex(ed.DriverError,'sequence is too long'):
            ed._bounded_value(list(range(65)))
        with self.assertRaisesRegex(ed.DriverError,'nested too deeply'):
            ed._bounded_value([[[[0]]]])


class ExpandedSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stack,cls.entries,cls.evidence=expanded_stack()

    def test_actual_extended_chunk_closes_with_exact_join(self):
        s,entries,e=self.stack,self.entries,self.evidence
        self.assertIsNone(s.driver.error)
        self.assertTrue(e['completeness']['complete'],e['reasons'])
        self.assertEqual(e['segments'][0]['segment_count'],258)
        self.assertEqual(e['segments'][0]['stage_segment_counts'],{'0':229,'1':29})
        self.assertEqual([len(r['proofs']) for r in e['completions']],[230,230,31,31])
        self.assertTrue(s.driver._journal_dropped>0)
        self.assertFalse(e['completeness']['raw_journal_consulted'])
        self.assertFalse(e['completeness']['raw_journal_flag_mutated'])
        paged=[x for x in entries if x['kind']=='report'
               and isinstance(x['completed_op_seqs'],dict)]
        self.assertTrue(paged)
        for row in paged:
            ids=nm.decode_report_vector(row['completed_op_seqs'])
            self.assertEqual(ids,sorted(set(ids)))
            proofs=nm.decode_report_vector(row['proofs'])
            self.assertTrue(all(p['op_seq'] in ids for p in proofs))

    def test_existing_aggregate_budgets_unchanged(self):
        stats=self.stack.driver.measurement_census.stats()['capacity']
        self.assertEqual(stats['segments_total_limit'],128*8192)
        self.assertEqual(stats['proofs_per_rank_total_limit'],128*8192)
        self.assertEqual(stats['completed_per_rank_total_limit'],128*8192)
        self.assertEqual(stats['retained_segments_total'],258)
        self.assertEqual(stats['retained_proofs_per_rank'],{0:230,1:230,2:31,3:31})

    def test_aggregate_segment_capacity_still_refuses(self):
        c,e=regrade(self.stack,self.entries,max_chunks_total=2,max_segments_per_chunk=128)
        self.assertFalse(e['completeness']['complete'])
        self.assertIn('segment-capacity',c.stats()['overflow']['reasons'])

    def test_aggregate_proof_capacity_still_refuses(self):
        c,e=regrade(self.stack,self.entries,max_chunks_total=2,max_segments_per_chunk=129,max_proofs_per_rank=114)
        self.assertFalse(e['completeness']['complete'])
        self.assertIn('proof-capacity',c.stats()['overflow']['reasons'])

    def test_exact_pooled_capacity_passes(self):
        c,e=regrade(self.stack,self.entries,max_chunks_total=2,max_segments_per_chunk=129,max_proofs_per_rank=115)
        self.assertTrue(e['completeness']['complete'],e['reasons'])
        self.assertEqual(c.stats()['overflow']['count'],0)

    def test_missing_expanded_segment_is_not_hidden(self):
        entries=copy.deepcopy(self.entries)
        target=next(x for x in entries if x['kind']=='segment' and x['stage']==0 and x['op_seq']==150)
        entries.remove(target)
        _,e=regrade(self.stack,entries)
        self.assertFalse(e['completeness']['complete'])
        self.assertTrue(any('segment-set-differs-from-report' in x for x in e['reasons']))

    def test_missing_paged_proof_is_not_hidden(self):
        entries=copy.deepcopy(self.entries)
        for row in entries:
            if row['kind']!='report' or row['worker_rank']!=0:continue
            proofs=nm.decode_report_vector(row['proofs'])
            row['proofs']=nm.encode_report_vector([p for p in proofs if not(p['kind']=='gpu_event' and p['op_seq']==150)])
        _,e=regrade(self.stack,entries)
        self.assertFalse(e['completeness']['complete'])
        self.assertTrue(any('gpu-event-set-differs-from-report' in x for x in e['reasons']))

    def test_malformed_page_is_inconclusive(self):
        entries=copy.deepcopy(self.entries)
        row=next(x for x in entries if x['kind']=='report' and isinstance(x['proofs'],dict))
        row['proofs']['length']+=1
        _,e=regrade(self.stack,entries)
        self.assertFalse(e['completeness']['complete'])
        self.assertTrue(any('report-vector-invalid' in x for x in e['reasons']))

    def test_default_per_chunk_refusal_retained_for_legacy_callers(self):
        c=fixtures.census_for(self.stack)
        for entry in self.entries:c.ingest(entry)
        a,b=fixtures.window_of(self.entries)
        self.assertFalse(c.evidence(a,b,now_ns=b+1)['completeness']['complete'])
        self.assertFalse(c.pool_chunk_capacity)

    def test_first_observation_and_counters_do_not_move_on_duplicate(self):
        c=fixtures.census_for(self.stack,pool_chunk_capacity=True)
        report=next(x for x in reversed(self.entries) if x['kind']=='report' and x['worker_rank']==0)
        for entry in self.entries:
            c.ingest(entry)
            if entry is report:
                stats=c.stats()['capacity'];chunk=c._chunks[report['transaction']]
                proofs=copy.deepcopy(chunk['ranks'][0]['proofs'])
                c.ingest(report)
                self.assertEqual(proofs,chunk['ranks'][0]['proofs'])
                for key in ('retained_segments_total','retained_proofs_per_rank','retained_completed_per_rank'):
                    self.assertEqual(stats[key],c.stats()['capacity'][key])
        a,b=fixtures.window_of(self.entries)
        self.assertTrue(c.evidence(a,b,now_ns=b+1)['completeness']['complete'])


if __name__=='__main__':unittest.main()
