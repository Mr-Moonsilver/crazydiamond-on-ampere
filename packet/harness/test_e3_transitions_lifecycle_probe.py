"""Synthetic typed receipt mutation tests only; no GPU/engine evidence."""
import copy
import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'patches/s5_engine'))
import wire_v1 as w
import e3_types as et
import e3_native_transitions_lifecycle_probe as p


class Tests(unittest.TestCase):
    def record(self):return p.predicate_receipt(w,et)
    def test_real_wire_dataclass_predicate_paths_execute(self):
        self.assertEqual(p.predicate_selftest(w,et)['verdict'],'PASS')
    def test_missing_request_cannot_pass(self):
        r=self.record();r['requests'].pop()
        self.assertNotEqual(p.grade_lifecycle(r)['verdict'],'PASS')
    def test_threshold_cannot_be_lowered_by_receipt(self):
        r=self.record();r['requests'][0]['chunks'][0]['rows']=32
        self.assertEqual(p.grade_lifecycle(r)['verdict'],'FAIL')
    def test_closed_nonterminal_row_is_refused(self):
        r=self.record();r['requests'][0]['chunks'][0]['release']['request_closed']=True
        self.assertEqual(p.grade_lifecycle(r)['verdict'],'FAIL')
    def test_no_state_write_stays_inconclusive(self):
        r=self.record();report=r['requests'][0]['chunks'][0]['report']
        report['proofs']=[x for x in report['proofs'] if x['kind']!='state_write']
        self.assertEqual(p.grade_lifecycle(r)['verdict'],'INCONCLUSIVE')
    def test_foreign_proof_transaction_fails(self):
        r=self.record();r['requests'][0]['chunks'][0]['report']['proofs'][0]['ticket']['transaction']['txn_seq']+=1
        self.assertEqual(p.grade_lifecycle(r)['verdict'],'FAIL')
    def test_pending_event_is_not_completion(self):
        r=self.record();r['requests'][0]['chunks'][0]['report']['pending_op_seqs']=[1]
        self.assertEqual(p.grade_lifecycle(r)['verdict'],'FAIL')
    def test_wrong_release_watermark_fails(self):
        r=self.record();r['requests'][0]['chunks'][0]['release']['completed_sequence']+=1
        self.assertEqual(p.grade_lifecycle(r)['verdict'],'FAIL')
    def test_no_observed_reallocation_is_inconclusive(self):
        r=self.record()
        for row in r['requests']:row['physical_reallocated']=[]
        self.assertEqual(p.grade_lifecycle(r)['verdict'],'INCONCLUSIVE')
    def test_wrong_handoff_token_count_fails(self):
        r=self.record();r['requests'][0]['ordinary_tokens']=2
        self.assertEqual(p.grade_lifecycle(r)['verdict'],'FAIL')
    def test_full_certificate_cannot_be_claimed(self):
        r=self.record();r['all_rank_full_certificate']=True
        self.assertEqual(p.grade_lifecycle(r)['verdict'],'FAIL')
    def test_exception_chain_retains_real_cause_context_messages_and_frames(self):
        try:
            try:raise ValueError('inner contract field')
            except ValueError as inner:raise RuntimeError('outer quarantine') from inner
        except RuntimeError as outer:rows=p.exception_chain(outer)
        self.assertEqual(rows[0]['message'],'outer quarantine')
        self.assertEqual(rows[rows[0]['cause']]['message'],'inner contract field')
        self.assertEqual(rows[0]['cause'],rows[0]['context'])
        self.assertTrue(rows[0]['stack'])
        self.assertTrue(rows[rows[0]['cause']]['stack'])
    def test_exception_chain_preserves_implicit_context_without_a_cause(self):
        try:
            try:raise KeyError('inner')
            except KeyError:raise RuntimeError('outer')
        except RuntimeError as outer:rows=p.exception_chain(outer)
        self.assertIsNone(rows[0]['cause'])
        self.assertEqual(rows[rows[0]['context']]['type'],'KeyError')
    def test_fixture_limits_are_machine_readable(self):
        d=json.loads(Path(__file__).with_name('native15_lifecycle_fixture_leaves.json').read_text())
        self.assertFalse(d['clears_box_hold'])
        for leaf in d['scenarios']['worker_lifecycle']['leaves']:
            for name in ('leaf','replaces','equivalent_for','more_convenient_for'):self.assertTrue(leaf[name])
            self.assertFalse(leaf['box_evidence'])


if __name__=='__main__':unittest.main()
