import copy
import math
from pathlib import Path
import subprocess
import unittest
from s5_variants import SplitController,score_profile

class Controller(unittest.TestCase):
    def setUp(self):
        self.table={(4,s):dict(validated=True,decode_share=1.,p99_ratio=1.,prefill_share=p) for s,p in [(24,.85),(34,.75),(44,.60)]}
    def test_fences_and_unknown_cost_block_switch(self):
        c=SplitController(self.table)
        for _ in range(5):self.assertFalse(c.choose(4,False,.1,1.)['switch'])
        self.assertFalse(c.choose(4,True,None,1.)['switch'])
        self.assertFalse(c.choose(4,True,2.,1.)['switch'])
        self.assertEqual(c.choose(4,True,.1,1.)['split'],24)
    def test_hysteresis_and_uncalibrated_load(self):
        c=SplitController(self.table)
        for _ in range(3):self.assertFalse(c.choose(4,True,.1,1.)['switch'])
        self.assertTrue(c.choose(4,True,.1,1.)['switch'])
        self.assertEqual(c.choose(5,True,.1,1.)['reason'],'infeasible_or_uncalibrated')
    def test_never_interpolate_or_use_unsafe_candidate(self):
        for v in self.table.values():v['decode_share']=.94
        self.assertEqual(SplitController(self.table).choose(4,True,.1,1.)['reason'],'infeasible_or_uncalibrated')
        self.table[4,24]['decode_share']=1.;self.table[4,24]['validated']=False
        self.assertFalse(SplitController(self.table).choose(4,True,.1,1.)['switch'])

class Frontier(unittest.TestCase):
    def fixture(self):
        m=dict(decode=dict(decode_steps_s=100.,gap_p99_ms=2.),prefill=dict(prefill_steps_s_upper=10.),
               decode_split=dict(decode_steps_s=99.),overlap=dict(decode_steps_s=96.,gap_p99_ms=2.3,
               prefill_steps_s=8.2,prefill_covers_decode_tail=True,prefill_tail_order_ambiguous=False))
        return dict(ranks=[dict(rank=i,completed=True,audit=dict(verdict='PASS'),modes=copy.deepcopy(m)) for i in (0,1)])
    def test_worst_rank_and_missing_audit(self):
        d=self.fixture();self.assertTrue(score_profile(d)['decode_gate'])
        d['ranks'][1]['modes']['overlap']['decode_steps_s']=90.
        self.assertFalse(score_profile(d)['decode_gate'])
        d['ranks'][1]['audit']['verdict']='INCONCLUSIVE'
        with self.assertRaises(AssertionError):score_profile(d)
    def test_uncertain_tail_not_pass(self):
        d=self.fixture();d['ranks'][0]['modes']['overlap']['prefill_tail_order_ambiguous']=True
        with self.assertRaises(AssertionError):score_profile(d)
    def test_queue_syntax_and_runtime_scope(self):
        root=Path(__file__).resolve().parent
        for name in ('runs/q_s5_variant.draft.sh','runs/q_s5_ladder.draft.sh','s5_variant_entry.sh'):
            s=(root/name).read_text();self.assertEqual(subprocess.run(['bash','-n'],input=s,text=True).returncode,0)
            for bad in ('systemctl stop','systemctl start','docker rm','--rm'):self.assertNotIn(bad,s)
        self.assertNotIn('globaltimer',(root/'s5_terminal_mode.py').read_text().split('import threading',1)[1])
if __name__=='__main__':unittest.main()
