import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from s5_value import comparison,grade,monitor_lines,summarize,table
from s5_collective_control import local_prefill_collectives

class Value(unittest.TestCase):
    def mode(self,rate=400,gaps=None):
        gaps=gaps or [2.,2.,2.,2.,2.,2.,2.,2.,4.,8.]
        return dict(decode_steps_s=rate,raw_decode=[{}]*11,gpu_completion_gaps_ms=gaps,
                    gap_p99_ms=4.,raw_prefill=[{}]*3,prefill_steps_s=16.,prefill_steps_s_upper=18.,
                    prefill_covers_decode_tail=True,prefill_tail_order_ambiguous=False)
    def fixture(self):
        corr=dict(passed=True,decode=dict(bitwise_first=True,concurrent_states_match=True),prefill=dict(bitwise_first=True,concurrent_states_match=True))
        ranks=[]
        for i in (0,1):
            ranks.append(dict(rank=i,completed=True,audit=dict(verdict='PASS'),max_connections='8',steps=11,
              requested_split=24,actual_split=[24,58],spec=dict(layers=4),value_probe=dict(enabled=True,prefill_noop=True,resident_prefill=True),
              correctness=copy.deepcopy(corr),noop_correctness=copy.deepcopy(corr),resident_correctness=copy.deepcopy(corr),
              noop_collective_calls=27,modes=dict(decode=self.mode(),decode_split=self.mode(),overlap=self.mode(200),
              overlap_noop=self.mode(300),overlap_resident=self.mode(350),prefill=dict(prefill_steps_s=25),
              noop_prefill=dict(prefill_steps_s=25),resident_prefill=dict(prefill_steps_s=25))))
        return dict(ranks=ranks)
    def test_strict_spike_threshold_and_per_rank(self):
        r=grade(self.fixture());self.assertEqual(r['verdict'],'PASS') # Target miss is NOT failure.
        self.assertEqual(len(r['ranks']),6);a=r['ranks'][0]
        self.assertEqual(a['spike_threshold_ms'],4.);self.assertEqual(a['spike_count_gt_2x_solo'],1)
        self.assertEqual(a['decode_share_vs_solo'],.5);self.assertEqual(a['prefill_share'],.64)
        self.assertEqual(a['vs_80_80_target'],'below')
        s=monitor_lines('test',r)[0]
        for text in ('rank=0','gap_p50_ms=','gap_p99_ms=','spikes_gt_2x=1/10','bitwise_equal=y','prefill_share='):self.assertIn(text,s)
        self.assertIn('prefill/own solo',table(dict(cases=[dict(case='test',**r)])))
        self.assertIn('metrics=unavailable',monitor_lines('failed',dict(verdict='FAIL'))[0])
    def test_hard_correctness_and_incomplete_control(self):
        d=self.fixture();d['ranks'][1]['noop_correctness']['passed']=False
        del d['ranks'][0]['modes']
        self.assertEqual(grade(d)['verdict'],'FAIL')
        d=self.fixture();d['ranks'][0]['noop_collective_calls']=0
        self.assertEqual(grade(d)['verdict'],'INCONCLUSIVE')
        d=self.fixture();d['ranks'][0]['modes']['overlap']['gpu_completion_gaps_ms'].pop()
        self.assertEqual(grade(d)['verdict'],'INCONCLUSIVE')
        d=self.fixture();d['ranks'][0]['modes']['overlap']['gpu_completion_gaps_ms'][0]=float('nan')
        self.assertEqual(grade(d)['verdict'],'INCONCLUSIVE')
    def test_noop_scope_restored_after_exception(self):
        events=[];original=SimpleNamespace(progress=SimpleNamespace(event=lambda *a,**k:events.append((a,k))))
        lane=SimpleNamespace(reduce=original);x=SimpleNamespace(shape=(1024,2560))
        with self.assertRaises(ValueError):
            with local_prefill_collectives(lane,lambda:23) as c:
                self.assertIs(c(x),x);self.assertEqual(c.collective,1);raise ValueError()
        self.assertIs(lane.reduce,original);self.assertEqual(len(events),2)
    def test_all_six_cases_visible_when_missing(self):
        root=Path(__file__).resolve().parents[2]/'tmp';root.mkdir(exist_ok=True)
        d=Path(tempfile.mkdtemp(prefix='s5-value-',dir=root))
        r=summarize(d);self.assertEqual(len(r['cases']),6);self.assertEqual(r['verdict'],'INCONCLUSIVE')

if __name__=='__main__':unittest.main()
