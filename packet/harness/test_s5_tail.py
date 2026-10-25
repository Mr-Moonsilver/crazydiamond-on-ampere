import copy
from contextlib import nullcontext
import sys
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from s5_terminal_stats import tail_coverage, summarize


def fixture():
    def row(i,a,b,c,guard=False):
        return dict(step=i,submitted_s=a,issued_s=b,observed_s=c,post_decode_guard=guard)
    rows=dict(decode=[row(0,.2,.21,.3),row(1,.5,.51,.6)],
              prefill=[row(0,.1,.11,.4),row(1,.45,.7,.8),row(2,.85,.9,1.,True)])
    proof=dict(version=1,basis='continuous_prefill_backlog',backlog_open_s=0.,
        tail_probe=dict(ticket_step=1,ticket_submitted_s=.45,probe_s=.61,
                        start_complete=True,end_recorded=False,end_complete=None,state='dispatch_pending'))
    return rows,proof


class Tail(unittest.TestCase):
    def test_pending_forward_not_mistaken_for_previous_completed_ticket(self):
        rows,proof=fixture();self.assertTrue(tail_coverage(rows,proof))
        r=summarize(rows,dict(decode=.4,prefill=.9),[300.],False,'overlap',proof)
        self.assertTrue(r['prefill_covers_decode_tail'])
        self.assertTrue(r['prefill_tail_dispatch_pending'])
        self.assertFalse(r['prefill_device_active_at_tail_probe'])
        self.assertEqual(r['window_s'],.4);self.assertEqual(r['gpu_completion_gaps_ms'],[300.])
        # No trimming, no counting the post-decode continuation as throughput.
        self.assertEqual(r['prefill_steps_s'],0.)
        self.assertEqual(r['prefill_steps_s_upper'],5.)

    def test_real_ticket_gap_is_workload_coverage_not_device_occupancy(self):
        rows,proof=fixture()
        rows['prefill'][1].update(issued_s=.46,observed_s=.58)
        proof['tail_probe'].update(end_recorded=True,end_complete=True,state='between_tickets')
        self.assertTrue(tail_coverage(rows,proof))
        r=summarize(rows,dict(decode=.4,prefill=.9),[300.],False,'overlap',proof)
        self.assertFalse(r['prefill_device_active_at_tail_probe'])
        self.assertFalse(r['prefill_tail_dispatch_pending'])

    def test_missing_continuation_bad_order_or_posthoc_work_cannot_pass(self):
        for change in ('no_guard','probe_after_guard','late_backlog','missing_ticket','bad_end','bad_order'):
            rows,proof=fixture()
            if change=='no_guard':rows['prefill'][-1]['post_decode_guard']=False
            if change=='probe_after_guard':proof['tail_probe']['probe_s']=.86
            if change=='late_backlog':proof['backlog_open_s']=.3
            if change=='missing_ticket':rows['prefill'][1]['step']=7
            if change=='bad_end':proof['tail_probe']['end_complete']=True
            if change=='bad_order':rows['prefill'][1]['issued_s']=.9
            self.assertFalse(tail_coverage(rows,proof),change)
        rows,proof=fixture()
        old=summarize(rows,dict(decode=.4,prefill=.9),[300.],False,'overlap')
        self.assertFalse(old['prefill_covers_decode_tail'])
        self.assertNotIn('tail_proof',old) # No retroactive salvage.

    def test_actual_thread_flow_publishes_before_forward_and_drains_guard(self):
        # Execute the real threaded runner. The second eager forward deliberately
        # holds its end-recording until the final decode tail probe is done.
        # GPU kernels/timing are not claimed by this CPU concurrency test.
        from s5_terminal_mode import run
        release=threading.Event();prefills=[]
        class Event:
            def __init__(self,**_):self.t=None
            def record(self):self.t=time.perf_counter()
            def query(self):
                if self.t is None:raise AssertionError('queried an unrecorded event')
                return True
            def elapsed_time(self,other):return (other.t-self.t)*1000
        def progress_event(lane,kind,**kw):
            if lane=='decode' and kind=='step-complete' and kw['step']==2:release.set()
        def forward(trace):
            prefills.append(len(prefills))
            if len(prefills)==2:
                if not release.wait(2.):raise RuntimeError('decoder failed to reach tail')
            time.sleep(.0001)
        dist=SimpleNamespace(barrier=lambda **_:None,all_reduce=lambda *_,**kw:None,ReduceOp=SimpleNamespace(MIN=0))
        cuda=SimpleNamespace(current_device=lambda:0,set_device=lambda _:None,synchronize=lambda:None,
            stream=lambda _:nullcontext(),Event=Event)
        torch=SimpleNamespace(cuda=cuda,distributed=dist,tensor=lambda x,**_:x,int32=0)
        lane=SimpleNamespace(reset=lambda:None,forward=forward,reduce=SimpleNamespace(step=0))
        progress=SimpleNamespace(event=progress_event)
        with patch.dict(sys.modules,{'torch':torch,'torch.distributed':dist}):
            r=run(lane,lane,SimpleNamespace(replay=lambda:time.sleep(.0001)),0,1,0,progress,3,'overlap',False)
        self.assertTrue(r['prefill_covers_decode_tail'])
        self.assertEqual(r['tail_proof']['tail_probe']['ticket_step'],1)
        self.assertTrue(r['prefill_tail_dispatch_pending'])
        self.assertFalse(r['prefill_device_active_at_tail_probe'])
        self.assertEqual(sum(x['post_decode_guard'] for x in r['raw_prefill']),1)
        self.assertGreaterEqual(r['raw_prefill'][-1]['submitted_s'],r['raw_decode'][-1]['observed_s'])


if __name__=='__main__':unittest.main()
