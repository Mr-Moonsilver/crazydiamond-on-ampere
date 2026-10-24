#!/usr/bin/env python3
"""CPU-only clock rules, native build contract, ordering and post-clock hurdles."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from s5_clock import (Inconclusive, WINDOWS_MS, analyze_clock, build_command, build_helper,
                      ordering_window, sm_restriction)
from s5_segment import grade_pair, MODES, plan
from test_s5_segment import fixture

HERE = Path(__file__).resolve().parent


def measurements(quantum=1000, ratio=1.):
    samples = []
    for target in WINDOWS_MS:
        for i in range(3):
            first = 10**12 + i * 10**9
            samples.append(dict(target_ms=target, cuda_event_ms=float(target),
                                raw_ticks=[first, first + int(target*1e6*ratio)]))
    resolution = dict(min_nonzero_ticks=quantum, nonzero_deltas=100,
                      backwards=0, compiled_arch=860, raw_back_to_back_ticks=[10**12,10**12+quantum])
    return samples, resolution


class ClockRules(unittest.TestCase):
    def test_fine_clock(self):
        result = analyze_clock(*measurements(32))
        self.assertTrue(result['valid'])
        self.assertFalse(result['coarse'])
        self.assertEqual(result['median_ratio'], 1)
        self.assertEqual(result['ordering_uncertainty_ns'], 64)

    def test_one_startup_outlier_does_not_spoil_three_window_medians(self):
        rows, g = measurements()
        rows[0]['cuda_event_ms'] = 100  # lazy-load/JIT-like event window
        result = analyze_clock(rows, g)
        self.assertEqual(result['median_ratio'], 1)
        self.assertTrue(result['valid'])
        self.assertEqual(result['samples'][0]['ratio'], .1)

    def test_stable_wrong_rate_reports_raw_evidence(self):
        rows, g = measurements(ratio=.03125)
        with self.assertRaises(Inconclusive) as cm:
            analyze_clock(rows, g)
        self.assertIn('globaltimer scale disagrees', str(cm.exception))
        for key in ('globaltimer_ms', 'cuda_event_ms', 'raw_ticks'):
            self.assertIn(key, str(cm.exception))
            self.assertIn(key, cm.exception.evidence['samples'][0])
        self.assertFalse(cm.exception.evidence['valid'])

    def test_bad_short_window_is_not_hidden_by_six_long_windows(self):
        rows, g = measurements()
        for row in rows[:3]:
            row['raw_ticks'][1] = row['raw_ticks'][0] + 3_000_000
        with self.assertRaisesRegex(Inconclusive, '10 ms globaltimer scale'):
            analyze_clock(rows, g)

    def test_coarse_clocks_keep_events_and_explicit_ordering_resolution(self):
        for quantum in (1000, 1_000_000):
            result = analyze_clock(*measurements(quantum))
            self.assertTrue(result['coarse'])
            self.assertEqual(result['duration_source'], 'same-stream CUDA events')
            self.assertEqual(result['observed_granularity_ns'], quantum)
        rows, g = measurements(1_000_000)
        for row in rows[:3]:
            row['raw_ticks'][1] += 2_000_000  # two quantized endpoints on 10ms
        self.assertTrue(analyze_clock(rows, g)['valid'])

    def test_stopped_backwards_or_too_coarse_clock_is_not_silently_accepted(self):
        for updates in ({'min_nonzero_ticks':0}, {'backwards':1}, {'min_nonzero_ticks':2_000_000},
                        {'nonzero_deltas':0}):
            rows, g = measurements(); g.update(updates)
            with self.assertRaises(Inconclusive): analyze_clock(rows, g)

    def test_missing_window_or_unexercised_sleep_target(self):
        rows, g = measurements()
        with self.assertRaisesRegex(Inconclusive, 'missing 200 ms'):
            analyze_clock(rows[:6], g)
        rows, g = measurements()
        for row in rows:
            row['cuda_event_ms'] /= 10
            row['raw_ticks'][1] = row['raw_ticks'][0] + int(row['cuda_event_ms']*1e6)
        with self.assertRaisesRegex(Inconclusive, 'was not exercised'):
            analyze_clock(rows, g)

    def test_invalid_event_or_ticks(self):
        for value in (None, float('nan'), float('inf'), 0., -1., True):
            rows, g = measurements(); rows[0]['cuda_event_ms'] = value
            with self.assertRaises(Inconclusive) as cm: analyze_clock(rows, g)
            json.dumps(cm.exception.evidence, allow_nan=False)  # diagnostics must survive invalid numbers
        for ticks in (None, [], [10,10], [20,10], [1.5,2.5]):
            rows, g = measurements(); rows[0]['raw_ticks'] = ticks
            with self.assertRaises(Inconclusive): analyze_clock(rows, g)


class BuildRules(unittest.TestCase):
    def test_native_sm86_no_foreign_ptx_and_explicit_runtime(self):
        command = build_command('src', 'out', (8,6))
        self.assertIn('--generate-code=arch=compute_86,code=sm_86', command)
        self.assertIn('--cudart=shared', command)
        self.assertIn('--cudart=static', build_command('src', 'out', (8,6), 'static'))
        with self.assertRaises(Inconclusive): build_command('src','out',(7,5))

    def test_actual_builder_with_mock_compiler_and_no_gpu(self):
        root = Path(tempfile.mkdtemp(prefix='s5-clock-', dir=HERE.parent.parent/'tmp'))
        library = root/'fake.so'
        seen = []
        def compiler(command, **kwargs):
            seen.append(command)
            if '-o' in command:
                Path(command[-1]).write_bytes(b'mocked-shared-object')
            return SimpleNamespace(returncode=0, stdout='mock nvcc', stderr='')
        with patch('s5_clock.subprocess.run', compiler), patch('s5_clock.shutil.which', return_value=None):
            build = build_helper(library, (8,6))
            self.assertEqual(build['arch'], 'sm_86')
            self.assertEqual(build['cudart'], 'shared')
            self.assertFalse(build['ptx_fallback'])
            self.assertEqual(build['command'], seen[-1])
            with self.assertRaises(Inconclusive): build_helper(library,(8,6))
        self.assertEqual(library.read_bytes(), b'mocked-shared-object')

    def test_probe_plan_needs_neither_torch_nor_gpu(self):
        p = subprocess.run([sys.executable,str(HERE/'s5_clock_probe.py'),'--plan'],capture_output=True,text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn('--cudart=shared', json.loads(p.stdout)['command'])


class FollowingHurdles(unittest.TestCase):
    def test_short_probe_needs_containment_not_complete_sm_coverage(self):
        membership = [set(range(24)),set(range(24,82))]
        self.assertTrue(sm_restriction(membership,[24,58],{2,3},{25,26},82))
        for a,b in ((set(),{25}), ({-1},{25}), ({30},{25}), ({2},{3})):
            self.assertFalse(sm_restriction(membership,[24,58],a,b,82))
        self.assertFalse(sm_restriction([set(range(23)),set(range(24,82))],[24,58],{1},{25},82))

    def test_sparse_sm_ids_are_not_wrongly_rejected(self):
        self.assertTrue(sm_restriction([{0,2},{4,6}],[2,2],{2},{6},4))

    def test_coarse_window_reports_lower_and_upper_counts(self):
        dec = [{'raw_ticks':[100,200]}]
        pre = [{'raw_ticks':[100,120]}, {'raw_ticks':[125,175]}, {'raw_ticks':[180,200]}]
        _,_,inside,possible,delta = ordering_window(dec,pre,2)
        self.assertEqual((len(inside),len(possible),delta),(1,3,0))
        _,_,inside,possible,_ = ordering_window([],pre,2)
        self.assertEqual(len(inside),len(possible))
        for rows in ([{'raw_ticks':[200,100]}], [{'raw_ticks':[100,200]},{'raw_ticks':[150,220]}]):
            with self.assertRaises(Inconclusive): ordering_window(rows,[],2)

    def test_threshold_straddle_is_inconclusive_not_no_go(self):
        record = fixture()
        for r in record['ranks']:
            for mode in ('green24','green34','green44'):
                r['modes'][mode].update(prefill_steps_s=39,prefill_steps_s_upper=41)
        self.assertEqual(grade_pair(record)['verdict'],'INCONCLUSIVE')

    def test_definite_pass_with_coarse_clock_is_allowed_but_tail_unknown_is_not(self):
        record = fixture()
        for r in record['ranks']:
            r['clock_check'].update(coarse=True, observed_granularity_ns=1_000_000)
        self.assertEqual(grade_pair(record)['verdict'],'GO')
        record['ranks'][0]['modes']['green24']['prefill_tail_order_ambiguous'] = True
        self.assertEqual(grade_pair(record)['verdict'],'INCONCLUSIVE')

    def test_old_globaltimer_duration_or_missing_clock_evidence_is_inconclusive(self):
        record = fixture(); record['ranks'][0]['clock_check'] = {}
        self.assertEqual(grade_pair(record)['verdict'],'INCONCLUSIVE')
        record = fixture(); record['ranks'][0]['modes']['decode']['duration_source'] = 'globaltimer'
        self.assertEqual(grade_pair(record)['verdict'],'INCONCLUSIVE')

    def test_partition_calibration_follows_production_24_24(self):
        config = json.loads((HERE.parent/'reference/model-config.json').read_text())
        p = plan(config)
        self.assertEqual(p['stage_counts'],{'0,2':{'gdn':18,'qsa':6},'1,3':{'gdn':18,'qsa':6}})
        self.assertEqual(plan(config, partition=(26,22))['stage_counts']['0,2']['gdn'],20)

class FakeCuda:
    """Execute captured callbacks on replay; no torch import, native code or device."""
    def __init__(self, drop_stamps=False):
        self.now = 10**12
        self.capture = None
        self.drop_stamps = drop_stamps
        self.cold = True

    def enqueue(self, fn):
        if self.capture is None: fn()
        else: self.capture.ops.append(fn)

    def Stream(self):
        return SimpleNamespace(cuda_stream=1, synchronize=lambda:None)

    def stream(self, stream):
        from contextlib import nullcontext
        return nullcontext()

    def synchronize(self): pass

    def _sleep(self, cycles):
        self.enqueue(lambda:setattr(self,'now',self.now+int(cycles/2)))  # 2 GHz

    def Event(self, **kwargs):
        cuda = self
        class Event:
            def record(self): cuda.enqueue(lambda:setattr(self,'t',cuda.now))
            def elapsed_time(self, other): return (other.t-self.t)/1e6
            def synchronize(self): pass
        return Event()

    def CUDAGraph(self):
        class Graph:
            def __init__(self): self.ops=[]
            def replay(self):
                for fn in self.ops: fn()
            def reset(self): self.ops=[]
        return Graph()

    def graph(self, graph, **kwargs):
        from contextlib import contextmanager
        @contextmanager
        def capture():
            self.capture=graph
            try: yield
            finally: self.capture=None
        return capture()

    def tensor(self, n, **kwargs):
        cuda=self
        class Tensor:
            def __init__(self, data, offset=0): self.data,self.offset=data,offset
            def data_ptr(self): return self
            def __getitem__(self, key): return Tensor(self.data,self.offset+key.start)
            def fill_(self, value):
                def fill(): self.data[self.offset:]=[value]*(len(self.data)-self.offset)
                cuda.enqueue(fill)
            def cpu(self): return self
            def tolist(self): return self.data[self.offset:]
        return Tensor([0]*n)

    def stamp(self, ptr, stream):
        def write():
            if self.cold:
                self.now += 100_000_000  # artificial first-use module-load delay
                self.cold=False
            ptr.data[ptr.offset]=self.now
        if not (self.drop_stamps and self.capture is not None): self.enqueue(write)
        return 0

    def samples(self, ptr, n, cycles, stream):
        def write():
            first=self.now; self.now+=max(128_000,cycles//2)
            ptr.data[:]=[first,self.now,1000,1000,0,100,0,101,860]+[first+i*1000 for i in range(n)]
        self.enqueue(write)
        return 0


class ActualValidationFlow(unittest.TestCase):
    def clock(self, drop=False):
        from s5_clock import DeviceClock
        cuda=FakeCuda(drop)
        clock=DeviceClock.__new__(DeviceClock)
        clock.torch=SimpleNamespace(cuda=cuda,empty=cuda.tensor,int64='int64')
        clock.lib=SimpleNamespace(s5_stamp=cuda.stamp,s5_samples=cuda.samples)
        clock.runtime={'mock':True}
        return clock

    def test_cold_launch_is_outside_every_timed_window(self):
        result=self.clock().validate()
        self.assertEqual(len(result['samples']),9)
        self.assertEqual(result['median_ratio'],1.)
        self.assertEqual({r['cuda_event_ms'] for r in result['samples']},{10.,50.,200.})
        self.assertTrue(result['graph_replay_validated'])

    def test_stamps_missing_from_graph_cannot_reuse_warmup_results(self):
        with self.assertRaises(Inconclusive) as cm:
            self.clock(True).validate()
        self.assertEqual(cm.exception.evidence['samples'][0]['raw_ticks'],[-1,-1])
        json.dumps(cm.exception.evidence,allow_nan=False)


if __name__ == '__main__':
    unittest.main(verbosity=2)
