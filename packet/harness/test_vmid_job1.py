import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('job1', HERE / 'vmid_job1.py')
job = importlib.util.module_from_spec(spec)
spec.loader.exec_module(job)


def complete():
    required = {'kernel': 2, 'memcpy': 0, 'memset': 1}
    slot = {'required': required, 'kernel_executions': 6, 'copy_executions': 0,
            'memset_executions': 3, 'node_execution_bounds': {
                'kernel': {'min': 3, 'max': 3}, 'memcpy': {'min': 0, 'max': 0},
                'memset': {'min': 3, 'max': 3}}}
    return {'output_exact': True, 'serialized_stable': True, 'serialized_bytes': 400,
            'cleanup_ok': True, 'replays': 3,
            'static': {'complete': True, 'graph_slots': [{'required': required}]},
            'window': {'complete': True, 'graph_slots': [slot], 'flags': {}, 'reasons': [],
                       'expected_context': {'is_green': True, 'sm_matching': 1, 'records': 1}}}


class GradeTests(unittest.TestCase):
    def test_complete(self):
        self.assertEqual(job.grade(complete())['verdict'], 'PASS')

    def test_missing_records_never_pass(self):
        r = complete()
        r['window']['graph_slots'][0]['kernel_executions'] = 5
        self.assertEqual(job.grade(r)['verdict'], 'INCONCLUSIVE')

    def test_duplicate_cannot_replace_missing_node(self):
        r = complete()
        r['window']['graph_slots'][0]['node_execution_bounds']['kernel'] = {'min': 2, 'max': 4}
        self.assertEqual(job.grade(r)['verdict'], 'INCONCLUSIVE')

    def test_wrong_context_wins_over_missing(self):
        r = complete()
        r['window']['complete'] = False
        r['window']['counts'] = {'known_wrong_context_records': 1}
        self.assertEqual(job.grade(r)['verdict'], 'FAIL')

    def test_bit_mismatch_fails(self):
        for field in ('output_exact', 'serialized_stable', 'cleanup_ok'):
            r = complete()
            r[field] = False
            self.assertEqual(job.grade(r)['verdict'], 'FAIL')

    def test_unknown_is_not_success(self):
        self.assertEqual(job.grade({})['verdict'], 'INCONCLUSIVE')
        r = complete()
        r['window']['flags']['drop_seen'] = True
        self.assertEqual(job.grade(r)['verdict'], 'INCONCLUSIVE')

    def test_no_kernels_cannot_pass(self):
        r = complete()
        r['static']['graph_slots'][0]['required']['kernel'] = 0
        self.assertNotEqual(job.grade(r)['verdict'], 'PASS')


class ResourceTests(unittest.TestCase):
    def test_apps(self):
        rows = job.parse_apps('GPU-x, 12, python3, 320\nGPU-x, 13, UnrealEditor, 2700\n')
        self.assertEqual([x['used_mib'] for x in rows], [320, 2700])
        with self.assertRaises(ValueError):
            job.parse_apps('GPU-x, 12, python3, N/A')

    def test_shared_usage_does_not_stop_foreign_process(self):
        self.run_monitor(300, False)

    def test_own_usage_stops_only_owned_container(self):
        self.run_monitor(2900, True)

    def run_monitor(self, own, stopped):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            (out / 'monitor.stop').touch()
            calls = []
            def command(argv):
                calls.append(argv)
                if '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory' in argv:
                    return 0, f'GPU-x, 12, python3, {own}\nGPU-x, 13, UnrealEditor, 2700\n'
                if 'top' in argv:
                    return 0, 'PID\n12\n'
                return 0, ''
            with patch.object(job, 'command', command):
                job.monitor(out)
            m = json.loads((out / 'monitor.json').read_text())
            self.assertEqual(m['own_peak_mib'], own)
            self.assertEqual(m['stop_triggered'], stopped)
            stops = [c for c in calls if 'stop' in c]
            self.assertEqual(len(stops), int(stopped))
            if stops:
                self.assertEqual(stops[0], ['sudo', '-n', 'docker', 'stop', '-t', '5', 'flashnext-job1'])

    def test_final_requires_samples_and_retains_failure(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            job.write(out / 'probe.json', complete())
            self.assertEqual(job.finalize(out, 0), 2)
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            job.write(out / 'probe.json', complete())
            job.write(out / 'monitor.json', {'own_samples': 2, 'errors': 0, 'own_peak_mib': 3100})
            self.assertEqual(job.finalize(out, 0), 1)

    def test_final_rejects_exit_receipt_disagreement(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            job.write(out / 'probe.json', complete())
            job.write(out / 'monitor.json', {'own_samples': 2, 'errors': 0, 'own_peak_mib': 500})
            self.assertEqual(job.finalize(out, 2), 2)

    def test_queue_and_backstop(self):
        queue = HERE / 'runs/vmid_job1.sh'
        subprocess.run(['bash', '-n', str(queue)], check=True)
        text = queue.read_text()
        self.assertIn('--rm --gpus all --name flashnext-job1', text)
        self.assertIn('--memory 4g --memory-swap 4g', text)
        self.assertIn('flock -n 9', text)
        self.assertNotIn('systemctl', text)
        self.assertNotIn('docker rm', text)
        probe = (HERE / 'vmid_job1.py').read_text()
        self.assertLess(probe.index('torch.cuda.set_per_process_memory_fraction(0.125'),
                        probe.index('torch.cuda.device_count()'))
        self.assertIn("CUDAGraph(keep_graph=True)", probe)
        self.assertLess(probe.index('graph.instantiate()'), probe.index('lib.s5ro_begin_window('))
        self.assertNotIn('import vllm', probe)


class ProbeFlowTests(unittest.TestCase):
    def test_real_entrypoint_orders_capture_late_enable_and_cleanup(self):
        events = []
        def event(name, result=None):
            def call(*a, **kw):
                events.append(name)
                return result
            return call
        class Scope:
            def __enter__(self):
                events.append('enter')
            def __exit__(self, *a):
                events.append('exit')
        class Tensor:
            def zero_(self): pass
            def add_(self, x): pass
            def __eq__(self, x): return True
        graph = NS(instantiate=event('instantiate'), raw_cuda_graph=lambda: 100,
                   replay=event('replay'), reset=event('graph_reset'))
        cuda = NS(set_per_process_memory_fraction=event('fraction'), device_count=lambda: 1,
                  get_device_properties=lambda _: NS(name='RTX 3090', total_memory=24*1024**3, multi_processor_count=82),
                  get_device_capability=lambda _: (8, 6), synchronize=event('synchronize'),
                  stream=lambda _: Scope(), CUDAGraph=lambda **kw: graph,
                  graph=lambda *a, **kw: Scope(), max_memory_allocated=lambda _: 32768,
                  max_memory_reserved=lambda _: 2*1024**2)
        torch = NS(cuda=cuda, __version__='fake', version=NS(cuda='13.0'), float32='float32',
                   zeros=lambda *a, **kw: Tensor(), zeros_like=lambda _: Tensor(),
                   all=lambda _: NS(item=lambda: True))
        split = NS(decode_sms=44, prefill_sms=38, total_sms=82, decode_context_id=9,
                   decode_stream=NS(synchronize=event('stream_drain')), close=event('split_close'))
        def begin(token, generation, out):
            events.append('begin_window')
            out._obj.value = 123
            return 0
        lib = NS(s5ro_bind_context=lambda *a: 0, s5ro_bind_graph=lambda *a: 0,
                 s5ro_static='static', s5ro_begin_window=begin, s5ro_end_window='window')
        def report(fn, *a, **kw):
            events.append('report_' + fn)
            return complete()[fn]
        binding = NS(_lib=lib, identity=lambda: {'fake': True})
        session = NS(_token=1, _native_reserved=1, start=event('observer_start'),
                     _report_call=report, close=event('observer_close'), binding=binding)
        ro = NS(ReplayBinding=lambda *a, **kw: binding, ObserverLimits=lambda **kw: None,
                ObserverSession=lambda *a: session, _validate_static=lambda *a: None,
                _validate_window=lambda *a: None)
        gc = NS(DriverBinding=lambda *a: None, open_split=event('split_create', split))
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            job.write(root / 'q17-helper-build.json', {'image': job.IMAGE, 'cupti_library_sha256': job.CUPTI,
                'files': {n: job.CUPTI for n in ('flash_s5_green.cu', 'flash_s5_replay_observer.cpp',
                                               'flash_s5_graph_bytes.cpp', 'flash_s5_replay_observer.so')}})
            with patch.dict('sys.modules', {'torch': torch, 'replay_observer': ro, 'green_context': gc}), \
                 patch.object(job, 'sha', return_value=job.CUPTI), \
                 patch.object(job, 'graph_bytes', side_effect=event('serialize', b'private')), \
                 patch.object(job.time, 'sleep'):
                self.assertEqual(job.probe(root, root, root), 0)
            r = json.loads((root / 'probe.json').read_text())
            self.assertNotIn('private', json.dumps(r))
        for first, second in (('fraction', 'observer_start'), ('observer_start', 'split_create'),
                              ('instantiate', 'begin_window'), ('begin_window', 'replay'),
                              ('report_window', 'observer_close'), ('observer_close', 'graph_reset'),
                              ('graph_reset', 'split_close')):
            self.assertLess(events.index(first), events.index(second))
        self.assertEqual(events.count('replay'), 3)


if __name__ == '__main__':
    unittest.main()
