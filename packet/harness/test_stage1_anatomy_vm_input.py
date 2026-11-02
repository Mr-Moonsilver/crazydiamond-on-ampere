#!/usr/bin/env python3
"""The VM probe's stage-1 helpers on the CPU rig: decode RPCs and releases through the generated WorkerProc reply path,
the reply read back, the probe-only final flush, and the summary that goes into probe.json. Also the probe patch
anchors on the real B5 probe text when the B5 input is on this hub."""
from contextlib import ExitStack
import json
import lzma
import os
from pathlib import Path
import sys
import tempfile
import time
import types
from types import SimpleNamespace
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'tools')]
import test_stage1_handler_anatomy as rigs  # noqa: E402
import stage1_anatomy_vm_input as vmi  # noqa: E402
import stage1_anatomy_vm_grade as vm_grade  # noqa: E402

eh, ha, nm = rigs.eh, rigs.ha, rigs.nm


class ProbeHelpers(unittest.TestCase):
    def run_helpers(self, async_scheduling, record=None):
        marks = []
        with ExitStack() as stack:
            # Rank 2 (stage 1, the output rank's own stage): execute_model returns None, sample_tokens replies.
            rig = rigs.Rig(stack, 2, costs=rigs.Fast, async_scheduling=async_scheduling)
            executor = types.ModuleType('vllm.v1.executor.multiproc_executor')
            executor.WorkerProc = rigs.proc_class(eh)
            package = types.ModuleType('vllm.v1.flash_e3')
            package.__path__ = []
            package.executor_hooks, package.native_measurement, package.handler_anatomy = eh, nm, ha
            stack.enter_context(mock.patch.dict(sys.modules, {
                'vllm.v1': types.ModuleType('vllm.v1'), 'vllm.v1.executor': types.ModuleType('vllm.v1.executor'),
                'vllm.v1.executor.multiproc_executor': executor, 'vllm.v1.flash_e3': package}))
            scope = dict(mark=lambda step, **fields: marks.append((step, fields)), SimpleNamespace=SimpleNamespace,
                         time=time, json=json, WAIT_SECONDS=5)
            exec(compile(vmi.PROBE_HELPERS, 'probe-helpers', 'exec'), scope)
            cfg = SimpleNamespace(scheduler_config=SimpleNamespace(async_scheduling=async_scheduling))
            proc = scope['stage1_anatomy_proc'](rig.worker, cfg)
            proc.rank = 2
            results = []
            for _ in range(3):
                results.append(scope['stage1_rpc']('execute_model', ('so',), output_rank=2))
                results.append(scope['stage1_rpc']('sample_tokens', (None,), output_rank=2))
            rig.torch.device.advance()
            scope['stage1_release'](SimpleNamespace(kind='release'))
            scope['stage1_rpc']('execute_model', ('so',), output_rank=2)
            scope['stage1_final_flush']()
            with tempfile.TemporaryDirectory() as out:
                out = record or out
                summary = scope['stage1_anatomy_summary'](Path(out))
                with lzma.open(os.path.join(out, 'stage1-anatomy-lines.jsonl.xz'), 'rt') as handle:
                    archived = [json.loads(line) for line in handle]
            return summary, results, marks, archived

    def test_async(self):
        summary, results, marks, archived = self.run_helpers(True)
        self.assertEqual(marks[0][0], 'stage1-anatomy-proc')
        self.assertTrue(marks[0][1]['async_scheduling'])
        self.assertEqual(summary['rpcs'], 8)
        self.assertEqual(summary['releases'], 1)
        self.assertEqual(summary['parse_errors'], [])
        self.assertEqual(summary['incomplete_flushes'], [])
        self.assertEqual(summary['flushes'], 2)                 # the release's, then the probe-only final flush
        self.assertGreater(summary['progress_lines'], 0)
        self.assertLess(summary['max_line_bytes'], ha.LINE_BYTES)
        self.assertEqual(summary['kinds'], {'e': 4, 's': 3})    # the release RPC is not a stock row
        self.assertEqual(summary['rows_with_device_forward'], summary['execute_rows'] - 1)  # the last: unresolved
        self.assertGreater(summary['output_rows_async_thread'], 0)
        self.assertTrue(summary['recorder_installed'])
        self.assertIn('forward', summary['seam_rows'])
        self.assertEqual(len(archived), summary['handler_lines'] + summary['progress_lines'])
        self.assertTrue(all(r is not None for r in results[1::2]))      # sample_tokens replies carry output

    def test_sync(self):
        summary, _, marks, _ = self.run_helpers(False)
        self.assertFalse(marks[0][1]['async_scheduling'])
        self.assertEqual(summary['output_rows_async_thread'], 0)
        self.assertGreater(summary['output_rows_main_thread'], 0)


class Grade(unittest.TestCase):
    """The VM grade on a record made of the rig's own summary and archive."""

    def record(self, root, wrapper='PASS', **change):
        summary = ProbeHelpers().run_helpers(True, record=root)[0]
        summary.update(change)
        with open(os.path.join(root, 'probe.json'), 'w') as handle:
            json.dump(dict(verdict='PASS', stage1_anatomy=summary), handle, default=str)
        with open(os.path.join(root, 'result.json'), 'w') as handle:
            json.dump(dict(verdict=wrapper), handle)

    def test_pass_and_readback(self):
        with tempfile.TemporaryDirectory() as root:
            self.record(root)
            checks, notes, readback = vm_grade.grade(root)
        self.assertTrue(all(checks.values()), checks)
        self.assertEqual(readback['torn'], 0)
        self.assertGreater(readback['rows']['main'], 0)

    def test_fail_and_inc(self):
        with tempfile.TemporaryDirectory() as root:
            self.record(root, rows_with_device_forward=0)
            checks = vm_grade.grade(root)[0]
        self.assertIs(checks['events_resolve'], False)
        with tempfile.TemporaryDirectory() as root:
            checks = vm_grade.grade(root)[0]
        self.assertIsNone(checks['wrapper_pass'])


class ProbePatch(unittest.TestCase):
    """Every anchor of patch_anatomy holds once on the real B5 probe after Step A's stepped patch."""

    def test_anchors(self):
        probe = vmi.sa.BASE / 'proof/e3_native29_overlap_probe.py'
        if not probe.is_file():
            raise unittest.SkipTest('B5 input not on this hub')
        with tempfile.TemporaryDirectory() as out:
            out = Path(out)
            (out / 'proof').mkdir()
            (out / 'proof/e3_native29_overlap_probe.py').write_text(probe.read_text())
            vmi.sa.patch_probe(out, False)
            vmi.patch_anatomy(out)
            text = (out / 'proof/e3_native29_overlap_probe.py').read_text()
        compile(text, 'probe', 'exec')
        self.assertNotIn('worker.execute_model(output)', text)
        self.assertNotIn('bridge.release(request)', text)
        for needle in ("os.environ['FLASH_S5_HANDLER_ANATOMY'] = '1'", "os.environ['FLASH_S5_NATIVE_PROGRESS'] = 'burst'",
                       '        stage1_anatomy_proc(worker, cfg)\n', "stage1_rpc('execute_model', (output,))",
                       "stage1_rpc('sample_tokens', (None,))", 'ack = stage1_release(request)', '        stage1_final_flush()\n',
                       'stage1_anatomy=stage1_anatomy_summary(args.out.parent)'):
            self.assertEqual(text.count(needle), 1, needle)


if __name__ == '__main__':
    unittest.main()
