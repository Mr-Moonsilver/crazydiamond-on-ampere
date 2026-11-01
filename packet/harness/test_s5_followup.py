"""Offline q11 verdict regressions and unarmed queue admission. No GPU calls."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from s5_followup import IMAGE, q11, measurement
from s5_segment import PAIRS, plan
from test_s5_segment import fixture

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = json.loads((HERE / 's5_source_manifest.json').read_text())


def write(path, value):
    path.write_text(json.dumps(value) + '\n')


def make_q11(root):
    cfg = json.loads((ROOT / 'reference/model-config.json').read_text())
    for mode in ('prio', 'green24'):
        directory = root / ('s5-20261025T030000Z-' + mode)
        directory.mkdir()
        (directory / 'image.txt').write_text(IMAGE + ' []\n')
        for pair in PAIRS:
            data = fixture(pair)
            data['run_spec'] = dict(plan(cfg), modes=[mode], steps=60, gdn_decode='packed',
                collectives={'decode': 'CustomAllreduce', 'prefill': 'PyNccl'},
                model_config_sha256=SOURCE['model_config_sha256'])
            for rank in data['ranks']:
                rank['source_sha256'] = copy.deepcopy(SOURCE['modules'])
                rank['diagnostic_completed'] = True
                rank['reduction_paths'] = {'decode': ['CustomAllreduce'], 'prefill': ['PyNccl']}
                rank['modes'] = {mode: rank['modes'][mode]}
                rank['modes'][mode].update(raw_decode=[{'observed_s': (i+1)*.01} for i in range(60)],
                                           raw_prefill=[{'observed_s': .1}, {'observed_s': .2}])
            p = directory / ('pair-' + pair.replace(',', '-') + '.json')
            write(p, data)
            for rank in (0, 1):
                write(Path(str(p) + f'.rank{rank}.progress.json'), {'rank': rank, 'phase': 'completed'})
                write(Path(str(p) + f'.rank{rank}.progress.jsonl'), {'rank': rank, 'kind': 'phase', 'phase': 'completed'})
    return root


class FollowupTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix='s5-followup-', dir=ROOT.parent / 'tmp'))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        make_q11(self.root)
        self.pair = self.root / 's5-20261025T030000Z-prio/pair-0-2.json'

    def test_pass_is_liveness_only(self):
        r = q11(self.root)
        self.assertEqual(r['verdict'], 'PASS', r['reasons'])
        self.assertTrue(r['measurement_allowed'])
        self.assertIn('not S5 GO', r['scope'])

    def test_empty_or_duplicate_ranks_never_pass(self):
        original = json.loads(self.pair.read_text())
        for ranks in ([], None, [original['ranks'][0]]*2, [None, None]):
            d = copy.deepcopy(original); d['ranks'] = ranks; write(self.pair, d)
            self.assertEqual(q11(self.root)['verdict'], 'INCONCLUSIVE')

    def test_completion_flag_without_work_never_passes(self):
        d = json.loads(self.pair.read_text())
        d['ranks'][0]['modes']['prio']['raw_decode'] = []
        write(self.pair, d)
        self.assertEqual(q11(self.root)['verdict'], 'INCONCLUSIVE')

    def test_error_after_success_overrides_complete_file(self):
        write(Path(str(self.pair) + '.rank1.error.json'), {'fatal_correctness': True})
        self.assertEqual(q11(self.root)['verdict'], 'FAIL')

    def test_failure_wins_even_with_missing_other_stage(self):
        # Preserve files; rename rather than delete the prior stage.
        (self.root / 's5-20261025T030000Z-green24').rename(self.root / 'retained-stage')
        write(Path(str(self.pair) + '.rank0.error.json'), {'partial_rank': {'finite': False}})
        self.assertEqual(q11(self.root)['verdict'], 'FAIL')

    def test_watchdog_shows_phase_and_lane_issue_order(self):
        base = str(self.pair) + '.rank0'
        write(Path(base + '.progress.json'), {'rank': 0, 'phase': 'prio:measurement', 'timeout_s': 120})
        write(Path(base + '.progress.jsonl'), {'rank': 0, 'kind': 'lane', 'phase': 'prio:measurement',
            'lane': 'prefill', 'stage': 'collective-enter', 'step': 2, 'collective': 18, 'seq': 40})
        write(Path(base + '.error.json'), {'watchdog': True, 'fatal_correctness': False})
        r = q11(self.root)
        self.assertEqual(r['verdict'], 'INCONCLUSIVE')
        p = r['stages']['prio']['pairs']['0,2']['rank_progress'][0]
        self.assertEqual(p['phase'], 'prio:measurement')
        self.assertTrue(p['watchdog'])
        self.assertEqual(p['last_lane_events']['prefill'][0]['collective'], 18)

    def test_ambiguous_stages_never_pass(self):
        shutil.copytree(self.root / 's5-20261025T030000Z-prio', self.root / 's5-other-prio')
        self.assertEqual(q11(self.root)['verdict'], 'INCONCLUSIVE')

    def test_changed_source_wrong_gpu_or_workload_never_pass(self):
        original = json.loads(self.pair.read_text())
        for edit in ('source', 'gpu', 'workload', 'sm'):
            d = copy.deepcopy(original)
            if edit == 'source': d['ranks'][0]['source_sha256'] = {}
            elif edit == 'gpu': d['ranks'][1]['pci_bus_id'] = d['ranks'][0]['pci_bus_id']
            elif edit == 'workload': d['run_spec']['chunk'] = 512
            else: d['ranks'][0]['diagnostic_completed'] = 'true'
            write(self.pair, d)
            self.assertEqual(q11(self.root)['verdict'], 'INCONCLUSIVE')

    def test_malformed_error_and_missing_progress_are_inconclusive(self):
        Path(str(self.pair) + '.rank0.error.json').write_text('{')
        Path(str(self.pair) + '.rank0.progress.json').write_text('{}')
        self.assertEqual(q11(self.root)['verdict'], 'INCONCLUSIVE')

    def test_measurement_keeps_existing_go_gate_and_units(self):
        out = self.root / 'measurement'; out.mkdir()
        for pair in PAIRS:
            p = out / ('pair-' + pair.replace(',', '-') + '.json')
            write(p, fixture(pair))
            for rank in (0, 1):
                write(Path(str(p) + f'.rank{rank}.progress.json'), {'rank': rank, 'phase': 'completed'})
                write(Path(str(p) + f'.rank{rank}.progress.jsonl'), {'rank': rank, 'kind': 'phase', 'phase': 'completed'})
        r = measurement(out)
        self.assertEqual(r['verdict'], 'GO')
        self.assertIn('NOT full-model tokens/s', r['scope'])
        self.assertAlmostEqual(r['measurements']['0,2']['ranks'][0]['modes']['green24']['decode_steps_s_vs_serial'], 100/30)
        write(out / 'pair-0-2.json.rank0.error.json', {'fatal_correctness': True})
        self.assertEqual(measurement(out)['verdict'], 'FAIL')

    def test_cli_missing_root_is_inconclusive(self):
        r = subprocess.run([sys.executable, str(HERE / 's5_followup.py'), 'q11', str(self.root / 'absent')],
                           text=True, capture_output=True)
        self.assertEqual(r.returncode, 2, r.stderr)
        self.assertFalse(json.loads(r.stdout)['measurement_allowed'])

    def test_nonfinite_json_is_inconclusive_not_crash(self):
        original = self.pair.read_text()
        for literal in ('NaN', '1e999'):
            self.pair.write_text(original[:-2] + ', "bad": '+literal+'}\n')
            r = subprocess.run([sys.executable, str(HERE/'s5_followup.py'), 'q11', str(self.root)],
                               text=True, capture_output=True)
            self.assertEqual(r.returncode, 2, r.stderr)
            self.assertEqual(json.loads(r.stdout)['verdict'], 'INCONCLUSIVE')

    def test_queue_refuses_before_any_gpu_or_engine_command(self):
        # Use real grader; fake Docker/systemctl/nvidia-smi would leave evidence.
        h, overlay, binaries = (self.root / p for p in ('h', 'overlay', 'bin'))
        for d in (h, overlay, binaries): d.mkdir()
        for name in ('s5_followup.py', 's5_segment.py', 's5_source_manifest.json', 'qlib.sh'):
            shutil.copyfile(HERE / name, h / name)
        (overlay / 'sentinel').write_text('overlay fixture')
        for d in (h, overlay):
            (d / 'SHA256SUMS').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name+'\n'
                                                for p in sorted(d.iterdir())))
        for name in ('docker', 'systemctl', 'nvidia-smi'):
            p = binaries / name
            p.write_text('#!/bin/sh\nprintf called > "$S5_TEST_MARKER"\nexit 99\n'); p.chmod(0o700)
        write(Path(str(self.pair) + '.rank0.error.json'), {'watchdog': True})
        env = dict(os.environ, H=str(h), Q11=str(self.root), Q=str(self.root / 'queue'), S5_OVERLAY=str(overlay),
                   S5_MEASURE_BOUNDED='1', PATH=str(binaries)+':'+os.environ['PATH'], S5_TEST_MARKER=str(self.root/'called'))
        r = subprocess.run(['bash', str(HERE/'runs/q_s5_measure.draft.sh')], env=env, text=True, capture_output=True)
        self.assertEqual(r.returncode, 2, r.stdout+r.stderr)
        self.assertFalse((self.root/'called').exists())
        self.assertIn('blocked-by-q11', r.stdout)


if __name__ == '__main__':
    unittest.main()
