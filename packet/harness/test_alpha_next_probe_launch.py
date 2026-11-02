#!/usr/bin/env python3
"""alpha-next probe launcher (harness/alpha_next_probe_launch.py; g4 engine self-run, HARNESS): every engine self-run probe case
runs through it, and the probes' receipt lines come from a REAL native_measurement.ReceiptWriter's records, read back with the
box grader's parser (receipt_lines), not from a writer kept off. CPU; leaf probe-launch-receipt-sink in
alpha_next_seams_fixture_leaves.json."""
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'patches' / 's5_engine'))
sys.path.insert(0, str(ROOT / 'harness'))
import alpha_next_probe_launch as launch  # noqa: E402
import native_measurement as nm  # noqa: E402
import receipt_lines  # noqa: E402

# One progress flush in two parts (one write_lines call is the parts of one logical line; the sha covers them together).
PARTS = ['flash-s5 progress ' + json.dumps(dict(side='engine', part=i, parts=2, line=0, rows=[]), separators=(',', ':'))
         for i in range(2)]
PROGRESS = PARTS[0]


class Sink(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


class Launcher(unittest.TestCase):
    def setUp(self):
        launch.LINES.clear()
        launch.STATE.update(writer='absent', modules=[], records=0, delivered=0)
        launch._WRITER.clear()
        self.sink = Sink()
        self.loggers = [logging.getLogger('alpha-next-launch-test.engine_driver'),
                        logging.getLogger('alpha-next-launch-test.executor_hooks')]
        for lg in self.loggers:           # one handler on both emit-site loggers, as the progress probe attaches its Capture
            lg.addHandler(self.sink)
        self.addCleanup(lambda: [lg.removeHandler(self.sink) for lg in self.loggers])
        self.addCleanup(launch._WRITER.clear)
        slot = nm._RECEIPT_WRITER        # _install sets this process's writer slot; give the runner its slot back
        self.addCleanup(setattr, nm, '_RECEIPT_WRITER', slot)

    def test_a_real_writer_record_reaches_each_handler_once_as_the_grader_reads_it(self):
        launch._install(nm, 'native_measurement-test-copy')
        writer = launch._WRITER[0]
        self.assertIs(type(writer), nm.ReceiptWriter)
        self.assertEqual(launch.STATE['writer'], 'real-ReceiptWriter-with-sink')
        writer.write_lines(PARTS)
        progress = [line for line in self.sink.lines if line.startswith('flash-s5 progress ')]
        self.assertEqual(progress, PARTS)                  # bare text, identity stripped, once per handler
        self.assertEqual([line for line in self.sink.lines if line.startswith('flash-s5 receipts receipt_path=')],
                         self.sink.lines[:1])                              # the writer's own path record, delivered once
        self.assertTrue(all(receipt_lines.JSON_IDENTITY.match(line[len('flash-s5 progress '):])
                            for line in launch.LINES if line.startswith('flash-s5 progress ')))
        counts, defects = launch._integrity()
        self.assertEqual((counts['missing_seq'], counts['bad_sha'], counts['incomplete'], defects), (0, 0, 0, 0))

    def test_a_framing_defect_is_counted(self):
        launch._install(nm, 'native_measurement-test-copy')
        launch._WRITER[0].write_lines(PARTS)
        index = max(i for i, line in enumerate(launch.LINES) if line.startswith('flash-s5 progress '))
        head, tail = launch.LINES[index].split('"sha":"', 1)
        launch.LINES[index] = head + '"sha":"' + '0' * 16 + tail[16:]          # the record's sha no longer matches its text
        counts, defects = launch._integrity()
        self.assertEqual((counts['bad_sha'], defects), (1, 1))      # the one logical line

    def test_an_overlay_without_the_writer_installs_nothing(self):
        class Parent:             # Step B's native_measurement: no ReceiptWriter
            pass
        launch._install(Parent, 'native_measurement')
        self.assertEqual((launch.STATE['writer'], launch._WRITER), ('absent', []))

    def test_every_probe_case_of_every_container_runs_through_the_launcher(self):
        # A subprocess: importing the self-run tools installs their command-line patches, which must not leak into this runner.
        code = ('import sys, json; sys.argv=["x"]; sys.path.insert(0, %r); import alpha_next_engine_selfrun_b as b; m=b.base; '
                'sets={"progress": (m.held_f.command_lines, m.held_f.CASES), "fused": (m.step_a.command_lines, m.step_a.CASES), '
                '"groups": (m.step_a.command_lines, m.step_b.CASES), "stage": (m.c0.command_lines, m.c0.CASES), '
                '"composed": (m.step_a.command_lines, m.expected_cases())}; '
                'print(json.dumps({k: f(list(c), ()) for k, (f, c) in sets.items()}))' % str(ROOT / 'tools'))
        done = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True, timeout=300, cwd=str(ROOT))
        self.assertEqual(done.returncode, 0, done.stderr[-2000:])
        lines = json.loads(done.stdout.strip().splitlines()[-1])
        counts = {}
        for part, text in lines.items():
            probes = [line for line in text if '_probe.py ' in line]
            counts[part] = len(probes)
            for line in probes:      # exactly one launcher, then a probe file of the bundle's harness
                self.assertEqual(line.count(' python3 -B alpha_next_probe_launch.py e3_'), 1, line)
                self.assertTrue(os.path.isfile(ROOT / 'harness' / line.split('alpha_next_probe_launch.py ')[1].split()[0]))
        # The g4 parts that ran probes (39 failed there) and the composed set: none is left unwrapped.
        self.assertEqual(counts, {'progress': 26, 'fused': 6, 'groups': 7, 'stage': 6, 'composed': 59})

if __name__ == '__main__':
    unittest.main()
