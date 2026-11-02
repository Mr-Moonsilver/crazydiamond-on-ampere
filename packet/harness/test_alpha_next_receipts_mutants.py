#!/usr/bin/env python3
"""Alpha-next PLAN 2 mutants: each writer mutant must make test_alpha_next_receipts fail; the unmutated copy must pass.

Each mutant is exact text replacements of patches/s5_engine/native_measurement.py (refused unless each old text occurs
exactly once), written to a temp dir that test_alpha_next_receipts puts first on sys.path (ALPHA_NEXT_RECEIPTS_ENGINE,
leaf alpha-next-receipts-mutant-engine-dir). CPU only. `ALPHA_NEXT_RECEIPTS_EVIDENCE=<dir>` keeps each run's output
and a JSON table there.
"""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'patches' / 's5_engine' / 'native_measurement.py'
SUITE = ROOT / 'harness' / 'test_alpha_next_receipts.py'

MUTANTS = {
    'two-writes-per-record': [(
        "            written = self._write(self.fd, data)\n",
        "            half = len(data) // 2\n"
        "            written = self._write(self.fd, data[:half])\n"
        "            written += self._write(self.fd, data[half:])\n")],
    'budget-above-pipe-buf': [
        ("RECEIPT_RECORD_BYTES = 3968\n", "RECEIPT_RECORD_BYTES = 8192\n"),
        ("PROGRESS_LINE_BYTES = 3968\n", "PROGRESS_LINE_BYTES = 8192\n")],
    'part-budget-left-at-12000': [
        ("PROGRESS_LINE_BYTES = 3968\n", "PROGRESS_LINE_BYTES = 12000\n")],
    'seq-not-advanced-on-kv-records': [(
        "        seq = self._next()\n        data = ('%s pid=%d seq=%d sha=%s\\n'",
        "        seq = self.seq\n        data = ('%s pid=%d seq=%d sha=%s\\n'")],
    'sha-over-the-part': [(
        "                            % (text[:cut], self.pid, seq, sha, text[cut:])).encode()",
        "                            % (text[:cut], self.pid, seq, receipt_sha([text]), text[cut:])).encode()")],
    'short-write-tolerated-silently': [(
        "        if written != len(data):\n            self.short_writes += 1\n            return False\n",
        "")],
}


def mutate(source, edits):
    for old, new in edits:
        if source.count(old) != 1:
            raise AssertionError('mutant text does not occur exactly once: %r' % old[:60])
        source = source.replace(old, new)
    return source


def run(name, edits):
    with tempfile.TemporaryDirectory(prefix='receipt-mutant-') as d:
        (Path(d) / 'native_measurement.py').write_text(mutate(SOURCE.read_text(), edits))
        process = subprocess.run([sys.executable, str(SUITE)], capture_output=True, text=True, timeout=1200,
                                 env=dict(os.environ, ALPHA_NEXT_RECEIPTS_ENGINE=d, PYTHONDONTWRITEBYTECODE='1'))
        output = process.stdout + process.stderr
        imported = re.search(r'^native_measurement under test: (.*)$', output, re.MULTILINE)
        failed = sorted(set(re.findall(r'^(?:FAIL|ERROR): (\w+) \(__main__\.(\w+)\.', output, re.MULTILINE)))
        row = dict(mutant=name, returncode=process.returncode, imported_from_mutant_dir=bool(
            imported and imported.group(1).startswith(d)), failing_tests=['%s.%s' % (c, t) for t, c in failed],
            summary=re.findall(r'^(?:Ran \d+ tests.*|OK.*|FAILED.*)$', output, re.MULTILINE))
    evidence = os.environ.get('ALPHA_NEXT_RECEIPTS_EVIDENCE')
    if evidence:
        Path(evidence).mkdir(parents=True, exist_ok=True)
        (Path(evidence) / ('mutant-%s.log' % name)).write_text(output)
    return row


class Mutants(unittest.TestCase):
    rows = []

    @classmethod
    def tearDownClass(cls):
        evidence = os.environ.get('ALPHA_NEXT_RECEIPTS_EVIDENCE')
        if evidence:
            (Path(evidence) / 'mutants.json').write_text(json.dumps(cls.rows, indent=1) + '\n')

    def test_unmutated_copy_passes(self):
        row = run('control-unmutated', [])
        self.rows.append(row)
        self.assertTrue(row['imported_from_mutant_dir'], row)
        self.assertEqual((row['returncode'], row['failing_tests']), (0, []), row)

    def test_every_mutant_is_caught(self):
        for name, edits in MUTANTS.items():
            with self.subTest(mutant=name):
                row = run(name, edits)
                self.rows.append(row)
                self.assertTrue(row['imported_from_mutant_dir'], row)
                self.assertNotEqual(row['returncode'], 0, row)
                self.assertTrue(row['failing_tests'], row)


if __name__ == '__main__':
    unittest.main()
