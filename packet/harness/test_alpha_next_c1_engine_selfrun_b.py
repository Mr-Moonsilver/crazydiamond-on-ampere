#!/usr/bin/env python3
"""tools/alpha_next_c1_engine_selfrun_b.py and tools/alpha_next_c1_gates_b.sh (HARNESS, chain a1, 3 Nov 03:32 CET).

The engine self-run's judge reads C1's pair fields and early-open counters from the full case records each part's container wrote,
not from the record's case_row summaries: over the same case files the original's judge (on summaries) fails every C1 pair and reads
early_opens None, _b's holds; a missing or differing case file fails closed. run() is the original's source with one exact-once swap
in a copied namespace (the original module untouched); the negative and the comparison are the original's. gates_b differs from
gates by exactly the four selfrun calls and its header. CPU only, temporary files; no container.
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import alpha_next_c1_engine_selfrun as base  # noqa: E402
import alpha_next_c1_engine_selfrun_b as b  # noqa: E402


def write_cases(out, early_opens=31, differ_field=None):
    """Case files for every mode the pairs and the early open read, with equal pair fields; and their case_row summaries."""
    modes = set(base.C1_PAIRS) | {a for a, _f in base.C1_PAIRS.values()} | {base.EARLY_OPEN_CASE}
    rows = {}
    for mode in sorted(modes):
        record = dict(verdict='PASS', request_tokens=[1, 2, 3], native=True, per_rank_segments_sha256='s' * 64, checks=7)
        if mode == base.EARLY_OPEN_CASE:
            record['overlap'] = dict(counters=dict(early_opens=early_opens, successor_frames=early_opens))
        if differ_field and mode in base.C1_PAIRS:
            record[differ_field] = 'other'
        path = Path(out) / base.part_of(mode) / (mode + '.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record))
        rows[mode] = dict(verdict='PASS', checks=7, sha256='x' * 64, canonical_sha256='y' * 64, error=None, error_type=None)
    return rows


class Judge(unittest.TestCase):
    def test_the_original_judge_on_summaries_fails_and_b_holds_on_the_same_files(self):
        with tempfile.TemporaryDirectory() as out:
            rows = write_cases(out)
            # a1's defect, kept visible: summaries carry no pair fields and no counters.
            _pairs, differ = base.c1_pairs(rows)
            self.assertEqual(sorted(differ), sorted(base.C1_PAIRS))
            self.assertEqual((base.c1.early_open(rows)['ok'], base.c1.early_open(rows)['early_opens']), (False, None))
            records = b.connected_records(out, rows)
            _pairs, differ = base.c1_pairs(records)
            self.assertEqual(differ, [])
            early = base.c1.early_open(records)
            self.assertEqual((early['ok'], early['early_opens'], early['successor_frames']), (True, 31, 31))

    def test_fails_closed(self):
        with tempfile.TemporaryDirectory() as out:
            rows = write_cases(out, early_opens=0)
            self.assertFalse(base.c1.early_open(b.connected_records(out, rows))['ok'])
        with tempfile.TemporaryDirectory() as out:
            rows = write_cases(out, differ_field='per_rank_segments_sha256')
            _pairs, differ = base.c1_pairs(b.connected_records(out, rows))
            self.assertEqual(sorted(differ), sorted(m for m, (_a, f) in base.C1_PAIRS.items() if 'per_rank_segments_sha256' in f))
        with tempfile.TemporaryDirectory() as out:
            rows = write_cases(out)
            (Path(out) / base.part_of(base.EARLY_OPEN_CASE) / (base.EARLY_OPEN_CASE + '.json')).unlink()
            records = b.connected_records(out, rows)
            self.assertEqual(records[base.EARLY_OPEN_CASE], {})
            self.assertFalse(base.c1.early_open(records)['ok'])
            self.assertIn(base.EARLY_OPEN_CASE, base.c1_pairs(records)[1])


class Derivation(unittest.TestCase):
    def test_run_is_the_original_with_one_swap_in_a_copied_namespace(self):
        import inspect
        original = inspect.getsource(base.run)
        old, new = b.SWAP
        self.assertEqual(original.count(old), 1)
        self.assertEqual(b.RUN_TEXT, original.replace(old, new))
        self.assertIsNot(b.run, base.run)
        self.assertIs(base.run.__globals__, base.__dict__)
        self.assertIsNot(b.run.__globals__, base.__dict__)
        self.assertNotIn('connected_records', base.__dict__)
        self.assertIs(b.run.__globals__['connected_records'], b.connected_records)
        self.assertIs(b.negative, base.negative)
        self.assertIs(b.compare, base.compare)

    def test_gates_b_differs_by_the_four_selfrun_calls_and_its_header(self):
        a = (ROOT / 'tools/alpha_next_c1_gates.sh').read_text().splitlines()
        g = (ROOT / 'tools/alpha_next_c1_gates_b.sh').read_text().splitlines()
        self.assertEqual(g[0], a[0])
        header = g[1:4]
        self.assertTrue(all(line.startswith('#') for line in header))
        rest = g[4:]
        self.assertEqual(len(rest), len(a) - 1)
        changed = [(x, y) for x, y in zip(a[1:], rest) if x != y]
        self.assertEqual(len(changed), 4)
        for x, y in changed:
            self.assertEqual(x.replace('tools/alpha_next_c1_engine_selfrun.py ', 'tools/alpha_next_c1_engine_selfrun_b.py '), y)
        self.assertEqual(subprocess.run(['bash', '-n', str(ROOT / 'tools/alpha_next_c1_gates_b.sh')]).returncode, 0)
        text = '\n'.join(g)
        self.assertIn('if [[ $ORC == 4 ]]; then', text)       # the bounded negative's INC stop is kept


if __name__ == '__main__':
    unittest.main()
