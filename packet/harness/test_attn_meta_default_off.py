"""tools/attn_meta_default_off.py: the proof holds on this tree and every check in it is live (hub, CPU)."""
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import attn_meta_default_off as proof  # noqa: E402

DOC = json.loads((ROOT / 'tools/attn_meta_default_off_rows.json').read_text())


class Holds(unittest.TestCase):
    def test_files_hold_on_this_tree(self):
        problems, rows = proof.files_check(proof.git('rev-parse', 'fbb9123e').decode().strip(), DOC)
        self.assertEqual(problems, [])
        self.assertEqual(sum(r['rows'] for r in rows), len(DOC['rows']))
        self.assertTrue(all(r['projected_equals_base'] for r in rows))

    def test_every_row_says_why(self):
        for row in DOC['rows']:
            self.assertTrue(row['why'].strip(), row['file'])


class Live(unittest.TestCase):
    base = None

    def setUp(self):
        self.base = proof.git('rev-parse', 'fbb9123e').decode().strip()

    def check(self, doc):
        return proof.files_check(self.base, doc)[0]

    def test_a_dropped_row_fails(self):
        for i in range(len(DOC['rows'])):
            doc = copy.deepcopy(DOC)
            del doc['rows'][i]
            with self.subTest(row=i):
                self.assertTrue(self.check(doc))

    def test_a_changed_row_fails(self):
        doc = copy.deepcopy(DOC)
        doc['rows'][0]['base'] += ' '
        self.assertTrue(self.check(doc))
        doc = copy.deepcopy(DOC)
        doc['rows'][1]['current'] = doc['rows'][1]['current'].replace('attn_meta', 'attn_metaX', 1)
        self.assertTrue(self.check(doc))

    def test_added_and_exempt_lists_are_exact(self):
        doc = copy.deepcopy(DOC)
        doc['added'] = doc['added'][:1]
        self.assertTrue(self.check(doc))
        doc = copy.deepcopy(DOC)
        doc['added'].append('nothing.py')
        self.assertTrue(self.check(doc))

    def test_an_undeclared_byte_fails(self):
        real = proof.git

        def git(*args):
            out = real(*args)
            if args[0] == 'show' and args[1].endswith('/lane_chunk.py'):
                return out + b'\n'
            return out
        with mock.patch.object(proof, 'git', git):
            problems = self.check(DOC)
        self.assertEqual(problems, ['patches/s5_engine/lane_chunk.py: undeclared change'])

    def test_failing_refusal_tests_fail_the_proof(self):
        with mock.patch.object(proof, 'REFUSAL_TESTS', ('test_attn_meta_default_off.Live.always_fails',)):
            rc, _ = proof.refusals()
        self.assertNotEqual(rc, 0)

    def always_fails(self):
        self.fail('planted')


if __name__ == '__main__':
    unittest.main()
