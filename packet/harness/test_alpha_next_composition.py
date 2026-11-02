#!/usr/bin/env python3
"""alpha-next composition proof (tools/alpha_next_composition.py) and its negative controls; git only, CPU only."""
import copy
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import alpha_next_composition as comp  # noqa: E402


class Composition(unittest.TestCase):
    def test_the_committed_composition_proves(self):
        record = comp.prove()
        self.assertEqual(record['verdict'], 'PASS')
        self.assertEqual([s['kind'] for s in record['steps']], [k for k, _, _ in comp.STEPS])

    def test_the_committed_declaration_is_the_draft(self):
        self.assertEqual(json.loads(comp.DECLARATION.read_text()), comp.draft())

    def test_a_changed_resolution_refuses(self):
        declaration = json.loads(comp.DECLARATION.read_text())
        bad = copy.deepcopy(declaration)
        bad['blocks'][0]['text'] += '# extra\n'
        with mock.patch.object(comp.Path, 'read_text', lambda self, *a, **k: json.dumps(bad)), \
                self.assertRaisesRegex(AssertionError, 'undeclared resolution'):
            comp.prove()

    def test_a_missing_block_refuses(self):
        declaration = json.loads(comp.DECLARATION.read_text())
        bad = dict(declaration, blocks=declaration['blocks'][1:])
        with mock.patch.object(comp.Path, 'read_text', lambda self, *a, **k: json.dumps(bad)), \
                self.assertRaises(AssertionError):
            comp.prove()

    def test_a_pick_against_another_original_refuses(self):
        with self.assertRaisesRegex(AssertionError, 'differs'):
            comp.check_pick('93729571', '01e6b369')

    def test_a_clean_path_changed_after_the_merge_refuses(self):
        real = comp.tree_blobs

        def tampered(tree):
            rows = real(tree)
            if tree == '5990af80':
                rows['harness/selftest.py'] = '0' * 40
            return rows
        with mock.patch.object(comp, 'tree_blobs', tampered), self.assertRaises(AssertionError):
            comp.check_merge('5990af80', 'c90a7f8e', json.loads(comp.DECLARATION.read_text())['blocks'])

    def test_text_outside_a_block_changed_refuses(self):
        marked = 'a\n<<<<<<< x\nb\n=======\nc\n>>>>>>> y\nd\n'
        self.assertEqual(comp.resolutions_of(marked, 'a\nbc\nd\n'), ['bc\n'])
        with self.assertRaises(AssertionError):
            comp.resolutions_of(marked, 'A\nbc\nd\n')


if __name__ == '__main__':
    unittest.main()
