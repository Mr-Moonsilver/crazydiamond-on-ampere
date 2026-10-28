"""CPU receipt/packaging guards; actual classes execute in the pinned image."""
import ast
import copy
import json
from pathlib import Path
import unittest
import e3_native_transitions_matrix_probe as probe


class Tests(unittest.TestCase):
    def test_numeric_predicate_positive_and_mutation_execute(self):
        result=probe.predicate_selftest()
        self.assertEqual(result['positive']['verdict'],'PASS')
        self.assertEqual(result['mutation']['verdict'],'FAIL')
        self.assertEqual(result['positive']['full_row2_verdict'],'INCONCLUSIVE')
        self.assertFalse(result['box_evidence'])

    def test_missing_or_duplicate_raw_position_cannot_pass(self):
        rows=[dict(stream=i,position=j,top={'token_id:'+str(k):-k*.2 for k in range(20)})
              for i in range(8) for j in range(256)]
        raw={name:copy.deepcopy(rows) for name in ('full-A1','full-A2','full-C','split-B')}
        raw['split-B'][-1]=copy.deepcopy(raw['split-B'][0])
        self.assertEqual(probe.regrade_sampler_rule4({'raw_numeric_rows':raw})['verdict'],'FAIL')

    def test_fixture_declaration_does_not_clear_box_hold(self):
        path=Path(__file__).with_name('native15_transitions_fixture_leaves.json')
        declaration=json.loads(path.read_text())
        self.assertFalse(declaration['matrix_complete'])
        self.assertFalse(declaration['clears_box_hold'])
        for scenario in declaration['scenarios'].values():
            for leaf in scenario['leaves']:
                self.assertTrue(leaf['replaces'])
                self.assertTrue(leaf['equivalent_for'])
                self.assertTrue(leaf['more_convenient_for'])
                self.assertFalse(leaf['box_evidence'])
        self.assertIn('1',declaration['deferred'])

    def test_input_builder_copies_fixture_declaration(self):
        source=(Path(__file__).resolve().parents[1]/'tools/build_native15_matrix_input.py').read_text()
        ast.parse(source)
        self.assertIn("out/'proof/fixture-leaves.json'",source)


if __name__=='__main__':unittest.main()
