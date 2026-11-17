#!/usr/bin/env python3
"""harness/stage1_step_robust_swap.py (lead 3 Nov 13:3x, option A with conditions): the map names exactly the declared four ids,
each found exactly once; a fifth id (mutant) is refused; the base-tree receipt and the default-off proof carry the claim (missing
-> INC, not holding -> FAIL); end to end over c1's two modules with a proving receipt the remaining 18 tests run green.
CPU only. The base-tree proof itself (tools/stage1_step_base_proof.py) is a chain step; its receipt here is written by the test
from the fields the tool writes (the tool's own run is in the chain and in selftest)."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import stage1_step_robust_swap as swap


def receipt(**changes):
    r = dict(schema=swap.BASE_SCHEMA, base=swap.BASE, rc=0, tests_run=22, expected_tests=22, skipped=0, passed=True,
             checkout_kind='shared-clone-detached')
    r.update(changes)
    return r


class Map(unittest.TestCase):
    def test_exactly_the_four(self):
        swap.check_map()
        self.assertEqual(sorted(swap.SUPERSEDED), sorted(swap._LEAD_FOUR))
        self.assertEqual(swap.MODULES, ['test_alpha_next_c1_pack', 'test_alpha_next_c1_rebind_default_off'])

    def test_mutant_fifth_id_refused(self):
        fifth = dict(swap.SUPERSEDED, **{'test_alpha_next_c1_pack.Delivery.test_the_declared_leaves_name_c1s_and_the_composition_fixs':
                                         'mutant'})
        with self.assertRaisesRegex(swap.Refused, 'extra'):
            swap.check_map(fifth)
        with mock.patch.object(swap, 'SUPERSEDED', fifth):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                rc = swap.main(['test_alpha_next_c1_rebind_default_off'])
        self.assertEqual(rc, 1)
        self.assertIn('refused', out.getvalue())

    def test_dropped_or_reasonless_refused(self):
        three = dict(list(swap.SUPERSEDED.items())[:3])
        with self.assertRaisesRegex(swap.Refused, 'missing'):
            swap.check_map(three)
        with self.assertRaisesRegex(swap.Refused, 'reason'):
            swap.check_map(dict(swap.SUPERSEDED, **{swap._LEAD_FOUR[0]: ''}))

    def test_an_id_that_does_not_exist_is_refused(self):
        suite = unittest.defaultTestLoader.loadTestsFromName('test_alpha_next_c1_rebind_default_off')
        renamed = {'test_alpha_next_c1_rebind_default_off.Holds.test_renamed': 'x'}
        with self.assertRaisesRegex(swap.Refused, 'not found exactly once'):
            swap.filtered(suite, renamed)
        kept, removed = swap.filtered(suite)
        self.assertEqual(removed, [swap._LEAD_FOUR[0]])


class Evidence(unittest.TestCase):
    def write(self, tmp, data):
        p = Path(tmp) / 'r.json'
        p.write_text(json.dumps(data) if isinstance(data, dict) else data)
        return str(p)

    def test_receipt_states(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(swap.base_receipt(None)[0], 'missing')
            self.assertEqual(swap.base_receipt(str(Path(tmp) / 'none.json'))[0], 'missing')
            self.assertEqual(swap.base_receipt(self.write(tmp, receipt()))[0], 'ok')
            for change in (dict(rc=1), dict(tests_run=21), dict(skipped=1), dict(passed=False), dict(base='0' * 40),
                           dict(schema='x'), dict(expected_tests=18)):
                with self.subTest(change=change):
                    self.assertEqual(swap.base_receipt(self.write(tmp, receipt(**change)))[0], 'fail')
            self.assertEqual(swap.base_receipt(self.write(tmp, '{not json'))[0], 'fail')

    def run_main(self, rec, off=('ok', {}), modules=('test_alpha_next_c1_rebind_default_off',)):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write(tmp, rec) if rec is not None else None
            out = io.StringIO()
            with mock.patch.object(swap, 'default_off', lambda: off), contextlib.redirect_stdout(out), \
                    contextlib.redirect_stderr(io.StringIO()):
                rc = swap.main((['--base-receipt', path] if path else []) + list(modules))
        return rc, out.getvalue()

    def test_missing_is_inc_and_failing_is_fail(self):
        self.assertEqual(self.run_main(None)[0], 2)
        self.assertEqual(self.run_main(receipt(), off=('missing', {}))[0], 2)
        self.assertEqual(self.run_main(receipt(rc=1))[0], 1)
        self.assertEqual(self.run_main(receipt(), off=('fail', {}))[0], 1)

    def test_end_to_end_the_rest_of_c1s_modules_pass(self):
        rc, out = self.run_main(receipt(), off=swap.default_off(),
                                modules=('test_alpha_next_c1_rebind_default_off', 'test_alpha_next_c1_pack'))
        self.assertEqual(rc, 0, out[-2000:])
        self.assertEqual(out.count('superseded (declared, stage1-step)'), 4)
        self.assertIn('test_stage1_step_default_off added to this run', out)


if __name__ == '__main__':
    unittest.main()
