#!/usr/bin/env python3
"""q18a-attn-meta box on the c1b base (harness/e3_attn_meta_box_b.py): c1b's open-rows rule and ring read bound for one grade over attn-meta's
table, on overlap arms only, restored afterwards; check 4 keeps INC ENGINE; the queue's core grade through it. CPU only."""
import unittest
from unittest import mock

import e3_alpha_next_c1b_box as c1b
import e3_attn_meta_box as am
import e3_attn_meta_box_b as box
import test_attn_meta_box as tbox

T = am.T
P = T.label('defer', 1, 'live', 0)
R = T.label('off', 1, 'live', 0)


def fake_grade(verdict='PASS', fill=None, seen=None):
    def grade(arms, ingress, checks, logs, watches=None, part='core', cross_run=None):
        if seen is not None:
            seen.update(receipts=box.progress.progress_receipts, ring=box.an.ring_read, table=c1b.T)
        if fill:
            c1b._C1B.setdefault('open_rows', {}).update(fill)
        return dict(verdict=verdict, classification=['ENGINE'] if verdict == 'FAIL' else None, failures=[], missing=[])
    return grade


class Binding(unittest.TestCase):
    def test_bound_for_one_grade_and_restored(self):
        before = (box.progress.progress_receipts, box.an.ring_read, c1b.T)
        seen = {}
        with mock.patch.object(box, 'AM_GRADE', fake_grade(seen=seen)):
            result = box.grade([], {}, {}, [])
        self.assertIs(seen['receipts'], c1b.progress_receipts)
        self.assertIs(seen['ring'], c1b.ring_read)
        self.assertIs(seen['table'], T)
        self.assertEqual((box.progress.progress_receipts, box.an.ring_read, c1b.T), before)
        self.assertEqual(result['open_rows_rule']['rule'], 'progress-line-open-rows-carried')
        self.assertEqual(result['open_rows_rule']['supersedes'], 'progress-line-with-open-rows')

    def test_restored_when_the_grade_raises(self):
        before = (box.progress.progress_receipts, box.an.ring_read, c1b.T)
        with mock.patch.object(box, 'AM_GRADE', side_effect=ValueError('x')):
            with self.assertRaises(ValueError):
                box.grade([], {}, {}, [])
        self.assertEqual((box.progress.progress_receipts, box.an.ring_read, c1b.T), before)
        self.assertEqual(c1b._C1B, {})

    def test_main_binds_grade_and_report_on_am_and_restores(self):
        seen = {}

        def main(argv):
            seen.update(grade=am.grade, report=am.report, argv=argv)
            return 7
        with mock.patch.object(am, 'main', main):
            self.assertEqual(box.main(['x']), 7)
        self.assertIs(seen['grade'], box.grade)
        self.assertIs(seen['report'], box.report)
        self.assertIs(am.grade, box.AM_GRADE)
        self.assertIs(am.report, box.AM_REPORT)


class OpenRows(unittest.TestCase):
    def receipts(self, labels, facts):
        arms = [dict(label=x) for x in labels]
        raw = dict(arms={x: dict(reasons=[box.rule.SUPERSEDES, 'other']) for x in labels})
        with mock.patch.object(c1b, 'RAW_PROGRESS_RECEIPTS', return_value=raw), mock.patch.object(c1b, 'RAW_ASSEMBLE', return_value=([], {})), \
                mock.patch.object(c1b.rule, 'check_arm', return_value=facts), mock.patch.dict(c1b.c1._RUN, {'overlap': []}), \
                box._bound(c1b, dict(T=T)):
            return c1b.progress_receipts(arms, [], [])

    def test_only_this_tables_overlap_arms_take_the_new_rule(self):
        self.assertTrue(T.by_label()[P]['overlap'])
        self.assertFalse(T.by_label()[R]['overlap'])
        c1b._C1B.clear()
        self.addCleanup(c1b._C1B.clear)
        out = self.receipts([P, R], dict(rule=box.rule.RULE, reasons=[], flagged=[], checked=4))
        self.assertEqual(out['arms'][P]['reasons'], ['other'])
        self.assertEqual(out['arms'][R]['reasons'], [box.rule.SUPERSEDES, 'other'])    # C1-off: the old rule, as it is
        self.assertEqual(sorted(c1b._C1B['open_rows']), [P])
        self.assertIs(c1b._C1B['open_rows'][P]['superseded_reason_present'], True)

    def test_a_decode_arm_label_is_not_an_overlap_arm(self):
        self.assertFalse(c1b._overlap.__globals__['T'].by_label().get('decode-am0'))
        with box._bound(c1b, dict(T=T)):
            self.assertFalse(c1b._overlap('decode-am0'))
            self.assertTrue(c1b._overlap(P))


class Classification(unittest.TestCase):
    ENGINE = dict(reasons=['c1-open-row-not-successor'], flagged=[{}], checked=1)
    PLAIN = dict(reasons=['c1-open-rows-not-carried'], flagged=[{}], checked=1)

    def run_grade(self, verdict, facts):
        with mock.patch.object(box, 'AM_GRADE', fake_grade(verdict, {P: facts})):
            return box.grade([], {}, {}, [])

    def test_check_4_is_inc_engine_never_fail(self):
        r = self.run_grade('INC', self.ENGINE)
        self.assertEqual((r['verdict'], r['classification'], r['open_rows_rule']['engine_classified_arms']), ('INC', ['ENGINE'], [P]))

    def test_the_other_reasons_stay_a_plain_inc(self):
        r = self.run_grade('INC', self.PLAIN)
        self.assertEqual((r['verdict'], r['classification'], r['open_rows_rule']['engine_classified_arms']), ('INC', None, []))

    def test_a_fail_keeps_its_own_classification(self):
        r = self.run_grade('FAIL', self.ENGINE)
        self.assertEqual((r['verdict'], r['classification']), ('FAIL', ['ENGINE']))

    def test_the_report_lists_the_open_rows(self):
        with mock.patch.object(box, 'AM_REPORT', return_value='base'):
            text = box.report(dict(open_rows_rule=dict(arms={P: self.ENGINE})))
        self.assertTrue(text.startswith('base'))
        self.assertIn('%s: flagged 1 of 1' % P, text)
        with mock.patch.object(box, 'AM_REPORT', return_value='base'):
            self.assertEqual(box.report({}), 'base')


class CoreGrade(tbox.Grade):
    """The amendment A grade's own cases and mutants, through this module: am.grade seen by those tests is the _b one, whose AM_GRADE
    is the real am.grade (a mutant patched on am still bites)."""

    def setUp(self):
        super().setUp()
        patcher = mock.patch.object(am, 'grade', box.grade)
        patcher.start()
        self.addCleanup(patcher.stop)


if __name__ == '__main__':
    unittest.main()
