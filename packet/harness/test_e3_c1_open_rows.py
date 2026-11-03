"""e3_c1_open_rows on its own: ``progress-line-open-rows-carried`` (each check and its named reason, line order, the stage-overlap
index) and ``ring_increments`` (running total to per-chunk values). The grade's use of both is in test_alpha_next_c1b_queue.
Pure functions; CPU only; no pass here is box evidence."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e3_c1_open_rows as rule  # noqa: E402


def flush(line, txn, open_rows, rows):
    return dict(line=line, identity=dict(txn=txn, open_rows=open_rows), rows=dict(n=rows))


def row(seq, submit, apply):
    return dict(seq=seq, t_submit=submit, t_apply=apply)


# txn -> (its stage-overlap sequences, the successor its line names)
INDEX = {1: (frozenset({1, 2}), 2), 2: (frozenset({3, 4, 5}), 3), 3: (frozenset({6}), None)}


def good():
    """txn 1 flushes with one open row: seq 4 of txn 2, submitted (18) before txn 1's last apply (20)."""
    return [flush(10, 1, 1, [row(1, 0, 10), row(2, 11, 20), row(3, 15, 18)]),
            flush(11, 2, 0, [row(4, 18, 30), row(5, 31, 40)]),
            flush(12, 3, 0, [row(6, 41, 50)])]


class Rule(unittest.TestCase):
    def test_names(self):
        self.assertEqual((rule.RULE, rule.SUPERSEDES), ('progress-line-open-rows-carried', 'progress-line-with-open-rows'))
        self.assertEqual(rule.ENGINE_REASONS, ('c1-open-row-not-successor',))
        self.assertTrue(set(rule.ENGINE_REASONS) <= set(rule.REASONS))

    def test_a_carried_successor_row_passes_and_is_recorded(self):
        out = rule.check_arm(good(), INDEX)
        self.assertEqual((out['rule'], out['reasons'], out['checked']), (rule.RULE, [], 3))
        (cell,) = out['flagged']
        self.assertEqual((cell['line'], cell['txn'], cell['open_rows'], cell['next_txn'], cell['carried_seqs'], cell['reasons']),
                         (10, 1, 1, 2, [4], []))

    def test_no_open_rows_flags_nothing(self):
        rows = good()
        rows[0]['identity']['open_rows'] = 0
        out = rule.check_arm(rows, INDEX)
        self.assertEqual((out['reasons'], out['flagged']), ([], []))

    def test_flushes_are_taken_in_line_order(self):
        self.assertEqual(rule.check_arm(list(reversed(good())), INDEX)['reasons'], [])

    def test_check_1_open_rows_at_the_last_line(self):
        rows = good()
        rows[2]['identity']['open_rows'] = 1
        self.assertEqual(rule.check_arm(rows, INDEX)['reasons'], ['c1-open-rows-at-last-line'])

    def test_check_2_next_flush_not_txn_plus_one_or_not_named(self):
        self.assertEqual(rule.check_arm([good()[0], good()[2]], INDEX)['reasons'], ['c1-open-rows-successor-mismatch'])
        named = {**INDEX, 1: (INDEX[1][0], 5)}
        self.assertEqual(rule.check_arm(good(), named)['reasons'], ['c1-open-rows-successor-mismatch'])
        unnamed = {k: v for k, v in INDEX.items() if k != 1}
        self.assertEqual(rule.check_arm(good(), unnamed)['reasons'], ['c1-open-rows-successor-mismatch'])
        rows = good()
        rows[0]['identity']['txn'] = '1'
        self.assertEqual(rule.check_arm(rows, INDEX)['reasons'], ['c1-open-rows-successor-mismatch'])

    def test_check_3_carried_count_differs_or_no_apply(self):
        rows = good()
        rows[0]['identity']['open_rows'] = 2
        self.assertEqual(rule.check_arm(rows, INDEX)['reasons'], ['c1-open-rows-not-carried'])
        rows = good()
        rows[1]['rows']['n'][0]['t_submit'] = 20         # submitted at the commit's own apply, not before: not carried
        self.assertEqual(rule.check_arm(rows, INDEX)['reasons'], ['c1-open-rows-not-carried'])
        rows = good()
        for r in rows[0]['rows']['n']:
            r['t_apply'] = None
        self.assertEqual(rule.check_arm(rows, INDEX)['reasons'], ['c1-open-rows-not-carried'])

    def test_check_4_carried_row_not_the_successors_is_the_engine_reason(self):
        rows = good()
        rows[1]['rows']['n'][0]['seq'] = 2                # the flushing txn's own sequence
        self.assertEqual(rule.check_arm(rows, INDEX)['reasons'], ['c1-open-row-not-successor'])
        rows = good()
        rows[1]['rows']['n'][0]['seq'] = 99               # in neither txn's overlap line
        self.assertEqual(rule.check_arm(rows, INDEX)['reasons'], ['c1-open-row-not-successor'])

    def test_unreadable_open_rows(self):
        for value in (None, '1', 1.0, True):
            rows = good()
            rows[0]['identity']['open_rows'] = value
            with self.subTest(value=value):
                self.assertEqual(rule.check_arm(rows, INDEX)['reasons'], ['c1-open-rows-unreadable'])

    def test_overlap_index_collects_sequences_and_the_last_successor(self):
        lines = ['a', 'b', 'skip', 'c']
        parsed = {'a': (1, [1], 2), 'b': (1, [2], 2), 'c': (2, [3, 4], None)}
        out = rule.overlap_index(lines, lambda text: parsed.get(text))
        self.assertEqual(out, {1: (frozenset({1, 2}), 2), 2: (frozenset({3, 4}), None)})


class Ring(unittest.TestCase):
    def test_rises_resets_and_skips(self):
        self.assertEqual(rule.ring_increments([0, 5, 5, 12, None, 3, 4]),
                         dict(per_chunk=[0, 5, 0, 7, 3, 1], total=16, max_chunk=7, resets=1))

    def test_a_running_total_sums_to_its_last_value(self):
        totals = [3, 10, 10, 25, 40]
        out = rule.ring_increments(totals)
        self.assertEqual((out['total'], out['max_chunk'], out['resets']), (40, 15, 0))

    def test_empty(self):
        self.assertEqual(rule.ring_increments([None, 'x']), dict(per_chunk=[], total=0, max_chunk=None, resets=0))


if __name__ == '__main__':
    unittest.main()
