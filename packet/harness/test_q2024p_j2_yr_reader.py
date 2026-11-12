#!/usr/bin/env python3
"""Yield-rule packets: the exploratory yr reader (tools/q2024p_j2_yr_reader.py). CPU only.

Synthetic records in fresh mkdtemp directories (left in place, never removed). Native-gap lines keep the V1 field names
(chunk_open: txn, request_key, request_generation, chunk_start, chunk_end, native_end; append_yield_selected: the owner's
request_key/generation and junction chunk; request_admitted: prompt_tokens); the Y1 receipt is the engine's line
(scheduler_hooks.yield_owner_refuses). No engine, torch or box access.
"""
import json
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
for extra in (HERE, HERE.parent / 'tools', HERE.parent / 'patches' / 's5_engine'):
    if extra.is_dir() and str(extra) not in sys.path:
        sys.path.append(str(extra))

import q2024p_j2_yr_reader as reader  # noqa: E402
import test_q2024p_j2_yrs_reader as base  # noqa: E402

T0 = base.T0


def gap(s, event, **fields):
    return base.gap(s, event, **fields)


def owner_chunk(s, key, txn, cs, ce, ne, gen=1):
    return gap(s, 'chunk_open', txn=txn, request_key=key, request_generation=gen, chunk_start=cs, chunk_end=ce, native_end=ne,
               terminal=ce == ne, transaction=txn)


def yield_at(s, owner, frontier, txn, append_key, append_txn, gen=1):
    return [gap(s, 'append_yield_selected', request_key=owner, request_generation=gen, chunk_start=frontier - 2048,
                chunk_end=frontier, txn=txn),
            gap(s + .1, 'append_yield_admitted', request_key=append_key, txn=append_txn),
            owner_chunk(s + .1, append_key, append_txn, 100000, 102000, 102000),
            gap(s + .5, 'append_yield_end', request_key=owner)]


def lane():
    """Fresh owner F (native rows 0..20000): yields at 4096 (left 15904) and 6144 after one owner chunk (left 13856), then
    at 18432 after five more chunks (left 1568: rows-left class); append owner A (native 100..6100, new 6000): one yield."""
    lines = [owner_chunk(1, 'F', 1, 0, 2048, 20000), owner_chunk(2, 'F', 2, 2048, 4096, 20000)]
    lines += yield_at(3, 'F', 4096, 2, 'a1', 3)
    lines += [owner_chunk(4, 'F', 4, 4096, 6144, 20000)]
    lines += yield_at(5, 'F', 6144, 4, 'a2', 5)
    t = 6
    for i, cs in enumerate(range(6144, 18432, 2048)):
        lines.append(owner_chunk(t, 'F', 6 + i, cs, cs + 2048, 20000))
        t += 1
    lines += yield_at(t, 'F', 18432, 11, 'a3', 12)
    lines += [owner_chunk(t + 1, 'F', 13, 18432, 20000, 20000)]
    lines += [owner_chunk(30, 'A', 20, 100, 2148, 6100)]
    lines += yield_at(31, 'A', 2148, 20, 'a4', 21)
    lines.append('%s flash-s5 n3-yield-owner-refused %s\n' % (base.at(32), json.dumps(dict(bulk_uid='A', rows_left=1904))))
    return lines


def turns_file(root, label, turns):
    head = dict(label=label, replay_start_s=T0 - 1, replay_config=dict(DURATION=480.0))
    (root / ('arm-' + label + '.json.turns.jsonl')).write_text('\n'.join(json.dumps(x) for x in [head] + turns) + '\n')


def record(lines, y1='0', y2='0', with_block=False, extra_turns=()):
    rec = base.real_record()
    arms = {label: base.arm(label, rec, rec) for label in reader.N3}
    if with_block:
        for label in reader.N3:
            for row, n in zip(arms[label]['statuses'], (0, 2)):
                row['native_admission'] = dict(queue_policy=dict(no_append_owner=dict(junctions=n, bulks=n // 2, receipts=n,
                                                                                      receipts_dropped=0, errors=0)))
    root = base.record(arms, lines)
    for phase in ('start', 'end'):
        path = root / ('engine.identity-%s.json' % phase)
        doc = json.loads(path.read_text())
        doc['identity']['env'].update({reader.Y1_ENV: y1, reader.Y2_ENV: y2})
        path.write_text(json.dumps(doc))
    turns_file(root, 'n3.g2-live-0', list(extra_turns))
    return root


class Yields(unittest.TestCase):
    def test_classes_follow_ons_and_lane_order_cut(self):
        result = reader.collect(record(lane()), 'off')
        y = result['arms']['n3.g2-live-0']['yield_rules']
        self.assertEqual(y['yields'], 4)
        self.assertEqual(y['classes'], {'owner-new<=max_new': 1, 'owner-new>max_new': 3,
                                        'rows-left<=max_new': 2, 'rows-left>max_new': 2})
        self.assertEqual(y['follow_on'], 2)
        self.assertEqual(y['owner_chunks_between'], {'1': 1, '6': 1})
        self.assertEqual(y['follow_on_after_one_owner_chunk'], 1)
        self.assertEqual(y['appends_per_window'], {'1': 4})
        self.assertIsNone(y['in_window_gap_ms'])
        self.assertEqual(result['arms']['n3.g2-live-0']['y1_receipts'], 1)
        self.assertIn('off-record-carries-y1:n3.g2-live-0', result['notes'])

    def test_y1_record_reads_its_block_and_identity(self):
        result = reader.collect(record(lane(), y1='1', with_block=True), 'y1')
        self.assertEqual(result['arms']['n3.g2-live-1']['no_append_owner'],
                         dict(junctions=2, bulks=1, receipts=2, receipts_dropped=0, errors=0))
        self.assertFalse([n for n in result['notes'] if 'y1' in n or 'YIELD' in n], result['notes'])
        missing = reader.collect(record(lane(), y1='0'), 'y1')
        self.assertIn('y1-record-lacks-no_append_owner:n3.g2-live-0', missing['notes'])
        self.assertTrue(any(n.startswith('identity-FLASH_S5_YIELD_NO_APPEND_OWNER-y1') for n in missing['notes']))


class Turns(unittest.TestCase):
    def test_join_pre_open_ttft_and_unplaced(self):
        lines = [gap(2, 'request_admitted', request_key='k1', prompt_tokens=5000),
                 owner_chunk(4, 'k1', 1, 0, 2048, 4999),
                 gap(10, 'request_admitted', request_key='k2', prompt_tokens=90000),
                 owner_chunk(15, 'k2', 2, 0, 2048, 89999)]
        turns = [dict(t=2.5, kind='append', prompt_tokens=5000, ttft=2.0),     # sent T0+1.5, open T0+4: 2.5 s
                 dict(t=10.0, kind='fresh', prompt_tokens=90000, ttft=9.0),    # sent T0+9, open T0+15: 6 s
                 dict(t=20.0, kind='append', prompt_tokens=777, ttft=4.0)]     # no admitted record: unplaced
        j = reader.collect(record(lines, extra_turns=turns), 'off')['arms']['n3.g2-live-0']['turn_join']
        self.assertAlmostEqual(j['append_preopen_agent_s'], 2.5, places=2)
        self.assertAlmostEqual(j['fresh_preopen_agent_s'], 6.0, places=2)
        self.assertEqual((j['append_ttft_mean_s'], j['fresh_ttft_max_s'], j['unplaced'], j['stock'], j['turns']),
                         (3.0, 9.0, 1, 0, 3))

    def test_render_has_no_verdict_words(self):
        text = reader.render([reader.collect(record(lane()), 'off')])
        tail = text.split('-- yield rules', 1)[1]
        for word in ('PASS', 'FAIL', 'GO', 'verdict='):
            self.assertNotIn(word, tail)


if __name__ == '__main__':
    unittest.main()
