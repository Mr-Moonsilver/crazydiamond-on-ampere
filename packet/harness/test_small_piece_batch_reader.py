#!/usr/bin/env python3
"""Host-gap batch pair: the status-row persistence helper and the exploratory off/on reader (tools/q2024p_j2_hgb_reader.py).

Pure CPU, synthetic records written to fresh mkdtemp directories (left in place, never removed). The stage-local receipt
line format is the frozen stage_progress.StagedSegmentDriver._stage_line's, copied from a real j2 engine log line; the
status-block shape is small_piece_batch._new_counters'. No engine, torch or box access.
"""
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
for extra in (HERE.parent / 'tools', HERE.parent / 'n3tools', HERE.parent / 'patches' / 's5_engine'):
    if extra.is_dir() and str(extra) not in sys.path:
        sys.path.append(str(extra))

import small_piece_batch_record as record  # noqa: E402
import q2024p_j2_hgb_reader as reader  # noqa: E402

LINE = ('2026-11-11T22:24:43.648015498Z flash-s5 stage-local txn=1 chunk=0-2048 commands=33,21 both_stages_ns=223526459 '
        'first_ns=483506633672026,483506634413416 last_ns=483507306461732,483507320796305 '
        'local_release_ns=483507036382791,483507320819606 headroom_refused=22 headroom_blocked_ns=37575569 pid=85 seq=68 '
        'sha=b4195964451bd23c\n')


def stage_line(at, start, end, first, last):
    return ('%s flash-s5 stage-local txn=9 chunk=%d-%d commands=12,12 both_stages_ns=1 first_ns=%s,%s last_ns=%s,%s '
            'local_release_ns=None,None headroom_refused=0 headroom_blocked_ns=0 pid=1 seq=1 sha=x\n'
            % (at, start, end, first[0], first[1], last[0], last[1]))


def block(**changes):
    import small_piece_batch as spb
    value = dict(schema=spb.STATUS_SCHEMA, installed=True, rows_max=spb.SMALL_PIECE_ROWS, batch_max=spb.BATCH_MAX,
                 outstanding=0, **spb._new_counters())
    value.update(changes)
    return value


class Persist(unittest.TestCase):
    def test_on_reply_keeps_the_engine_block_unchanged(self):
        status = block(sent=3)
        row = {'label': 'after'}
        self.assertIs(record.persist(row, {'route': 'full', 'small_piece_batch': status}), row)
        self.assertIs(row['small_piece_batch'], status)

    def test_off_reply_leaves_the_row_shape_untouched(self):
        for reply in ({'route': 'full'}, None, [], 'status'):
            row = {'label': 'before', 'route': 'full'}
            self.assertEqual(record.persist(row, reply), {'label': 'before', 'route': 'full'})
            self.assertNotIn('small_piece_batch', row)


class Reader(unittest.TestCase):
    def test_real_stage_line_parses(self):
        row = reader.parse_stage_line(LINE)
        self.assertEqual(row['rows'], 2048)
        self.assertEqual(row['commands'], [33, 21])
        self.assertEqual(row['stage_ns'], [483507306461732 - 483506633672026, 483507320796305 - 483506634413416])
        self.assertAlmostEqual(row['at_s'], 1794435883.648015, places=5)
        self.assertIsNone(reader.parse_stage_line('flash-s5 stage-local nonsense'))

    def test_small_pieces_only_inside_the_arm_window(self):
        lines = [stage_line('1970-01-01T00:00:10.000000000Z', 0, 176, (0, 0), (1000000, 3000000)),
                 stage_line('1970-01-01T00:00:11.000000000Z', 176, 2224, (0, 0), (9000000, 9000000)),
                 stage_line('1970-01-01T00:00:12.000000000Z', 0, 64, (0, 'None'), (5000000, 'None')),
                 stage_line('1970-01-01T00:00:30.000000000Z', 0, 64, (0, 0), (7000000, 7000000)),
                 'unrelated line\n']
        out = reader.stage_pieces(io.StringIO(''.join(lines)), {'n3': (9.0, 20.0)})['n3']
        self.assertEqual((out['pieces'], out['small_pieces']), (3, 2))
        self.assertEqual(out['small_stage']['n'], 3)
        self.assertEqual(out['small_stage']['mean_ms'], 3.0)
        self.assertEqual(out['small_stage0'], dict(n=2, mean_ms=3.0, p50_ms=1.0, p90_ms=5.0))
        self.assertEqual(out['small_stage1']['n'], 1)

    def test_block_shape_is_the_engine_counters(self):
        counters = block()
        for name in reader.COUNTERS + reader.MAPS + ('submit_ns', 'send_to_apply_ns', 'worker_host_ns', 'piece_stage_ns'):
            self.assertIn(name, counters)

    def test_batch_delta_is_after_minus_before(self):
        before = block(sent=2, commands_sent=4, second_applied=1, submit_ns=[2000000, 2], worker_host_ns=[10, 1],
                       send_to_apply_ns=[4000000, 3000000, 2])
        before['refused'] = dict(before['refused'], rows=0, last_layer=5)
        before['worker_stop'] = dict(before['worker_stop'], complete=1, ring=1)
        after = block(sent=12, commands_sent=24, second_applied=7, second_not_run=2, partial=1, failures=0, outstanding=0,
                      submit_ns=[12000000, 12], worker_host_ns=[110, 11], send_to_apply_ns=[24000000, 5000000, 12])
        after['refused'] = dict(after['refused'], last_layer=9, next_not_layer=1)
        after['worker_stop'] = dict(after['worker_stop'], complete=7, ring=3, run_ahead=2)
        after['piece_stage_ns'] = dict(batched=[30000000, 3], unbatched=[8000000, 2])
        before['piece_stage_ns'] = dict(batched=[10000000, 1], unbatched=[0, 0])
        delta = reader.batch_delta(before, after)
        self.assertEqual(delta['sent'], 10)
        self.assertEqual(delta['commands_per_submission'], 2.0)
        self.assertEqual(delta['both_ran_without_yield'], 6)
        self.assertEqual(delta['second_not_run'], 2)
        self.assertEqual(delta['failures'], 0)
        self.assertEqual(delta['refused'], dict(last_layer=4, next_not_layer=1))
        self.assertEqual(delta['worker_stop'], dict(complete=6, ring=2, run_ahead=2))
        self.assertEqual(delta['send_ms_per_submission'], 1.0)
        self.assertEqual(delta['worker_host_ms_per_command'], 0.00001)
        self.assertEqual(delta['send_to_apply_ms_mean'], 2.0)
        self.assertEqual(delta['piece_stage'], dict(batched=dict(n=2, mean_ms=10.0), unbatched=dict(n=2, mean_ms=4.0)))
        self.assertIsNone(reader.batch_delta(None, after))
        self.assertIsNone(reader.batch_delta(before, None))

    def test_token_gaps(self):
        root = Path(tempfile.mkdtemp(prefix='hgb-reader-gaps-'))
        path = root / 'turns.jsonl'
        path.write_text(json.dumps({'header': True}) + '\n' + json.dumps({'arrivals_ms': [0, 10, 30, 60]}) + '\n'
                        + json.dumps({'arrivals_ms': [5]}) + '\nnot json\n' + json.dumps({'arrivals_ms': [100, 200]}) + '\n')
        self.assertEqual(reader.token_gaps(path), dict(n=4, p50_ms=20, p95_ms=100, p99_ms=100))
        self.assertIsNone(reader.token_gaps(root / 'absent.jsonl'))


def write_record(root, variant, with_block):
    root.mkdir(parents=True)
    (root / 'compose5-grade.json').write_text(json.dumps(dict(
        verdict='PASS', numerics_o3=dict(verdict='WITHIN'),
        coverage={label: dict(live_native_share=1.0, all_native_share=1.0, live_native_share_strict=0.9989,
                              all_native_share_strict=0.999) for label in reader.ARMS if label.startswith('n3.')})))
    (root / 'queue-grade.json').write_text(json.dumps(dict(verdict='PASS', rc=0, engine_crashed=False)))
    turns = dict(zip(reader.ARMS, (100, 110, 120, 96)))
    log = []
    for index, label in enumerate(reader.ARMS):
        start = 1000.0 + 600 * index
        statuses = []
        for tag, sent in (('entry', 0), ('before', 1), ('after', 9)):
            row = dict(label=tag, route='full')
            if with_block:
                record.persist(row, dict(small_piece_batch=block(sent=sent, commands_sent=2 * sent, second_applied=sent - 1)))
            statuses.append(row)
        (root / ('arm-' + label + '.json')).write_text(json.dumps(dict(
            label=label, complete=True, reasons=[], statuses=statuses, replay_start_s=start, replay_end_s=start + 480,
            replay_config=dict(DURATION=480.0, WARMUP=0.0),
            replay_summary=dict(warm_turns=turns[label] + 2, turns_finished_after_window=2))))
        (root / ('arm-' + label + '.json.turns.jsonl')).write_text(json.dumps(dict(arrivals_ms=[0, 20, 40])) + '\n')
        if label.startswith('n3.'):
            stamp = '1970-01-01T00:%02d:%02d.000000000Z' % divmod(int(start) + 10, 60)
            log.append(stage_line(stamp, 0, 176, (0, 0), (20000000, 30000000)))
    (root / 'engine.log').write_text(''.join(log))
    (root / 'engine-stream.log').write_text(
        ''.join('x flash-s5 memory point=open rank=%d allocated=1 ooms=0 free=1\n' % rank for rank in range(4)))
    return root


class Collect(unittest.TestCase):
    def test_on_record_reads_every_column(self):
        base = Path(tempfile.mkdtemp(prefix='hgb-reader-on-'))
        result = reader.collect(write_record(base / 'rec', 'on', True), 'on')
        self.assertEqual(result['notes'], [])
        self.assertEqual(result['x_sp'], round(230 / 196, 4))
        self.assertEqual((result['numerics'], result['queue_grade'], result['engine']['oom_max']), ('WITHIN', 'PASS', 0))
        n3 = result['arms']['n3.g2-live-0']
        self.assertEqual(n3['turns_per_hour'], 110 * 3600 / 480)
        self.assertEqual(n3['x_sp'], 1.1)
        self.assertEqual(n3['native_share']['live_native_share_strict'], 0.9989)
        self.assertEqual(n3['small_piece_batch']['sent'], 8)
        self.assertEqual(n3['small_piece_batch']['commands_per_submission'], 2.0)
        self.assertEqual(n3['stage_pieces']['small_stage']['mean_ms'], 25.0)
        self.assertEqual(n3['decode_gaps']['p95_ms'], 20)
        self.assertIsNone(result['arms']['sp.g2-live-0']['native_share'])

    def test_off_and_on_mismatches_are_notes_never_verdicts(self):
        base = Path(tempfile.mkdtemp(prefix='hgb-reader-notes-'))
        off = reader.collect(write_record(base / 'off', 'off', True), 'off')
        self.assertIn('off-record-carries-small_piece_batch:n3.g2-live-0', off['notes'])
        on = reader.collect(write_record(base / 'on', 'on', False), 'on')
        self.assertIn('on-record-lacks-small_piece_batch:sp.g2-live-1', on['notes'])
        clean_off = reader.collect(write_record(base / 'clean-off', 'off', False), 'off')
        self.assertEqual(clean_off['notes'], [])
        self.assertIsNone(clean_off['arms']['n3.g2-live-0']['small_piece_batch'])

    def test_main_writes_a_fresh_json_only(self):
        base = Path(tempfile.mkdtemp(prefix='hgb-reader-main-'))
        off, on = write_record(base / 'off', 'off', False), write_record(base / 'on', 'on', True)
        out = base / 'read.json'
        self.assertEqual(reader.main(['off=' + str(off), 'on=' + str(on), '--json-out', str(out)]), 0)
        data = json.loads(out.read_text())
        self.assertEqual((data['schema'], data['exploratory']), ('q2024p-j2-hgb-read/1', True))
        self.assertEqual([r['variant'] for r in data['results']], ['off', 'on'])
        with self.assertRaises(FileExistsError):
            reader.main(['off=' + str(off), '--json-out', str(out)])
        with self.assertRaises(SystemExit):
            reader.main(['middle=' + str(off)])


if __name__ == '__main__':
    unittest.main()
